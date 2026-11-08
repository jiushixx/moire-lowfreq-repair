# Extracted from the executed Kaggle implementation; numerical logic preserved.
import types
from . import joint as J
from pathlib import Path
import json, random, hashlib, time, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm
from IPython.display import display, FileLink
DCID_SOURCE = None
FUSION_ZIP = None
UHDM_ROOT = None
WEIGHT = None
REGION = 768
SEED = 2026
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
PRIMARY_ALPHA = 10.0
MAX_SHIFT = 0.25
ROOT = Path('/kaggle/working/color_calibration')

def fingerprint(model):
    digest = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        digest.update(k.encode())
        digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def prepare():
    assert torch.cuda.is_available(), '缓存预测需要GPU；缓存完成后的拟合与评价使用CPU'
    J.REGION = REGION
    J.DCID_SOURCE = DCID_SOURCE
    J.FUSION_ZIP = FUSION_ZIP
    protocol, ck, old_hash = J.load_old_fusion()
    if UHDM_ROOT is None:
        roots = sorted((p for p in Path('/kaggle/input').rglob('uhdm-dataset') if p.is_dir()))
        if len(roots) != 1:
            raise RuntimeError('请挂载 soumikrakshit/uhdm-dataset，或设置 UHDM_ROOT')
        root = roots[0]
    else:
        root = Path(UHDM_ROOT)
    J.H.DATA_SOURCE = DCID_SOURCE
    source = J.H.PairSource(J.H.locate_source())
    rows = sorted(source.rows, key=lambda r: r['sample_id'])
    random.Random(SEED).shuffle(rows)
    assert len(rows) == 20, '本轮固定复用原20对，不能悄悄更换划分'
    target = []
    for i, r in enumerate(tqdm(rows, desc='读取DCID')):
        a, g = J.crop_pair(source.read(r, 'input'), source.read(r, 'target'), r['sample_id'])
        target.append({'id': 'DCID/' + r['sample_id'], 'domain': 'DCID', 'split': 'train' if i < 12 else 'val', 'x': a, 'gt': g, 'gt_hash': hashlib.sha256(source.read_bytes(r, 'target')).hexdigest(), 'input_hash': hashlib.sha256(source.read_bytes(r, 'input')).hexdigest()})
    train = J.load_source(protocol['manifest']['train'][:20], root, 'train')
    val = J.load_source(protocol['manifest']['val'], root, 'val')
    for p in train + val:
        p['domain'] = 'UHDM'
    items = target + train + val
    J.check_split([p for p in items if p['split'] == 'train'], [p for p in items if p['split'] == 'val'])
    if {p['group'] for p in train} & {p['group'] for p in val}:
        raise RuntimeError('UHDM组冲突')
    J.H.ROOT = ROOT / 'official'
    J.H.ROOT.mkdir(parents=True, exist_ok=True)
    J.H.LOCAL_WEIGHT = WEIGHT
    if WEIGHT is None:
        known = [Path('/kaggle/working/joint_moire_runs/official/uhdm_checkpoint.pth'), Path('/kaggle/working/dcid_baseline/uhdm_checkpoint.pth')]
        J.H.LOCAL_WEIGHT = next((p for p in known if p.is_file()), None)
    model, meta = J.H.load_official_model()
    assert fingerprint(model) == protocol['weight_sha256'], '主干权重与旧融合协议不匹配'
    band = J.ScaleFusion('band', int(ck['settings']['preferred'])).cuda().eval()
    band.load_state_dict(ck['model'], strict=True)
    band.requires_grad_(False)
    return (items, model, band, {**meta, 'old_fusion_hash': old_hash})

@torch.inference_mode()
def cache_predictions(items, model, band, meta):
    signature = {'meta': meta, 'region': REGION, 'protocol': 'clipped_float16_original_and_band_v1', 'samples': [{k: p[k] for k in ['id', 'split', 'domain', 'input_hash', 'gt_hash']} for p in items]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / 'cache' / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'signature.json').write_text(json.dumps(signature, indent=2), encoding='utf-8')
    records = []
    for i, p in enumerate(tqdm(items, desc='缓存两种固定输出（可复用）')):
        dest = folder / f'{i:03d}.npz'
        if not dest.is_file():
            x = J.tensor(p['x'])
            c = torch.stack([torch.from_numpy(J.H.predict(model, p['x'], s)).permute(2, 0, 1).cuda() for s in (0.5, 0.75, 1.0)])[None]
            lows = J.fusion_lowpass(c[0])[None]
            old = J.arr(band(x, c, lows))
            original = J.arr(c[:, 2])
            tmp = dest.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez(f, x=p['x'], gt=p['gt'], original=original.astype(np.float16), old_band=old.astype(np.float16))
            tmp.replace(dest)
        records.append({k: v for k, v in p.items() if k not in ['x', 'gt']} | {'cache': str(dest)})
    return (records, signature)

def features(x, b):
    axes = (0, 1)
    return np.concatenate([x.mean(axes), x.std(axes), b.mean(axes), b.std(axes), (x - b).mean(axes), (x - b).std(axes)]).astype(np.float64)

def weighted_fit(X, Y, domains, alpha):
    groups = sorted(set(domains))
    w = np.array([1.0 / (len(groups) * sum((d == g for d in domains))) for g in domains])
    mean = (w[:, None] * X).sum(0)
    std = np.sqrt((w[:, None] * (X - mean) ** 2).sum(0))
    std = np.maximum(std, 0.01)
    A = np.column_stack([(X - mean) / std, np.ones(len(X))])
    penalty = np.eye(A.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    at = torch.from_numpy(A)
    wt = torch.from_numpy(w[:, None])
    yt = torch.from_numpy(Y)
    coeff = torch.linalg.solve(at.T @ (wt * at) + torch.from_numpy(penalty) / len(X), at.T @ (wt * yt)).numpy()
    return {'mean': mean, 'std': std, 'coeff': coeff}

def infer_shift(f, fit):
    row = np.r_[(f - fit['mean']) / fit['std'], 1.0]
    return np.clip((row[:, None] * fit['coeff']).sum(0), -MAX_SHIFT, MAX_SHIFT)

def evaluate(records, fits, domain_means, out):
    rows = []
    for p in tqdm([r for r in records if r['split'] == 'val'], desc='CPU评价'):
        with np.load(p['cache']) as a:
            x = a['x'].astype(np.float32) / 255
            gt = a['gt'].astype(np.float32) / 255
            predictions = {'input': x}
            for base in ['original', 'old_band']:
                b = a[base].astype(np.float32)
                feat = features(x, b)
                predictions[base] = b
                predictions[base + '_zero_shift'] = np.clip(b + np.zeros(3, dtype=np.float32), 0, 1)
                predictions[base + '_mean_anchor'] = np.clip(b + (x - b).mean((0, 1), keepdims=True), 0, 1)
                predictions[base + '_domain_constant'] = np.clip(b + np.clip(domain_means[base, p['domain']], -MAX_SHIFT, MAX_SHIFT), 0, 1)
                for alpha in RIDGE_ALPHAS:
                    predictions[f'{base}_ridge_{alpha:g}'] = np.clip(b + infer_shift(feat, fits[base, alpha]), 0, 1)
        for method, pred in predictions.items():
            rows.append({'domain': p['domain'], 'id': p['id'], 'method': method, **J.H.metrics(pred, gt)})
        i = sum((r['domain'] == p['domain'] and r['split'] == 'val' and (r['id'] < p['id']) for r in records))
        chosen = ['input', 'original', 'original_mean_anchor', f'original_ridge_{PRIMARY_ALPHA:g}', 'old_band', f'old_band_ridge_{PRIMARY_ALPHA:g}']
        for method in chosen:
            dest = out / 'images' / p['domain'] / method
            dest.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(predictions[method] * 255).astype('uint8')).save(dest / f'{i:02d}.png')
        dest = out / 'previews'
        dest.mkdir(exist_ok=True)
        J.H.preview([(k, predictions[k]) for k in chosen] + [('GT', gt)], dest / f'{p['domain']}_{i:02d}.jpg')
        pd.DataFrame(rows).to_csv(out / 'per_image.csv', index=False)
    frame = pd.DataFrame(rows)
    for base in ['original', 'old_band']:
        a = frame[frame.method == base].set_index('id')
        b = frame[frame.method == base + '_zero_shift'].set_index('id')
        assert np.allclose(a.psnr, b.psnr, atol=1e-09), '零修正对照不一致'
    table = frame.groupby(['domain', 'method']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'))
    table.to_csv(out / 'summary.csv')
    display(table)
    return frame

def run_calibration():
    assert PRIMARY_ALPHA in RIDGE_ALPHAS
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / datetime.now().strftime('calibration_%Y%m%d_%H%M%S')
    out.mkdir()
    try:
        items, model, band, meta = prepare()
        records, signature = cache_predictions(items, model, band, meta)
        model.cpu()
        band.cpu()
        del model, band, items
        torch.cuda.empty_cache()
        print('固定预测缓存完成；后续在CPU拟合及评价，不再反向传播主干。')
        pd.DataFrame(records).to_csv(out / 'manifest.csv', index=False)
        settings = {'signature': signature, 'primary_alpha': PRIMARY_ALPHA, 'alphas': RIDGE_ALPHAS, 'max_shift': MAX_SHIFT, 'fit': '18 input/prediction statistics -> 3 global RGB shifts; ridge; equal domain weight; train only', 'gt_label': 'mean(GT - clipped_base) over native center768; preclipping global MSE optimum', 'domain_constant': 'separate train mean shift per known dataset; uses dataset label, only a comparator', 'precision': 'float16 frozen predictions cached then float32 evaluation; not pixel-identical to prior float32 baseline', 'limits': 'same 12/8 DCID image split and 20/10 UHDM development split as previous run; no independent test; scene overlap unknown; alpha10 primary, others development sensitivity; not guaranteed gain'}
        (out / 'protocol.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
        train = [p for p in records if p['split'] == 'train']
        fits = {}
        domain_means = {}
        feature_rows = []
        for base in ['original', 'old_band']:
            X = []
            Y = []
            domains = []
            for p in train:
                with np.load(p['cache']) as a:
                    x = a['x'].astype(np.float64) / 255
                    g = a['gt'].astype(np.float64) / 255
                    b = a[base].astype(np.float64)
                    f = features(x, b)
                    y = (g - b).mean((0, 1))
                X.append(f)
                Y.append(y)
                domains.append(p['domain'])
                feature_rows.append({'id': p['id'], 'domain': p['domain'], 'base': base, **{f'f{i}': v for i, v in enumerate(f)}, **{f'label{i}': v for i, v in enumerate(y)}})
            X, Y = (np.array(X), np.array(Y))
            for d in sorted(set(domains)):
                domain_means[base, d] = Y[np.array(domains) == d].mean(0)
            for alpha in RIDGE_ALPHAS:
                fit = weighted_fit(X, Y, domains, alpha)
                fits[base, alpha] = fit
                folder = out / 'models'
                folder.mkdir(exist_ok=True)
                np.savez(folder / f'{base}_ridge_{alpha:g}.npz', **fit, max_shift=MAX_SHIFT)
        pd.DataFrame(feature_rows).to_csv(out / 'train_statistics.csv', index=False)
        (out / 'domain_constant.json').write_text(json.dumps({f'{b}/{d}': v.tolist() for (b, d), v in domain_means.items()}, indent=2))
        evaluate(records, fits, domain_means, out)
        print('完成。先比较固定alpha10与各自base、mean_anchor；同时检查另一域有无退化，不自动挑验证集最高分。')
    except Exception as error:
        (out / 'error.txt').write_text(repr(error), encoding='utf-8')
        raise
    finally:
        bundle = out.parent / (out.name + '_results.zip')
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        print('结果ZIP：', bundle)
        display(FileLink(str(bundle)))
    return out
