# Extracted from the executed Kaggle implementation; numerical logic preserved.
import types
from . import repair as I
from pathlib import Path
from datetime import datetime
import io, json, random, hashlib, zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm
from huggingface_hub import HfApi, hf_hub_download
from IPython.display import display, FileLink
FROZEN_BUNDLE = None
UHDM_ROOT = None
WEIGHT = None
HF_REPO = 'Mrduckk/DCID'
HF_REVISION = '51c674aa3f71c4cbbddb5cfcba9fbe3536eb0722'
HF_SUBDIR = 'test/iphone_1'
N_UHDM = 20
N_DCID = 20
SEED = 20261108
DOWNLOAD_LIMIT_MIB = 200
ROOT = Path('/kaggle/working/frozen_moire_test')

def load_bundle():
    if FROZEN_BUNDLE is not None:
        p = Path(FROZEN_BUNDLE)
    else:
        roots = [Path('/kaggle/input'), Path('/kaggle/working')]
        candidates = [p for r in roots if r.exists() for p in r.rglob('moire_frozen_test_models.zip')]
        if not candidates:
            candidates = [p.parent for r in roots if r.exists() for p in r.rglob('frozen_protocol.json')]
        if len(candidates) != 1:
            raise RuntimeError(f'请挂载冻结测试权重包，或明确填写FROZEN_BUNDLE。候选={candidates}')
        p = candidates[0]
    if p.is_dir():
        read = lambda n: (p / n).read_bytes()
    else:
        z = zipfile.ZipFile(p)
        read = z.read
    frozen = json.loads(read('frozen_protocol.json'))
    protocol = json.loads(read('old_fusion/protocol.json'))
    band_data = read('old_fusion/band.pt')
    target_data = read('DCID_100.pt')
    assert hashlib.sha256(band_data).hexdigest() == frozen['band_sha256']
    assert hashlib.sha256(target_data).hexdigest() == frozen['DCID_100_sha256']
    band = torch.load(io.BytesIO(band_data), map_location='cpu', weights_only=True)
    target = torch.load(io.BytesIO(target_data), map_location='cpu', weights_only=True)
    assert target['domain'] == 'DCID' and target['step'] == 100
    return (frozen, protocol, band, target)

def choose_pairs(files, n, seed):
    pairs = [{'input': p[:-7] + '_m.jpg', 'target': p} for p in sorted(files) if p.endswith('_gt.jpg') and p[:-7] + '_m.jpg' in files]
    random.Random(seed).shuffle(pairs)
    if len(pairs) < n:
        raise RuntimeError(f'test目录仅找到{len(pairs)}对，需要{n}对；停止，不回退到train')
    return pairs[:n]

def download_target(out):
    assert HF_SUBDIR.startswith('test/'), '只能使用test池'
    api = HfApi()
    try:
        files = {p.path: int(p.size) for p in api.list_repo_tree(HF_REPO, path_in_repo=HF_SUBDIR, repo_type='dataset', revision=HF_REVISION, recursive=True) if getattr(p, 'size', None) is not None}
    except Exception as error:
        raise RuntimeError(f'无法读取锁定版本的{HF_SUBDIR}。检查Internet或HF访问；不会回退train。原错误：{error}') from error
    selected = choose_pairs(files, N_DCID, SEED)
    size = sum((files[p[k]] for p in selected for k in ['input', 'target']))
    plan = {'repo': HF_REPO, 'revision': HF_REVISION, 'subdir': HF_SUBDIR, 'seed': SEED, 'bytes': size, 'pairs': selected}
    (out / 'DCID_download_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'DCID只下载{len(selected)}对，文件大小合计{size / 1024 ** 2:.1f}MiB')
    if size > DOWNLOAD_LIMIT_MIB * 1024 ** 2:
        raise RuntimeError('超过下载上限，未下载图片。不要根据结果更换抽样。')
    records = []
    for pair in tqdm(selected, desc='按需获取DCID test'):
        paths = {k: hf_hub_download(HF_REPO, filename=pair[k], repo_type='dataset', revision=HF_REVISION, local_dir=str(ROOT / 'target_images')) for k in ['input', 'target']}
        records.append({'domain': 'DCID', 'id': 'DCID/' + pair['target'][:-7], **paths})
    return records

def source_pairs(root):
    gts = sorted((p for p in root.rglob('*gt.jpg') if any((part.lower() in ['test', 'testing', 'testset', 'test_data'] for part in p.relative_to(root).parts[:-1]))))
    rows = []
    for gt in gts:
        x = gt.with_name(gt.name[:4] + '_moire.jpg')
        if not x.is_file():
            raise RuntimeError(f'UHDM test缺少输入：{x}')
        rows.append({'domain': 'UHDM', 'id': 'UHDM/' + gt.relative_to(root).as_posix(), 'input': str(x), 'target': str(gt)})
    if len(rows) < N_UHDM:
        raise RuntimeError(f'UHDM test仅找到{len(rows)}对；请挂载完整含test的UHDM，不回退train')
    random.Random(SEED).shuffle(rows)
    return rows[:N_UHDM]

def audit(records, root, old_protocol, target_ck, out):
    prior = {k: set() for k in ['input_hash', 'gt_hash']}
    for row in tqdm(old_protocol['manifest']['train'] + old_protocol['manifest']['val'], desc='排查旧UHDM精确重复'):
        prior['input_hash'].add(I.S.C.J.H.sha_file(I.S.C.J.source_path(row['x'], root)))
        prior['gt_hash'].add(I.S.C.J.H.sha_file(I.S.C.J.source_path(row['gt'], root)))
    for row in target_ck['config']['signature']['samples']:
        prior['input_hash'].add(row['input_hash'])
        prior['gt_hash'].add(row['gt_hash'])
    seen = {k: set() for k in prior}
    rows = []
    issues = []
    for p in tqdm(records, desc='测试配对和哈希检查'):
        r = dict(p)
        for key, hkey in [('input', 'input_hash'), ('target', 'gt_hash')]:
            h = I.S.C.J.H.sha_file(p[key])
            r[hkey] = h
            if h in prior[hkey]:
                issues.append({'id': p['id'], 'issue': 'exact_duplicate_of_prior_data', 'kind': key})
            if h in seen[hkey]:
                issues.append({'id': p['id'], 'issue': 'exact_duplicate_within_test', 'kind': key})
            seen[hkey].add(h)
        with Image.open(p['input']) as a, Image.open(p['target']) as b:
            a.load()
            b.load()
            r['width'], r['height'] = a.size
            if a.size != b.size or min(a.size) < 768:
                issues.append({'id': p['id'], 'issue': 'invalid_size'})
        if r['input_hash'] == r['gt_hash']:
            issues.append({'id': p['id'], 'issue': 'identical_input_target'})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / 'test_manifest.csv', index=False)
    (out / 'audit.json').write_text(json.dumps({'issues': issues, 'pairs': len(rows), 'limits': 'hashes rule out exact files only, not same content across views. Author test split; scene independence not established.'}, indent=2), encoding='utf-8')
    if issues:
        raise RuntimeError('配对/重复检查未通过；停止，不自动删图重抽，见audit.json')
    return rows

@torch.inference_mode()
def infer_all(records, model, band, repair, out):
    rows = []
    for index, p in enumerate(tqdm(records, desc='冻结方法测试，不训练')):
        with Image.open(p['input']) as im:
            a = np.array(im.convert('RGB'))
        with Image.open(p['target']) as im:
            g = np.array(im.convert('RGB'))
        a, g = I.S.C.J.crop_pair(a, g, p['id'])
        x = I.S.C.J.tensor(a)
        candidates = torch.stack([torch.from_numpy(I.S.C.J.H.predict(model, a, s)).permute(2, 0, 1).cuda() for s in (0.5, 0.75, 1.0)])[None]
        lows = I.S.C.J.fusion_lowpass(candidates[0])[None]
        b = band(x, candidates, lows).clamp(0, 1)
        o = candidates[:, 2]
        o = o.half().float()
        b = b.half().float()
        anchored = (o + (x - o).mean((-2, -1), keepdim=True)).clamp(0, 1)
        preds = {'input': x, 'original': o, 'old_band': b, 'original_mean_anchor': anchored}
        if p['domain'] == 'DCID':
            preds['DCID_spatial100'] = repair(x, o, b)[0].clamp(0, 1)
        gt = g.astype(np.float32) / 255
        for label, value in preds.items():
            image = I.S.C.J.arr(value)
            rows.append({'domain': p['domain'], 'id': p['id'], 'method': label, **I.S.C.J.H.metrics(image, gt)})
            folder = out / 'images' / p['domain'] / f'{index:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(image * 255).astype('uint8')).save(folder / (label + '.png'))
        Image.fromarray(g).save(folder / 'gt.png')
        folder = out / 'previews'
        folder.mkdir(exist_ok=True)
        I.S.C.J.H.preview([(k, I.S.C.J.arr(v)) for k, v in preds.items()] + [('GT', gt)], folder / f'{p['domain']}_{index:02d}.jpg')
        pd.DataFrame(rows).to_csv(out / 'per_image.csv', index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['domain', 'method']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'))
    summary.to_csv(out / 'summary.csv')
    display(summary)
    differences = []
    for domain in ['DCID', 'UHDM']:
        pivot = frame[frame.domain == domain].pivot(index='id', columns='method', values='psnr')
        comparisons = [('old_band', 'original'), ('original_mean_anchor', 'original')]
        if domain == 'DCID':
            comparisons.append(('DCID_spatial100', 'original_mean_anchor'))
        for method, ref in comparisons:
            delta = pivot[method] - pivot[ref]
            differences.append({'domain': domain, 'method': method, 'reference': ref, 'mean_gain': float(delta.mean()), 'median_gain': float(delta.median()), 'wins': int((delta > 0).sum()), 'images': len(delta)})
    pd.DataFrame(differences).to_csv(out / 'paired_comparisons.csv', index=False)
    display(pd.DataFrame(differences))

def run_frozen_test():
    assert torch.cuda.is_available(), '需要GPU做固定推理，没有训练'
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / datetime.now().strftime('frozen_%Y%m%d_%H%M%S')
    out.mkdir()
    try:
        frozen, old_protocol, band_ck, target_ck = load_bundle()
        if UHDM_ROOT is None:
            found = sorted((p for p in Path('/kaggle/input').rglob('uhdm-dataset') if p.is_dir()))
            if len(found) != 1:
                raise RuntimeError('挂载soumikrakshit/uhdm-dataset，或设置UHDM_ROOT')
            source_root = found[0]
        else:
            source_root = Path(UHDM_ROOT)
        records = source_pairs(source_root)
        protocol = {'frozen_models': frozen, 'seed': SEED, 'counts': {'UHDM': N_UHDM, 'DCID': N_DCID}, 'HF_revision': HF_REVISION, 'HF_subdir': HF_SUBDIR, 'region': 768, 'train': False, 'primary_methods': {'UHDM': 'old_band vs original', 'DCID': 'original_mean_anchor vs original'}, 'secondary': 'DCID development-selected fixed100 spatial vs mean anchor; no test-based model selection', 'precision': 'same-view center768; fixed predictions float16 -> float32 as development cache', 'limitations': 'sampled author test subsets, not official full-image score; exact hashes do not prove scene/content independence'}
        (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        pd.DataFrame(records).to_csv(out / 'UHDM_sampling_plan.csv', index=False)
        records += download_target(out)
        records = audit(records, source_root, old_protocol, target_ck, out)
        H = I.S.C.J.H
        H.ROOT = ROOT / 'official'
        H.ROOT.mkdir(exist_ok=True)
        H.LOCAL_WEIGHT = WEIGHT
        if WEIGHT is None:
            known = [Path('/kaggle/working/independent_repair/official/uhdm_checkpoint.pth'), Path('/kaggle/working/color_calibration/official/uhdm_checkpoint.pth'), Path('/kaggle/working/joint_moire_runs/official/uhdm_checkpoint.pth')]
            H.LOCAL_WEIGHT = next((p for p in known if p.is_file()), None)
        model, meta = H.load_official_model()
        if I.S.C.fingerprint(model) != old_protocol['weight_sha256']:
            raise RuntimeError('主干权重与冻结融合不匹配')
        assert target_ck['config']['signature']['meta']['weight_sha256'] == meta['weight_sha256'], 'DCID候选主干权重不一致'
        band = I.S.C.J.ScaleFusion('band', int(band_ck['settings']['preferred'])).cuda().eval()
        band.load_state_dict(band_ck['model'], strict=True)
        band.requires_grad_(False)
        I.GRID = int(target_ck['config']['grid'])
        I.MAX_RESIDUAL = float(target_ck['config']['max_residual'])
        repair = I.IndependentRepair('DCID').cuda().eval()
        repair.load_state_dict(target_ck['model'], strict=True)
        repair.requires_grad_(False)
        protocol['actual_backbone'] = meta
        (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        infer_all(records, model, band, repair, out)
        print('完成：仅冻结方法测试；不根据测试结果重新调参或重抽样。')
    except Exception as error:
        (out / 'error.txt').write_text(repr(error), encoding='utf-8')
        raise
    finally:
        I.S.pack(out)
    return out
