"""Frozen iPhone-trained repair on Xiaomi and P40."""
from . import evaluation as F
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from IPython.display import display
FROZEN_BUNDLE = None
WEIGHT = None
DEVICE_PREFIXES = ['xiaomi', 'p40']
N_IMAGES = 20
SEED = 20261108

def run(device_prefix):
    assert torch.cuda.is_available(), '请启用GPU'
    F.FROZEN_BUNDLE = FROZEN_BUNDLE
    F.ROOT = Path('/kaggle/working/cross_device_moire')
    F.ROOT.mkdir(parents=True, exist_ok=True)
    out = F.ROOT / datetime.now().strftime(device_prefix + '_%Y%m%d_%H%M%S')
    out.mkdir()
    try:
        frozen, old, band_ck, ck = F.load_bundle()
        entries = list(HfApi().list_repo_tree(F.HF_REPO, path_in_repo='test', repo_type='dataset', revision=F.HF_REVISION))
        folders = sorted((e.path for e in entries if e.path.split('/')[-1].lower().startswith(device_prefix) and getattr(e, 'size', None) is None))
        if not folders:
            raise RuntimeError('未找到' + device_prefix + '测试目录。目录信息：' + str([e.path for e in entries]))
        F.HF_SUBDIR = folders[0]
        F.N_DCID = N_IMAGES
        F.SEED = SEED
        protocol = {'experiment': 'frozen iPhone-trained repair on ' + device_prefix + ' test', 'test_folder': F.HF_SUBDIR, 'revision': F.HF_REVISION, 'seed': SEED, 'images': N_IMAGES, 'region': 768, 'training': False, 'models': frozen, 'methods': ['original', 'original_mean_anchor', 'DCID_spatial100'], 'primary_comparison': 'DCID_spatial100 vs original_mean_anchor'}
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        records = F.download_target(out)
        H = F.I.S.C.J.H
        prior = {r[k] for r in ck['config']['signature']['samples'] for k in ['input_hash', 'gt_hash']}
        seen = set()
        for r in records:
            for key in ['input', 'target']:
                digest = H.sha_file(r[key])
                assert digest not in prior and digest not in seen, '重复图像：' + r['id']
                seen.add(digest)
                r[key + '_hash'] = digest
            with Image.open(r['input']) as x, Image.open(r['target']) as g:
                assert x.size == g.size and min(x.size) >= 768, '配对尺寸异常：' + r['id']
        pd.DataFrame(records).to_csv(out / 'manifest.csv', index=False)
        H.ROOT = F.ROOT / 'official'
        H.ROOT.mkdir(exist_ok=True)
        H.LOCAL_WEIGHT = WEIGHT
        model, meta = H.load_official_model()
        assert meta['weight_sha256'] == ck['config']['signature']['meta']['weight_sha256']
        F.I.GRID = int(ck['config']['grid'])
        F.I.MAX_RESIDUAL = float(ck['config']['max_residual'])
        net = F.I.IndependentRepair('DCID').cuda().eval()
        net.load_state_dict(ck['model'], strict=True)
        net.requires_grad_(False)
        protocol['actual_backbone'] = meta
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        results = []
        with torch.inference_mode():
            for i, r in enumerate(tqdm(records, desc=device_prefix + '跨设备测试')):
                a = np.array(Image.open(r['input']).convert('RGB'))
                g = np.array(Image.open(r['target']).convert('RGB'))
                a, g = F.I.S.C.J.crop_pair(a, g, r['id'])
                x = F.I.S.C.J.tensor(a)
                o = torch.from_numpy(H.predict(model, a, 1.0)).permute(2, 0, 1)[None].cuda().half().float()
                base = (o + (x - o).mean((-2, -1), keepdim=True)).clamp(0, 1)
                y = net(x, o, o)[0].clamp(0, 1)
                preds = {'input': x, 'original': o, 'original_mean_anchor': base, 'DCID_spatial100': y}
                folder = out / 'images' / f'{i:02d}'
                folder.mkdir(parents=True)
                panels = []
                for label, t in preds.items():
                    arr = F.I.S.C.J.arr(t)
                    results.append({'device': F.HF_SUBDIR, 'id': r['id'], 'method': label, **H.metrics(arr, g.astype(np.float32) / 255)})
                    Image.fromarray(np.round(arr * 255).astype('uint8')).save(folder / (label + '.png'))
                    panels.append((label, arr))
                Image.fromarray(g).save(folder / 'gt.png')
                H.preview(panels + [('GT', g.astype(np.float32) / 255)], folder / 'preview.jpg')
                pd.DataFrame(results).to_csv(out / 'per_image.csv', index=False)
        frame = pd.DataFrame(results)
        summary = frame.groupby('method').agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'))
        summary.to_csv(out / 'summary.csv')
        display(summary)
        pivot = frame.pivot(index='id', columns='method', values='psnr')
        differences = []
        for method, ref in [('original_mean_anchor', 'original'), ('DCID_spatial100', 'original_mean_anchor')]:
            delta = pivot[method] - pivot[ref]
            differences.append(dict(method=method, reference=ref, mean_gain=delta.mean(), median_gain=delta.median(), wins=int((delta > 0).sum()), images=len(delta)))
        pd.DataFrame(differences).to_csv(out / 'paired_comparisons.csv', index=False)
        display(pd.DataFrame(differences))
    except Exception as e:
        (out / 'error.txt').write_text(repr(e), encoding='utf-8')
        raise
    finally:
        F.I.S.pack(out)
    return out

def run_transfer():
    completed = []
    failures = []
    for device_prefix in DEVICE_PREFIXES:
        try:
            completed.append(run(device_prefix))
        except Exception as error:
            failures.append({'device': device_prefix, 'error': repr(error)})
            print(device_prefix, '运行中断，详情见该设备结果包：', repr(error))
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    combined = F.ROOT / datetime.now().strftime('multi_device_%Y%m%d_%H%M%S')
    combined.mkdir()
    import shutil
    for folder in completed:
        shutil.copytree(folder, combined / folder.name)
    if completed:
        frame = pd.concat([pd.read_csv(p / 'per_image.csv') for p in completed], ignore_index=True)
        frame.to_csv(combined / 'per_image.csv', index=False)
        summary = frame.groupby(['device', 'method']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'))
        summary.to_csv(combined / 'summary_by_device.csv')
        display(summary)
        comparisons = []
        for device, part in frame.groupby('device'):
            for metric in ['psnr', 'ssim']:
                wide = part.pivot(index='id', columns='method', values=metric)
                for method, ref in [('original_mean_anchor', 'original'), ('DCID_spatial100', 'original_mean_anchor')]:
                    delta = wide[method] - wide[ref]
                    comparisons.append(dict(device=device, metric=metric, method=method, reference=ref, mean_gain=delta.mean(), median_gain=delta.median(), wins=int((delta > 0).sum()), images=len(delta)))
        table = pd.DataFrame(comparisons)
        table.to_csv(combined / 'paired_comparisons.csv', index=False)
        display(table)
    (combined / 'run_status.json').write_text(json.dumps({'requested': DEVICE_PREFIXES, 'completed': [p.name for p in completed], 'failures': failures}, ensure_ascii=False, indent=2), encoding='utf-8')
    F.I.S.pack(combined)
    if failures:
        raise RuntimeError('部分设备未完成，已保存成功设备结果及错误记录。')
