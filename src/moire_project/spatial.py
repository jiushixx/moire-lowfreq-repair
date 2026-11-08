# Extracted from the executed Kaggle implementation; numerical logic preserved.
import types
from . import calibration as C
from pathlib import Path
from datetime import datetime
import json, random, hashlib, time, zipfile, copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import display, FileLink
CACHE_DIR = None
DCID_SOURCE = None
FUSION_ZIP = None
UHDM_ROOT = None
WEIGHT = None
STEPS = 1000
EVAL_EVERY = 250
BATCH = 4
GRID = 96
LR = 0.0003
SEED = 2026
TV_WEIGHT = 0.001
MAX_RESIDUAL = 0.35
OUT_ROOT = Path('/kaggle/working/spatial_moire')

def get_cache():
    if CACHE_DIR is not None:
        folders = [Path(CACHE_DIR)]
    else:
        roots = [Path('/kaggle/working/color_calibration/cache')]
        folders = [p.parent for root in roots if root.exists() for p in root.glob('*/signature.json')]
    if len(folders) > 1:
        raise RuntimeError(f'缓存不唯一，请填写CACHE_DIR：{folders}')
    if folders:
        folder = folders[0]
        sig = json.loads((folder / 'signature.json').read_text())
        assert sig['protocol'] == 'clipped_float16_original_and_band_v1' and sig['region'] == 768, '缓存协议不匹配'
        records = [dict(r, cache=str(folder / f'{i:03d}.npz')) for i, r in enumerate(sig['samples'])]
        if not all((Path(r['cache']).is_file() for r in records)):
            raise RuntimeError('缓存不完整；恢复缺失文件，或将该缓存目录另存后重新建立缓存')
        print('复用上一轮缓存：', folder)
        return (records, sig)
    C.DCID_SOURCE = DCID_SOURCE
    C.FUSION_ZIP = FUSION_ZIP
    C.UHDM_ROOT = UHDM_ROOT
    C.WEIGHT = WEIGHT
    items, model, band, meta = C.prepare()
    records, sig = C.cache_predictions(items, model, band, meta)
    model.cpu()
    band.cpu()
    del items, model, band
    torch.cuda.empty_cache()
    return (records, sig)

def load_memory(records):
    expected = {('DCID', 'train'): 12, ('DCID', 'val'): 8, ('UHDM', 'train'): 20, ('UHDM', 'val'): 10}
    counts = {k: sum(((r['domain'], r['split']) == k for r in records)) for k in expected}
    assert counts == expected and len(records) == 50, f'样本划分不匹配：{counts}'
    C.J.check_split([r for r in records if r['split'] == 'train'], [r for r in records if r['split'] == 'val'])
    assert len({r['id'] for r in records}) == 50, '样本ID重复'
    items = []
    for r in tqdm(records, desc='检查并载入缓存'):
        with np.load(r['cache']) as z:
            a = {k: np.array(z[k], copy=True) for k in ['x', 'gt', 'original', 'old_band']}
        for k, v in a.items():
            assert v.shape == (768, 768, 3) and np.isfinite(v).all(), f'{r['id']}/{k} 异常'
            if k in ['x', 'gt']:
                assert v.dtype == np.uint8
            else:
                assert v.min() >= 0 and v.max() <= 1
        items.append(dict(r, arrays=a))
    return items

def tensors(p, flip=0):
    output = []
    for k in ['x', 'gt', 'original', 'old_band']:
        a = p['arrays'][k]
        if flip & 1:
            a = np.flip(a, 1)
        if flip & 2:
            a = np.flip(a, 0)
        t = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().cuda()
        output.append(t / 255 if k in ['x', 'gt'] else t)
    return output

def candidates(x, o, b):

    def mean_anchor(y):
        return (y + (x - y).mean((-2, -1), keepdim=True)).clamp(0, 1)
    return torch.stack([o, b, mean_anchor(o), mean_anchor(b)], dim=1)

class LowGridNet(nn.Module):

    def __init__(self, outputs):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU())
        self.up = nn.Sequential(nn.Conv2d(72, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(24, outputs, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, f):
        e = self.enc(f)
        d = self.down(e)
        return self.final(self.up(torch.cat([e, F.interpolate(d, size=e.shape[-2:], mode='bilinear', align_corners=False)], dim=1)))

class RepairHead(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.net = LowGridNet(3 if mode == 'spatial' else 4)
        if mode == 'gate':
            with torch.no_grad():
                self.net.final.bias[1] = 2.0

    def forward(self, x, o, b):
        features = torch.cat([x, o, b, x - b], dim=1)
        f = F.interpolate(features, size=(GRID, GRID), mode='area')
        small = self.net(f)
        if self.mode == 'spatial':
            field = MAX_RESIDUAL * torch.tanh(small)
            y = b + F.interpolate(field, size=b.shape[-2:], mode='bilinear', align_corners=False)
        else:
            field = small.softmax(1)
            weights = F.interpolate(field, size=b.shape[-2:], mode='bilinear', align_corners=False)
            y = (weights.unsqueeze(2) * candidates(x, o, b)).sum(1)
        return (y, field)

def tv(field):
    return (field[:, :, 1:] - field[:, :, :-1]).abs().mean() + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()

def pack(out):
    bundle = out.parent / (out.name + '_results.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print('下载结果包：', bundle)
    display(FileLink(str(bundle)))

@torch.inference_mode()
def evaluate(items, out, method, step, net=None, ssim=True):
    if net is not None:
        net.eval()
    rows = []
    for i, p in enumerate([p for p in items if p['split'] == 'val']):
        x, g, o, b = tensors(p)
        if net is None:
            c = candidates(x, o, b)
            preds = {'input': x, 'original': c[:, 0], 'old_band': c[:, 1], 'original_mean_anchor': c[:, 2], 'old_band_mean_anchor': c[:, 3]}
        else:
            preds = {method: net(x, o, b)[0].clamp(0, 1)}
        gt = C.J.arr(g)
        for label, pred in preds.items():
            image = C.J.arr(pred)
            if ssim:
                metric = C.J.H.metrics(image, gt)
            else:
                metric = {'psnr': float(-10 * np.log10(max(np.mean((image.astype(np.float64) - gt.astype(np.float64)) ** 2), 1e-12))), 'ssim': float('nan')}
            rows.append({'domain': p['domain'], 'id': p['id'], 'method': label, 'step': step, **metric})
            if ssim:
                dest = out / 'images' / p['domain'] / label / f'step_{step}'
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.round(image * 255).astype('uint8')).save(dest / f'{i:02d}.png')
        if ssim:
            views = [('Input', C.J.arr(x)), ('Old band', C.J.arr(b))]
            views += [(k, C.J.arr(v)) for k, v in preds.items() if k not in ['input', 'old_band']]
            dest = out / 'previews'
            dest.mkdir(exist_ok=True)
            C.J.H.preview(views + [('GT', gt)], dest / f'{p['domain']}_{method}_{step}_{i:02d}.jpg')
    return rows

def persist(out, rows, history):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'per_image.csv', index=False)
    summary = frame.groupby(['domain', 'method', 'step']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'), ssim_images=('ssim', 'count'))
    summary.to_csv(out / 'summary.csv')
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    return summary

def run_spatial():
    assert torch.cuda.is_available(), '需要GPU训练小分支'
    assert BATCH > 0 and BATCH % 2 == 0 and (GRID >= 4)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / datetime.now().strftime('spatial_%Y%m%d_%H%M%S')
    out.mkdir()
    rows = []
    history = []
    try:
        records, sig = get_cache()
        items = load_memory(records)
        pd.DataFrame(records).to_csv(out / 'manifest.csv', index=False)
        config = {'seed': SEED, 'steps_per_head': STEPS, 'batch': BATCH, 'grid': GRID, 'lr': LR, 'tv_weight': TV_WEIGHT, 'max_residual': MAX_RESIDUAL, 'signature': sig, 'region': 768, 'input': 'center768 only; downsample whole center view to96; NOT original full-image guidance', 'sampling': 'equal batch counts DCID/UHDM; full center regions; synchronized horizontal/vertical flips', 'base': 'frozen old_band for spatial; gate convex weights over original,old_band,and their input-mean corrected outputs', 'loss': 'unclipped RGB MSE + TV on low-resolution residual field or gate weights; clipped output for metrics', 'selection': 'fixed last step primary; best macro development improvement saved separately', 'limits': '12/8 target and20/10 source development splits; prior inspected; no independent scene-generalization claim', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
        (out / 'protocol.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        print('阶段1：强基线（含SSIM）')
        rows = evaluate(items, out, 'baseline', 0)
        baseline = pd.DataFrame(rows).groupby(['domain', 'method']).psnr.mean()
        references = {'DCID': float(baseline['DCID', 'original_mean_anchor']), 'UHDM': float(baseline['UHDM', 'old_band'])}
        display(persist(out, rows, history))
        pack(out)
        pools = {d: [p for p in items if p['split'] == 'train' and p['domain'] == d] for d in ['DCID', 'UHDM']}
        for mode in ['gate', 'spatial']:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            net = RepairHead(mode).cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            rng = random.Random(SEED)
            parameter_count = sum((p.numel() for p in net.parameters()))
            print(mode, '参数量', parameter_count)
            x, g, o, b = tensors(pools['DCID'][0])
            initial, field = net(x, o, b)
            if mode == 'spatial':
                assert torch.equal(initial, b), '空间分支零初始化未复现基础输出'
            del x, g, o, b, initial, field
            initial_rows = evaluate(items, out, mode, 0, net=net, ssim=True)
            rows += initial_rows
            initial_score = np.mean([np.mean([r['psnr'] for r in initial_rows if r['domain'] == d]) - references[d] for d in references])
            best = float(initial_score)
            payload = lambda step: {'model': net.state_dict(), 'mode': mode, 'step': step, 'config': config, 'parameters': parameter_count}
            torch.save(payload(0), out / (mode + '_best_dev.pt'))
            persist(out, rows, history)
            start = time.perf_counter()
            recent = []
            for step in tqdm(range(1, STEPS + 1), desc=mode + '训练'):
                selected = [rng.choice(pools[d]) for d in ['DCID', 'UHDM'] for _ in range(BATCH // 2)]
                samples = [tensors(p, rng.randrange(4)) for p in selected]
                x, g, o, b = [torch.cat([s[j] for s in samples]) for j in range(4)]
                net.train()
                optimizer.zero_grad(set_to_none=True)
                pred, field = net(x, o, b)
                mse = F.mse_loss(pred, g)
                smooth = tv(field)
                loss = mse + TV_WEIGHT * smooth
                if not torch.isfinite(loss):
                    raise RuntimeError('训练loss非有限')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                recent.append(float(mse.detach()))
                del samples, pred, field, x, g, o, b, loss, mse, smooth
                if step == 20:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    print(f'{mode}前20步{elapsed:.1f}s，剩余训练约{(STEPS - step) * elapsed / 20 / 60:.1f}分钟，另有评价时间')
                if step % EVAL_EVERY == 0 or step == STEPS:
                    scored = evaluate(items, out, mode, step, net, ssim=step == STEPS)
                    rows += scored
                    means = {d: float(np.mean([r['psnr'] for r in scored if r['domain'] == d])) for d in references}
                    score = float(np.mean([means[d] - references[d] for d in references]))
                    history.append({'mode': mode, 'step': step, 'train_mse_recent': float(np.mean(recent)), 'DCID_psnr': means['DCID'], 'UHDM_psnr': means['UHDM'], 'macro_gain_vs_strong_refs': score, 'elapsed_seconds': time.perf_counter() - start})
                    recent = []
                    torch.save(payload(step), out / (mode + '_last.pt'))
                    if score > best:
                        best = score
                        torch.save(payload(step), out / (mode + '_best_dev.pt'))
                    print(mode, step, means)
                    persist(out, rows, history)
            del optimizer, net
            torch.cuda.empty_cache()
            pack(out)
        display(persist(out, rows, history))
        print('完成。固定1000步是主结果；中间步骤仅PSNR，NaN SSIM表示未计算。')
    except Exception as error:
        (out / 'error.txt').write_text(repr(error), encoding='utf-8')
        if rows:
            persist(out, rows, history)
        raise
    finally:
        pack(out)
    return out
