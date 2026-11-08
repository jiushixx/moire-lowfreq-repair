# Extracted from the executed Kaggle implementation; numerical logic preserved.
import types
from . import spatial as S
from pathlib import Path
from datetime import datetime
import json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from IPython.display import display
CACHE_DIR = None
DCID_SOURCE = None
FUSION_ZIP = None
UHDM_ROOT = None
WEIGHT = None
STEPS = 500
EVAL_EVERY = 100
BATCH = 4
GRID = 96
LR = 0.0003
TV_WEIGHT = 0.001
MAX_RESIDUAL = 0.35
SEED = 2026
ROOT = Path('/kaggle/working/independent_repair')

def base_for_domain(domain, x, o, b):
    if domain == 'DCID':
        return (o + (x - o).mean((-2, -1), keepdim=True)).clamp(0, 1)
    if domain == 'UHDM':
        return b
    raise ValueError(domain)

class IndependentRepair(nn.Module):

    def __init__(self, domain):
        super().__init__()
        self.domain = domain
        self.net = S.LowGridNet(3)

    def forward(self, x, o, b):
        base = base_for_domain(self.domain, x, o, b)
        f = F.interpolate(torch.cat([x, o, base, x - base], 1), size=(GRID, GRID), mode='area')
        field = MAX_RESIDUAL * torch.tanh(self.net(f))
        return (base + F.interpolate(field, size=base.shape[-2:], mode='bilinear', align_corners=False), field)

def persist(out, rows, history):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'per_image.csv', index=False)
    table = frame.groupby(['domain', 'split', 'method', 'step']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'), ssim_images=('ssim', 'count'), raw_mse=('raw_mse', 'mean'), objective=('objective', 'mean'), field_tv=('field_tv', 'mean'))
    table.to_csv(out / 'summary.csv')
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    return table

@torch.inference_mode()
def evaluate(items, domain, out, step, net=None):
    if net is not None:
        net.eval()
    rows = []
    for index, p in enumerate([p for p in items if p['domain'] == domain]):
        x, g, o, b = S.tensors(p)
        base = base_for_domain(domain, x, o, b)
        if net is None:
            raw, field = (base, None)
        else:
            raw, field = net(x, o, b)
        image = S.C.J.arr(raw)
        gt = S.C.J.arr(g)
        do_ssim = p['split'] == 'val' and step in (0, STEPS)
        if do_ssim:
            metric = S.C.J.H.metrics(image, gt)
        else:
            metric = {'psnr': float(-10 * np.log10(max(np.mean((image.astype(np.float64) - gt.astype(np.float64)) ** 2), 1e-12))), 'ssim': float('nan')}
        raw_mse = float(F.mse_loss(raw, g))
        smooth = 0.0 if field is None else float(S.tv(field))
        method = 'strong_base' if net is None else 'independent_spatial'
        rows.append({'domain': domain, 'split': p['split'], 'id': p['id'], 'method': method, 'step': step, **metric, 'raw_mse': raw_mse, 'field_tv': smooth, 'objective': raw_mse + TV_WEIGHT * smooth})
        if do_ssim:
            dest = out / 'images' / domain / method / f'step_{step}'
            dest.mkdir(parents=True, exist_ok=True)
            for name, a in [('input', S.C.J.arr(x)), ('gt', gt), ('base', S.C.J.arr(base)), ('output', image)]:
                Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype('uint8')).save(dest / f'{index:02d}_{name}.png')
            folder = out / 'previews'
            folder.mkdir(exist_ok=True)
            S.C.J.H.preview([('Input', S.C.J.arr(x)), ('Strong base', S.C.J.arr(base)), ('Repair', image), ('GT', gt)], folder / f'{domain}_{method}_{step}_{index:02d}.jpg')
    return rows

def run_independent():
    assert torch.cuda.is_available(), '需要GPU训练；无需重训主干'
    assert BATCH > 0 and STEPS > 0 and (GRID >= 4)
    for name in ['CACHE_DIR', 'DCID_SOURCE', 'FUSION_ZIP', 'UHDM_ROOT', 'WEIGHT']:
        setattr(S, name, globals()[name])
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / datetime.now().strftime('independent_%Y%m%d_%H%M%S')
    out.mkdir()
    rows = []
    history = []
    try:
        records, signature = S.get_cache()
        items = S.load_memory(records)
        pd.DataFrame(records).to_csv(out / 'manifest.csv', index=False)
        config = {'steps_per_domain': STEPS, 'eval_every': EVAL_EVERY, 'batch': BATCH, 'grid': GRID, 'lr': LR, 'tv_weight': TV_WEIGHT, 'max_residual': MAX_RESIDUAL, 'seed': SEED, 'signature': signature, 'base': {'DCID': 'original_mean_anchor', 'UHDM': 'old_band'}, 'train': 'independent head per known source; full center768, flips; frozen cached backbones', 'evaluation': 'all fixed training images and all fixed validation images, without augmentation; raw MSE/objective and clipped PSNR; val SSIM at0/500', 'selection': 'fixed500 primary; validation-best including zero checkpoint saved as development selection only', 'limits': 'known-source adaptation, NOT blind domain identification; 12/8 DCID and20/10 UHDM prior-inspected image splits, unknown scene overlap, no independent test', 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
        (out / 'protocol.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        for domain in ['DCID', 'UHDM']:
            print('开始独立适配：', domain)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            net = IndependentRepair(domain).cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            pool = [p for p in items if p['domain'] == domain and p['split'] == 'train']
            x, g, o, b = S.tensors(pool[0])
            initial, _ = net(x, o, b)
            assert torch.equal(initial, base_for_domain(domain, x, o, b)), '零初始化没有复现强基线'
            del x, g, o, b, initial
            base_rows = evaluate(items, domain, out, 0)
            zero_rows = [dict(r, method='independent_spatial') for r in base_rows]
            rows += base_rows + zero_rows
            val_base = float(np.mean([r['psnr'] for r in base_rows if r['split'] == 'val']))
            best = val_base
            parameter_count = sum((p.numel() for p in net.parameters()))
            payload = lambda step: {'model': net.state_dict(), 'domain': domain, 'step': step, 'config': config, 'parameters': parameter_count}
            torch.save(payload(0), out / (domain + '_best_dev.pt'))
            print('参数量', parameter_count, '零初始化验证通过，验证基线', val_base)
            display(persist(out, rows, history))
            S.pack(out)
            rng = random.Random(SEED)
            start = time.perf_counter()
            recent = []
            for step in tqdm(range(1, STEPS + 1), desc=domain + '独立训练'):
                samples = [S.tensors(rng.choice(pool), rng.randrange(4)) for _ in range(BATCH)]
                x, g, o, b = [torch.cat([s[j] for s in samples]) for j in range(4)]
                net.train()
                optimizer.zero_grad(set_to_none=True)
                pred, field = net(x, o, b)
                mse = F.mse_loss(pred, g)
                loss = mse + TV_WEIGHT * S.tv(field)
                if not torch.isfinite(loss):
                    raise RuntimeError('loss非有限值')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                recent.append(float(mse.detach()))
                del samples, x, g, o, b, pred, field, mse, loss
                if step == 20:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    print(f'前20步{elapsed:.1f}s，余下训练约{(STEPS - step) * elapsed / 20 / 60:.1f}分钟，另有评价时间')
                if step % EVAL_EVERY == 0 or step == STEPS:
                    scored = evaluate(items, domain, out, step, net)
                    rows += scored
                    fixed = {s: {k: float(np.mean([r[k] for r in scored if r['split'] == s])) for k in ['psnr', 'raw_mse', 'objective']} for s in ['train', 'val']}
                    val_score = fixed['val']['psnr']
                    history.append({'domain': domain, 'step': step, 'batch_mse_recent': float(np.mean(recent)), 'fixed_train_psnr': fixed['train']['psnr'], 'fixed_val_psnr': val_score, 'fixed_train_raw_mse': fixed['train']['raw_mse'], 'fixed_val_raw_mse': fixed['val']['raw_mse'], 'fixed_train_objective': fixed['train']['objective'], 'fixed_val_objective': fixed['val']['objective'], 'val_gain_vs_strong_base': val_score - val_base, 'elapsed_seconds': time.perf_counter() - start})
                    recent = []
                    torch.save(payload(step), out / (domain + '_last.pt'))
                    if val_score > best:
                        best = val_score
                        torch.save(payload(step), out / (domain + '_best_dev.pt'))
                    print(domain, step, '固定训练/验证PSNR', fixed['train']['psnr'], val_score, '相对强基线', val_score - val_base)
                    persist(out, rows, history)
            del optimizer, net
            torch.cuda.empty_cache()
            S.pack(out)
        display(persist(out, rows, history))
        print('完成。主结果固定500步；同时检查固定训练集与验证集的差异。最佳开发检查点可能仍为step0，代码不会强行选择退化模型。')
    except Exception as error:
        (out / 'error.txt').write_text(repr(error), encoding='utf-8')
        if rows:
            persist(out, rows, history)
        raise
    finally:
        S.pack(out)
    return out
