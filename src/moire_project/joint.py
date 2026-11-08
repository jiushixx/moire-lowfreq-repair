# Extracted from the executed Kaggle implementation; numerical logic preserved.
import types
from . import baseline as H
from pathlib import Path
import io, csv, json, random, hashlib, zipfile, time, copy
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import display, FileLink
DCID_SOURCE = None
FUSION_ZIP = None
WEIGHT = None
UHDM_ROOT = None
STEPS = 300
EVAL_EVERY = 100
DCID_TRAIN = 12
SOURCE_REPLAY = 20
REGION = 768
CROP = 256
BATCH = 2
LR = 1e-05
SEED = 2026
RUN_ROOT = Path('/kaggle/working/joint_moire_runs')

def fusion_lowpass(x):
    v = torch.arange(-4, 5, device=x.device, dtype=x.dtype)
    k = torch.exp(-v.square() / 8)
    k = k / k.sum()
    c = x.shape[1]
    a = F.conv2d(F.pad(x, (4, 4, 0, 0), mode='replicate'), k.view(1, 1, 1, 9).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(a, (0, 0, 4, 4), mode='replicate'), k.view(1, 1, 9, 1).repeat(c, 1, 1, 1), groups=c)

class ScaleFusion(nn.Module):

    def __init__(self, mode='band', preferred=0):
        super().__init__()
        assert mode in ('spatial', 'band')
        self.mode = mode
        outputs = 6 if mode == 'band' else 3
        self.net = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, outputs, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[preferred] = 6.0
            if mode == 'band':
                self.net[-1].bias[preferred + 3] = 6.0

    def forward(self, x, candidates, lows):
        b, _, _, h, w = candidates.shape
        features = torch.cat((x, candidates.reshape(b, 9, h, w)), dim=1)
        features = F.interpolate(features, size=(max(1, h // 4), max(1, w // 4)), mode='area')
        logits = F.interpolate(self.net(features), size=(h, w), mode='bilinear', align_corners=False)
        wl = logits[:, :3].softmax(dim=1).unsqueeze(2)
        wh = logits[:, 3:].softmax(dim=1).unsqueeze(2) if self.mode == 'band' else wl
        return (wl * lows + wh * (candidates - lows)).sum(dim=1)

def find_unique(pattern, explicit):
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    found = sorted((p for root in [Path('/kaggle/input'), Path('/kaggle/working')] if root.exists() for p in root.rglob(pattern)))
    if len(found) != 1:
        raise RuntimeError(f'请在配置中填写 {pattern} 的路径，候选={found}')
    return found[0]

def load_old_fusion():
    if FUSION_ZIP is not None and Path(FUSION_ZIP).is_dir():
        folder = Path(FUSION_ZIP)
    else:
        found = [Path(FUSION_ZIP)] if FUSION_ZIP is not None else [p for root in [Path('/kaggle/input'), Path('/kaggle/working')] if root.exists() for p in root.rglob('fusion_training_results.zip')]
        if len(found) > 1:
            raise RuntimeError(f'旧融合包不唯一，请填写 FUSION_ZIP：{found}')
        if found:
            with zipfile.ZipFile(found[0]) as z:
                p = z.read('protocol.json')
                ck = z.read('band/best.pt')
            return (json.loads(p), torch.load(io.BytesIO(ck), map_location='cpu', weights_only=True), hashlib.sha256(p + ck).hexdigest())
        folders = sorted({p.parent.parent for root in [Path('/kaggle/input'), Path('/kaggle/working')] if root.exists() for p in root.rglob('best.pt') if p.parent.name == 'band' and (p.parent.parent / 'protocol.json').is_file()})
        if len(folders) != 1:
            raise RuntimeError(f'请挂载旧融合结果包，FUSION_ZIP 可填写 ZIP 路径或展开后的根目录。候选={folders}')
        folder = folders[0]
    p = (folder / 'protocol.json').read_bytes()
    ck = (folder / 'band/best.pt').read_bytes()
    return (json.loads(p), torch.load(io.BytesIO(ck), map_location='cpu', weights_only=True), hashlib.sha256(p + ck).hexdigest())

def source_path(old, root):
    p = Path(old)
    if p.is_file():
        return p
    tail = old.replace('\\', '/').split('/uhdm-dataset/', 1)[-1]
    direct = root / tail
    if direct.is_file():
        return direct
    matches = [p for p in root.rglob(Path(tail).name) if p.parent.name == Path(tail).parent.name]
    if len(matches) != 1:
        raise RuntimeError(f'UHDM 图片无法唯一映射：{old}; {matches}')
    return matches[0]

def crop_pair(a, g, label):
    if a.shape != g.shape or min(a.shape[:2]) < REGION:
        raise RuntimeError(f'尺寸异常：{label}')
    h, w = a.shape[:2]
    t, l = ((h - REGION) // 2, (w - REGION) // 2)
    return (a[t:t + REGION, l:l + REGION].copy(), g[t:t + REGION, l:l + REGION].copy())

def load_source(items, root, split):
    output = []
    for row in tqdm(items, desc='UHDM ' + split):
        xp, gp = (source_path(row['x'], root), source_path(row['gt'], root))
        with Image.open(xp) as im:
            a = np.array(im.convert('RGB'))
        with Image.open(gp) as im:
            g = np.array(im.convert('RGB'))
        a, g = crop_pair(a, g, str(gp))
        output.append({'id': 'UHDM/' + gp.parent.name + '/' + gp.stem, 'x': a, 'gt': g, 'gt_hash': H.sha_file(gp), 'input_hash': H.sha_file(xp), 'path': str(gp), 'split': split, 'group': str(row['group'])})
    return output

def tensor(a):
    return torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().cuda() / 255

def anchor(y, x):
    return y + (x - y).mean(dim=(-2, -1), keepdim=True)

@torch.inference_mode()
def forward_native(model, a):
    x = tensor(a)
    h, w = x.shape[-2:]
    y = model(F.pad(x, (0, -w % 32, 0, -h % 32), mode='replicate'))[0][:, :, :h, :w]
    return (y, x)

def arr(t):
    return t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

@torch.inference_mode()
def evaluate(model, items, label, domain, out, step, anchored=False, include_baselines=False, fusion=None):
    model.eval()
    records = []
    for i, p in enumerate(items):
        raw, x = forward_native(model, p['x'])
        g = p['gt'].astype(np.float32) / 255
        predictions = {label: arr(anchor(raw, x) if anchored else raw)}
        if include_baselines:
            predictions = {'input': p['x'].astype(np.float32) / 255, 'original': arr(raw), 'mean_anchor': arr(anchor(raw.clamp(0, 1), x))}
            if fusion is not None:
                candidates = torch.stack([torch.from_numpy(H.predict(model, p['x'], s)).permute(2, 0, 1).cuda() for s in (0.5, 0.75, 1.0)])[None]
                lows = fusion_lowpass(candidates[0])[None]
                predictions['old_band'] = arr(fusion(x, candidates, lows))
        for method, pred in predictions.items():
            score = H.metrics(pred, g)
            records.append({'domain': domain, 'id': p['id'], 'method': method, 'step': step, **score})
            dest = out / 'images' / domain / method
            dest.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(pred * 255).astype('uint8')).save(dest / f'{i:02d}.png')
        if i < 3:
            dest = out / 'previews'
            dest.mkdir(exist_ok=True)
            H.preview([('Input', p['x'] / 255)] + list(predictions.items()) + [('GT', g)], dest / f'{domain}_{label}_{step}_{i}.jpg')
    return records

def check_split(training, validation):
    for field in ['gt_hash', 'input_hash']:
        overlap = {p[field] for p in training} & {p[field] for p in validation}
        if overlap:
            raise RuntimeError(f'训练/验证有精确重复 {field}；请修复划分，尚未启动训练')

def draw_batch(items, rng):
    xs = []
    gs = []
    for _ in range(BATCH):
        p = rng.choice(items)
        t = rng.randrange(REGION - CROP + 1)
        l = rng.randrange(REGION - CROP + 1)
        a = p['x'][t:t + CROP, l:l + CROP]
        g = p['gt'][t:t + CROP, l:l + CROP]
        if rng.random() < 0.5:
            a, g = (np.flip(a, 1), np.flip(g, 1))
        xs.append(tensor(a))
        gs.append(tensor(g))
    return (torch.cat(xs), torch.cat(gs))

def save_bundle(out):
    bundle = out.parent / (out.name + '_results.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print('结果包：', bundle)
    display(FileLink(str(bundle)))

def run_joint():
    assert torch.cuda.is_available(), '需要 GPU；CPU 会话先保存数据即可'
    assert REGION % 32 == 0 and CROP % 32 == 0 and (CROP <= REGION)
    protocol, checkpoint, old_fusion_hash = load_old_fusion()
    if UHDM_ROOT is None:
        roots = sorted((p for p in Path('/kaggle/input').rglob('uhdm-dataset') if p.is_dir()))
        if len(roots) != 1:
            raise RuntimeError('请 Add Input 挂载 soumikrakshit/uhdm-dataset，或填写 UHDM_ROOT')
        source_root = roots[0]
    else:
        source_root = Path(UHDM_ROOT)
    H.DATA_SOURCE = DCID_SOURCE
    source = H.PairSource(H.locate_source())
    rows = sorted(source.rows, key=lambda r: r['sample_id'])
    random.Random(SEED).shuffle(rows)
    assert len(rows) == 20, '首轮固定使用已检查的 20 对，不自动加入其他图片'
    target = []
    for i, row in enumerate(tqdm(rows, desc='DCID 配对')):
        a, g = crop_pair(source.read(row, 'input'), source.read(row, 'target'), row['sample_id'])
        target.append({'id': 'DCID/' + row['sample_id'], 'x': a, 'gt': g, 'gt_hash': hashlib.sha256(source.read_bytes(row, 'target')).hexdigest(), 'input_hash': hashlib.sha256(source.read_bytes(row, 'input')).hexdigest(), 'split': 'train' if i < DCID_TRAIN else 'development_val', 'path': row['target']})
    target_train, target_val = (target[:DCID_TRAIN], target[DCID_TRAIN:])
    source_train = load_source(protocol['manifest']['train'][:SOURCE_REPLAY], source_root, 'replay_train')
    source_val = load_source(protocol['manifest']['val'], source_root, 'old_development_val')
    check_split(source_train + target_train, source_val + target_val)
    if {p['group'] for p in source_train} & {p['group'] for p in source_val}:
        raise RuntimeError('UHDM 旧组划分有交集')
    out = RUN_ROOT / datetime.now().strftime('joint_%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    H.ROOT = RUN_ROOT / 'official'
    H.ROOT.mkdir(exist_ok=True)
    H.LOCAL_WEIGHT = WEIGHT
    model, metadata = H.load_official_model()
    digest = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        digest.update(k.encode())
        digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != protocol['weight_sha256']:
        raise RuntimeError('原始主干与旧融合实验权重不一致，停止，不能混接')
    band = ScaleFusion('band', int(checkpoint['settings']['preferred'])).cuda().eval()
    band.load_state_dict(checkpoint['model'], strict=True)
    for p in band.parameters():
        p.requires_grad_(False)
    manifest = [{k: v for k, v in p.items() if k not in ['x', 'gt']} for p in target + source_train + source_val]
    pd.DataFrame(manifest).to_csv(out / 'manifest.csv', index=False)
    settings = {**metadata, 'seed': SEED, 'region': REGION, 'crop': CROP, 'batch': BATCH, 'steps_per_branch': STEPS, 'lr': LR, 'source_replay_images': len(source_train), 'source_val_images': len(source_val), 'target_train_images': len(target_train), 'target_val_images': len(target_val), 'replay_schedule': 'three DCID batches then one UHDM batch; total optimizer steps equal', 'trainable': 'decoder only; encoder frozen eval; MSE on unclipped input-mean-anchored prediction; adaptation eval anchors raw output then clips; fixed mean_anchor baseline anchors clipped original prediction', 'old_fusion_protocol_and_band_sha256': old_fusion_hash, 'old_protocol': protocol, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'limitations': 'DCID image split, scene grouping unknown, prior inspected data; UHDM old development val; no independent test claim. Fixed last step is primary; best target checkpoint is exploratory. Same total steps means different target update counts.'}
    (out / 'protocol.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    records = []
    history = []
    try:
        print('阶段 1：两数据集固定基线与旧融合，先得到可下载结果')
        for domain, items in [('DCID', target_val), ('UHDM', source_val)]:
            records += evaluate(model, items, 'baseline', domain, out, 0, include_baselines=True, fusion=band)
        del band
        torch.cuda.empty_cache()

        def persist():
            frame = pd.DataFrame(records)
            frame.to_csv(out / 'per_image.csv', index=False)
            table = frame.groupby(['domain', 'method', 'step']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('id', 'nunique'))
            table.to_csv(out / 'summary.csv')
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            return table
        display(persist())
        save_bundle(out)
        for branch in ['target_only', 'target_with_replay']:
            model.load_state_dict(initial, strict=True)
            model.requires_grad_(False)
            model.decoder.requires_grad_(True)
            opt = torch.optim.Adam(model.decoder.parameters(), lr=LR)
            rng = random.Random(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            best = -float('inf')
            begin = time.perf_counter()
            for step in tqdm(range(1, STEPS + 1), desc=branch):
                model.train()
                model.encoder.eval()
                pool = source_train if branch == 'target_with_replay' and step % 4 == 0 else target_train
                x, g = draw_batch(pool, rng)
                opt.zero_grad(set_to_none=True)
                y = model(x)[0]
                loss = F.mse_loss(anchor(y, x), g)
                if not torch.isfinite(loss):
                    raise RuntimeError('loss 非有限值，停止')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                opt.step()
                if step == 20:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    print(f'{branch} 前20步 {elapsed:.1f}s；仅训练部分预计剩余 {(STEPS - step) * elapsed / 20 / 60:.1f} 分钟，另有评价开销')
                if step % EVAL_EVERY == 0 or step == STEPS:
                    metrics = []
                    for domain, items in [('DCID', target_val), ('UHDM', source_val)]:
                        metrics += evaluate(model, items, branch, domain, out, step, anchored=True)
                    records += metrics
                    score = float(np.mean([r['psnr'] for r in metrics if r['domain'] == 'DCID']))
                    history.append({'branch': branch, 'step': step, 'train_loss': float(loss.detach()), 'target_val_psnr': score, 'elapsed_seconds': time.perf_counter() - begin})
                    payload = {'decoder': model.decoder.state_dict(), 'step': step, 'branch': branch, 'settings': {k: v for k, v in settings.items() if k != 'old_protocol'}}
                    torch.save(payload, out / (branch + '_last.pt'))
                    if score > best:
                        best = score
                        torch.save(payload, out / (branch + '_best_target_dev.pt'))
                    display(persist())
            del opt
            save_bundle(out)
        print('完成：优先比较两个分支固定第300步与 mean_anchor；同时查看 UHDM 是否退化。')
        return out
    except Exception as error:
        (out / 'error.txt').write_text(repr(error), encoding='utf-8')
        if records:
            pd.DataFrame(records).to_csv(out / 'partial_per_image.csv', index=False)
        save_bundle(out)
        raise
