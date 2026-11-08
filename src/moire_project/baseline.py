# Extracted from the executed Kaggle implementation; numerical logic preserved.
"""复制整个文件到 Kaggle 新单元格执行。先安装 gdown pyyaml scikit-image。
独立入口：不依赖原 UHDM 数据集、不加载融合模型、不训练。
"""
from pathlib import Path
import csv, io, json, hashlib, zipfile, sys, subprocess, importlib.util, time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm
from IPython.display import display, FileLink
DATA_SOURCE = None
LOCAL_WEIGHT = None
REGION = 768
SCALES = (1.0, 0.5)
MAX_PAIRS = 20
ROOT = Path('/kaggle/working/dcid_baseline')

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class PairSource:

    def __init__(self, source):
        self.path = Path(source)
        self.archive = zipfile.ZipFile(self.path) if self.path.suffix.lower() == '.zip' else None
        if self.archive:
            members = [n for n in self.archive.namelist() if n.endswith('_manifest_draft.csv')]
            if len(members) != 1:
                raise RuntimeError(f'ZIP 内应有唯一原始清单，实际为 {members}')
            self.manifest_bytes = self.archive.read(members[0])
            self.prefix = members[0].rsplit('/', 1)[0] + '/' if '/' in members[0] else ''
        else:
            self.manifest_bytes = self.path.read_bytes()
        self.rows = list(csv.DictReader(io.StringIO(self.manifest_bytes.decode('utf-8-sig'), newline='')))
        if not self.rows:
            raise RuntimeError('清单为空')
        if not {'sample_id', 'input', 'target'} <= set(self.rows[0]):
            raise RuntimeError('清单缺少必要字段')
        ids = [r['sample_id'] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError('sample_id 重复，请检查清单')

    def read_bytes(self, row, key):
        name = row[key].replace('\\', '/')
        relative = 'images/' + name.split('/images/', 1)[1] if '/images/' in name else name
        if self.archive:
            return self.archive.read(self.prefix + relative)
        p = Path(name)
        if not p.is_file():
            p = self.path.parent / relative
        if not p.is_file():
            raise FileNotFoundError(f'{key}: {name}，也未找到 {p}')
        return p.read_bytes()

    def read(self, row, key):
        with Image.open(io.BytesIO(self.read_bytes(row, key))) as image:
            return np.array(image.convert('RGB'), copy=True)

def locate_source():
    if DATA_SOURCE is not None:
        return Path(DATA_SOURCE)
    preferred = Path('/kaggle/working/dcid_hf_subset/train_iphone_1_part1_20_2026_manifest_draft.csv')
    if preferred.is_file():
        return preferred
    for pattern in ['*_manifest_draft.csv', '*_with_images.zip']:
        found = sorted((p for base in [Path('/kaggle/working'), Path('/kaggle/input')] if base.exists() for p in base.rglob(pattern)))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(f'候选不唯一，请设置 DATA_SOURCE：{found}')
    raise FileNotFoundError('没有找到 DCID 清单或 ZIP。请恢复下载结果，或将原图子集挂载为 Kaggle Input。')

def load_official_model():
    repo = Path('/kaggle/working/moire_project/UHDM')
    if not (repo / 'model/nets.py').is_file():
        repo = ROOT / 'UHDM'
        if not repo.exists():
            subprocess.check_call(['git', 'clone', '--depth', '1', 'https://github.com/CVMI-Lab/UHDM.git', str(repo)])
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if LOCAL_WEIGHT is not None:
        weight = Path(LOCAL_WEIGHT)
        if not weight.is_file():
            raise FileNotFoundError(weight)
    else:
        known = [Path('/kaggle/working/moire_project/uhdm_checkpoint.pth'), ROOT / 'uhdm_checkpoint.pth']
        weight = next((p for p in known if p.is_file()), None)
        if weight is None and Path('/kaggle/input').exists():
            found = list(Path('/kaggle/input').rglob('uhdm_checkpoint.pth'))
            if len(found) > 1:
                raise RuntimeError(f'找到多个权重，请指定 LOCAL_WEIGHT：{found}')
            if found:
                weight = found[0]
        if weight is None:
            import gdown
            weight = ROOT / 'uhdm_checkpoint.pth'
            temporary = weight.with_suffix('.download')
            gdown.download(id='1HT_ubcAYRqzFIJ46XuPhrulJk2YFBIEl', output=str(temporary), quiet=False)
            if not temporary.is_file():
                raise RuntimeError('官方权重下载失败，请挂载旧权重并填写 LOCAL_WEIGHT')
            temporary.replace(weight)
    spec = importlib.util.spec_from_file_location('dcid_esdnet', repo / 'model/nets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = yaml.safe_load((repo / 'config/uhdm_config.yaml').read_text())['MODEL']
    model = module.my_model(en_feature_num=cfg['EN_FEATURE_NUM'], en_inter_num=cfg['EN_INTER_NUM'], de_feature_num=cfg['DE_FEATURE_NUM'], de_inter_num=cfg['DE_INTER_NUM'], sam_number=cfg['SAM_NUMBER'])
    state = torch.load(weight, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    return (model.cuda().eval().requires_grad_(False), {'repo_commit': commit, 'weight_sha256': sha_file(weight), 'weight_path': str(weight)})

@torch.inference_mode()
def predict(model, array, scale):
    h, w = array.shape[:2]
    x = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float().cuda() / 255
    sh, sw = (round(h * scale), round(w * scale))
    if (sh, sw) != (h, w):
        x = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=False, antialias=True)
    x = F.pad(x, (0, -sw % 32, 0, -sh % 32), mode='replicate')
    y = model(x)
    y = y[0] if isinstance(y, (tuple, list)) else y
    y = y[:, :, :sh, :sw]
    if (sh, sw) != (h, w):
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
    return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

def metrics(pred, gt):
    pred = np.clip(pred, 0, 1).astype(np.float32)
    gt = gt.astype(np.float32)
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    return {'psnr': float(-10 * np.log10(max(mse, 1e-12))), 'ssim': float(structural_similarity(gt, pred, data_range=1.0, channel_axis=-1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))}

def preview(images, path):
    sheet = Image.new('RGB', (320 * len(images), 350), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, array) in enumerate(images):
        im = Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype('uint8'))
        sheet.paste(ImageOps.contain(im, (315, 315)), (i * 320, 30))
        draw.text((i * 320 + 5, 5), label, fill='black')
    sheet.save(path, quality=95)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('当前没有 GPU。先保存下载结果；GPU 空闲后，在 Notebook 设置中启用 GPU 再运行。')
    ROOT.mkdir(parents=True, exist_ok=True)
    source = PairSource(locate_source())
    rows = source.rows[:MAX_PAIRS]
    fingerprints = []
    for row in tqdm(rows, desc='检查配对'):
        a, g = (source.read(row, 'input'), source.read(row, 'target'))
        if a.shape != g.shape or min(a.shape[:2]) < REGION:
            raise RuntimeError(f'配对尺寸不适用：{row['sample_id']} {a.shape} {g.shape}')
        fingerprints.append({'sample_id': row['sample_id'], **{k + '_sha256': hashlib.sha256(source.read_bytes(row, k)).hexdigest() for k in ['input', 'target']}})
    out = ROOT / datetime.now().strftime('center768_%Y%m%d_%H%M%S_%f')
    out.mkdir()
    model, metadata = load_official_model()
    protocol = {**metadata, 'data_source': str(source.path), 'manifest_sha256': hashlib.sha256(source.manifest_bytes).hexdigest(), 'fingerprints': fingerprints, 'region': REGION, 'crop': 'native-resolution center; identical for all methods', 'scales': SCALES, 'pairs': len(rows), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'training': False, 'status': 'exploratory target training-pool subset; not official test or full-image score', 'ssim': 'RGB, gaussian sigma1.5, population covariance, data_range1', 'timing': 'synchronized predict call incl CPU/GPU transfer; no speedup claim; first image includes warmup overhead'}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out / 'evaluated_manifest.csv', index=False)
    results = []
    for index, row in enumerate(tqdm(rows, desc='DCID 基线推理')):
        a, g = (source.read(row, 'input'), source.read(row, 'target'))
        h, w = a.shape[:2]
        top, left = ((h - REGION) // 2, (w - REGION) // 2)
        a = a[top:top + REGION, left:left + REGION].copy()
        gt = g[top:top + REGION, left:left + REGION].astype(np.float32) / 255
        x = a.astype(np.float32) / 255
        images = [('Input', x)]
        folder = out / f'{index:02d}'
        folder.mkdir()
        Image.fromarray(a).save(folder / 'input.png')
        Image.fromarray(g[top:top + REGION, left:left + REGION]).save(folder / 'gt.png')
        base = {'sample_id': row['sample_id'], 'index': index, 'top': top, 'left': left, 'region': REGION}
        results.append({**base, 'method': 'input', **metrics(x, gt), 'seconds': 0.0})
        for scale in SCALES:
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                pred = predict(model, a, scale)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError('显存不足：当前实验停止，不混合不同区域的成绩。将 REGION 改成 512 后整轮重跑；结果目录会独立保存。') from None
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            method = f'esd_scale_{scale:g}'
            results.append({**base, 'method': method, **metrics(pred, gt), 'seconds': seconds})
            Image.fromarray(np.round(pred * 255).astype('uint8')).save(folder / f'{method}.png')
            images.append((method, pred))
        preview(images + [('GT', gt)], folder / 'comparison.jpg')
        pd.DataFrame(results).to_csv(out / 'per_image.csv', index=False)
    frame = pd.DataFrame(results)
    summary = frame.groupby('method').agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'), images=('index', 'nunique'))
    summary.to_csv(out / 'summary.csv')
    pivot = frame.pivot(index='sample_id', columns='method', values='psnr')
    delta = pivot.sub(pivot['input'], axis=0).drop(columns='input')
    delta.to_csv(out / 'gain_vs_input.csv')
    display(summary)
    print('相对输入的逐图 PSNR 增益统计：')
    display(delta.agg(['mean', 'median', 'min', 'max']))
    print('正增益图片数：')
    print((delta > 0).sum())
    with Image.open(out / '00/comparison.jpg') as im:
        display(im.copy())
    bundle = out.parent / (out.name + '_results.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print('完成。当前仅为中心区域开发诊断；不训练、不使用正式测试集、不宣称实际加速。')
    print('下载并上传此 ZIP：', bundle)
    display(FileLink(str(bundle)))
    return out
