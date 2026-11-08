# 在原 Kaggle Notebook 运行第 1～4 节后，把本文件全部复制到新单元格。
# 不运行旧训练、不改旧验证划分。本脚本不读取 Test。
from pathlib import Path
import json, hashlib, random, math, time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity
from IPython.display import display, FileLink

# ---------- 首轮训练设置 ----------
F_TRAIN_IMAGES=100
F_REGION=768                 # 中心原分辨率区域，三种尺度共用同一视野
F_CROP=256
F_SCALES=(0.5,0.75,1.0)
F_STEPS=2000                 # 每个可训练模式各 2000 步
F_BATCH=4
F_MODES=('spatial','band')   # 先普通融合，再颜色/细节分开融合
F_VALIDATE_EVERY=200
F_LR=0.001
F_SEED=2026
F_DEVICE='cuda'
F_DO_TRAIN=True
# ---------------------------------

def fusion_lowpass(x):
    # 固定 9×9 高斯滤波（可分离），sigma=2；分解不是完美的颜色/纹理语义分离。
    v=torch.arange(-4,5,device=x.device,dtype=x.dtype)
    k=torch.exp(-v.square()/8); k=k/k.sum()
    c=x.shape[1]
    a=F.conv2d(F.pad(x,(4,4,0,0),mode='replicate'),k.view(1,1,1,9).repeat(c,1,1,1),groups=c)
    return F.conv2d(F.pad(a,(0,0,4,4),mode='replicate'),k.view(1,1,9,1).repeat(c,1,1,1),groups=c)

class ScaleFusion(nn.Module):
    def __init__(self,mode='band',preferred=0):
        super().__init__()
        assert mode in ('spatial','band')
        self.mode=mode
        outputs=6 if mode=='band' else 3
        self.net=nn.Sequential(nn.Conv2d(12,24,3,padding=1),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,outputs,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        # 从接近验证集选出的最佳固定尺度开始，避免初始就退化为任意平均。
        with torch.no_grad():
            self.net[-1].bias[preferred]=6.
            if mode=='band': self.net[-1].bias[preferred+3]=6.
    def forward(self,x,candidates,lows):
        # candidates / lows: B×3尺度×3颜色×H×W
        b,_,_,h,w=candidates.shape
        features=torch.cat((x,candidates.reshape(b,9,h,w)),dim=1)
        features=F.interpolate(features,size=(max(1,h//4),max(1,w//4)),mode='area')
        logits=F.interpolate(self.net(features),size=(h,w),mode='bilinear',align_corners=False)
        wl=logits[:,:3].softmax(dim=1).unsqueeze(2)
        wh=logits[:,3:].softmax(dim=1).unsqueeze(2) if self.mode=='band' else wl
        # 不额外生成纹理，融合现有候选；高频残差可正可负，训练中不 clamp。
        return (wl*lows+wh*(candidates-lows)).sum(dim=1)

@torch.inference_mode()
def predict_fusion_candidate(model,x,scale,device):
    x=x.to(device); h,w=x.shape[-2:]
    sh,sw=round(h*scale),round(w*scale)
    if (sh,sw)!=(h,w): x=F.interpolate(x,size=(sh,sw),mode='bilinear',align_corners=False,antialias=True)
    x=F.pad(x,(0,(-sw)%32,0,(-sh)%32),mode='replicate')
    y=model(x); y=y[0] if isinstance(y,(tuple,list)) else y
    y=y[:,:,:sh,:sw]
    if (sh,sw)!=(h,w): y=F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
    return y.clamp(0,1).cpu()

def fusion_manifest(all_items,validation,ntrain):
    assert validation
    val_groups={str(p['group']) for p in validation}
    available={}
    for p in all_items:
        g=str(p['group'])
        if g not in val_groups: available.setdefault(g,p)
    keys=sorted(available); random.Random(F_SEED).shuffle(keys)
    assert len(keys)>=ntrain, f'排除验证组后只有 {len(keys)} 组，请降低 F_TRAIN_IMAGES'
    training=[available[g] for g in keys[:ntrain]]
    assert not ({str(p['group']) for p in training}&val_groups)
    # 同一 GT 若有多个不同编号，也不能跨训练/验证；其他同场景关系仍需人工/元数据分组。
    seen={hashlib.sha256(Path(p['gt']).read_bytes()).hexdigest() for p in validation}
    for p in training:
        assert hashlib.sha256(Path(p['gt']).read_bytes()).hexdigest() not in seen, '训练/验证有重复 GT，请先修复分组'
    return {'train':training,'val':validation}

def atomic_npy(path,array):
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f: np.save(f,array)
    tmp.replace(path)

def create_fusion_cache(model,items,split,folder):
    paths=[]
    for i,p in enumerate(tqdm(items,desc=f'三尺度缓存 {split}')):
        dest=folder/'cache'/split/f'{i:04d}'; dest.mkdir(parents=True,exist_ok=True)
        paths.append(dest)
        if (dest/'ready.json').exists(): continue
        a=np.asarray(Image.open(p['x']).convert('RGB')).copy()
        g=np.asarray(Image.open(p['gt']).convert('RGB')).copy()
        assert a.shape==g.shape and min(a.shape[:2])>=F_REGION, f'尺寸不适合：{p}'
        h,w=a.shape[:2]; t,l=(h-F_REGION)//2,(w-F_REGION)//2
        a=a[t:t+F_REGION,l:l+F_REGION]; g=g[t:t+F_REGION,l:l+F_REGION]
        x=torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255
        try:
            candidates=torch.stack([predict_fusion_candidate(model,x,s,F_DEVICE)[0] for s in F_SCALES])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError('整视野缓存显存不足。可把 F_REGION 改成 512 后重跑本单元格，将自动使用新实验目录；不要临时换成分块。') from None
        # 在完整区域上滤波后缓存，训练随机裁剪不会改变频段分解。
        lows=fusion_lowpass(candidates).numpy()
        atomic_npy(dest/'x.npy',a.transpose(2,0,1))
        atomic_npy(dest/'gt.npy',g.transpose(2,0,1))
        atomic_npy(dest/'candidates.npy',candidates.numpy().astype(np.float16))
        atomic_npy(dest/'lows.npy',lows.astype(np.float16))
        (dest/'ready.json').write_text(json.dumps(p,indent=2))
    return paths

class FusionData(Dataset):
    def __init__(self,paths,crop=None): self.paths=paths; self.crop=crop
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        p=self.paths[i]
        arrays=[np.load(p/name,mmap_mode='r') for name in ['x.npy','candidates.npy','lows.npy','gt.npy']]
        h,w=arrays[0].shape[-2:]
        if self.crop:
            t,l=random.randrange(h-self.crop+1),random.randrange(w-self.crop+1)
            arrays=[a[...,t:t+self.crop,l:l+self.crop] for a in arrays]
            if random.random()<.5: arrays=[np.flip(a,-1) for a in arrays]
            if random.random()<.5: arrays=[np.flip(a,-2) for a in arrays]
        tensors=[torch.from_numpy(np.array(a,dtype=np.float32,copy=True)) for a in arrays]
        tensors[0]/=255; tensors[3]/=255
        return tuple(tensors)

def image_metrics(pred,gt):
    pred=pred.clamp(0,1)
    error=float((pred-gt).square().mean())
    return -10*math.log10(max(error,1e-12))

@torch.inference_mode()
def evaluate_fusion(net,dataset,device,method):
    if net is not None: net.eval()
    records=[]
    for i in range(len(dataset)):
        x,c,lo,g=[v[None].to(device) for v in dataset[i]]
        if method=='mean': pred=c.mean(1)
        elif method.startswith('scale_'): pred=c[:,int(method.split('_')[1])]
        else: pred=net(x,c,lo)
        records.append({'image':i,'method':method,'psnr':image_metrics(pred,g)})
    return records

def save_fusion_preview(net,dataset,preferred,out,device):
    net.eval(); rows=[]
    with torch.inference_mode():
        for i in range(min(3,len(dataset))):
            x,c,lo,g=[v[None].to(device) for v in dataset[i]]
            y=net(x,c,lo).clamp(0,1)
            fig,axes=plt.subplots(1,5,figsize=(20,4))
            for ax,img,title in zip(axes,[x,c[:,preferred],c.mean(1),y,g],['Input','Best fixed scale','Mean','Learned','GT']):
                ax.imshow(img[0].clamp(0,1).permute(1,2,0).cpu()); ax.set_title(title); ax.axis('off')
            plt.tight_layout(); plt.savefig(out/f'preview_{i}.png',dpi=150); plt.show(); plt.close(fig)
            for label,pred in [('fixed',c[:,preferred]),('mean',c.mean(1)),('learned',y)]:
                a=pred[0].clamp(0,1).permute(1,2,0).cpu().numpy()
                b=g[0].permute(1,2,0).cpu().numpy()
                rows.append({'image':i,'method':label,'ssim':structural_similarity(a,b,channel_axis=2,data_range=1.)})
    pd.DataFrame(rows).to_csv(out/'preview_ssim_first3_only.csv',index=False)

def train_fusion(mode,trainset,valset,preferred,out):
    out.mkdir(parents=True,exist_ok=True)
    random.seed(F_SEED); np.random.seed(F_SEED); torch.manual_seed(F_SEED)
    if F_DEVICE=='cuda': torch.cuda.manual_seed_all(F_SEED)
    net=ScaleFusion(mode,preferred).to(F_DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=F_LR)
    state_path=out/'last.pt'; start=0; history=[]; best=-float('inf')
    settings={'mode':mode,'crop':F_CROP,'batch':F_BATCH,'seed':F_SEED,'lr':F_LR,'preferred':preferred,'loss':'MSE'}
    if state_path.exists():
        ck=torch.load(state_path,map_location='cpu',weights_only=True)
        assert ck['settings']==settings, '训练设置已变化；请使用新实验目录，避免混接 checkpoint'
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start,best,history=ck['step'],ck['best'],ck['history']
        print('继续训练:',mode,start,'步；随机裁剪/批次序列重新开始，不保证逐位复现中断轨迹')
    else:
        initial=evaluate_fusion(net,valset,F_DEVICE,mode)
        best=float(np.mean([r['psnr'] for r in initial]))
        torch.save({'model':net.state_dict(),'mode':mode,'step':0,'val_psnr':best,'settings':settings},out/'best.pt')
        history.append({'step':0,'train_mse':None,'val_psnr':best})
    loader=DataLoader(trainset,batch_size=F_BATCH,shuffle=True,num_workers=0)
    it=iter(loader); begin=time.perf_counter()
    print(mode,'可训练参数:',sum(p.numel() for p in net.parameters()))
    for step in tqdm(range(start+1,F_STEPS+1),desc=f'融合训练 {mode}'):
        try: batch=next(it)
        except StopIteration: it=iter(loader); batch=next(it)
        x,c,lo,g=[v.to(F_DEVICE) for v in batch]
        net.train(); opt.zero_grad(set_to_none=True)
        pred=net(x,c,lo); loss=F.mse_loss(pred,g)
        loss.backward(); opt.step()
        if step%F_VALIDATE_EVERY==0 or step==F_STEPS:
            metrics=evaluate_fusion(net,valset,F_DEVICE,mode)
            score=float(np.mean([r['psnr'] for r in metrics]))
            history.append({'step':step,'train_mse':float(loss.detach()),'val_psnr':score})
            if score>best:
                best=score
                torch.save({'model':net.state_dict(),'mode':mode,'step':step,'val_psnr':score,'settings':settings},out/'best.pt')
            payload={'model':net.state_dict(),'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,'settings':settings}
            torch.save(payload,out/'last.tmp'); (out/'last.tmp').replace(state_path)
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f'{mode} step={step} val={score:.4f} best={best:.4f} elapsed={time.perf_counter()-begin:.0f}s')
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['model'])
    records=evaluate_fusion(net,valset,F_DEVICE,mode)
    pd.DataFrame(records).to_csv(out/'val_per_image.csv',index=False)
    save_fusion_preview(net,valset,preferred,out,F_DEVICE)
    return records

def run_fusion_training():
    for name in ['backbone','all_train','val_pairs','ROOT']:
        assert name in globals(), f'缺少 {name}，先运行原 Notebook 第 1～4 节'
    assert F_REGION>=F_CROP and F_REGION%32==0 and F_CROP>=16
    if F_DEVICE=='cuda': assert torch.cuda.is_available()
    manifest=fusion_manifest(all_train,val_pairs,F_TRAIN_IMAGES)
    model=backbone.to(F_DEVICE).eval().requires_grad_(False)
    # 对实际已加载权重计算哈希，不依赖旧单元格中的变量是否仍准确。
    digest=hashlib.sha256()
    for key,v in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    cfg={'manifest':manifest,'region':F_REGION,'scales':F_SCALES,'seed':F_SEED,
         'weight_sha256':digest.hexdigest(),'protocol':'same_view_fusion_v1','lowpass':'gaussian9_sigma2',
         'commit':globals().get('commit','unknown')}
    ident=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    folder=Path(ROOT)/('fusion_'+ident); folder.mkdir(parents=True,exist_ok=True)
    (folder/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print('本次实验:',folder,'训练/验证:',len(manifest['train']),len(manifest['val']))
    print('不读取 Test；每图只使用一个中心区域；验证成绩不等同于论文整图成绩。')
    print('100 张训练图只代表本轮可行性实验；正式泛化结论需要独立测试。')
    train_paths=create_fusion_cache(model,manifest['train'],'train',folder)
    val_paths=create_fusion_cache(model,manifest['val'],'val',folder)
    model.cpu()
    if F_DEVICE=='cuda': torch.cuda.empty_cache()
    trainset=FusionData(train_paths,F_CROP); valset=FusionData(val_paths)
    baselines=[]
    for method in ['scale_0','scale_1','scale_2','mean']:
        baselines.extend(evaluate_fusion(None,valset,F_DEVICE,method))
    frame=pd.DataFrame(baselines)
    baseline_mean=frame.groupby('method').psnr.mean()
    preferred=int(baseline_mean[[f'scale_{i}' for i in range(3)]].idxmax().split('_')[1])
    frame.to_csv(folder/'baselines_per_image.csv',index=False)
    print('候选编号对应尺度:',dict(enumerate(F_SCALES)))
    print('验证集最佳固定尺度:',F_SCALES[preferred]); display(baseline_mean.to_frame())
    records=list(baselines)
    if F_DO_TRAIN:
        for mode in F_MODES:
            records.extend(train_fusion(mode,trainset,valset,preferred,folder/mode))
            if F_DEVICE=='cuda': torch.cuda.empty_cache()
    result=pd.DataFrame(records)
    table=result.groupby('method').psnr.mean().sort_values(ascending=False)
    result.to_csv(folder/'comparison_per_image.csv',index=False)
    table.to_csv(folder/'comparison_summary.csv')
    print('开发验证集最终对照（已用于模型选择，不是独立测试成绩）:'); display(table.to_frame())
    bundle=folder/'fusion_training_results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob('*'):
            if f.is_file() and 'cache' not in f.relative_to(folder).parts and f.suffix in ('.csv','.json','.png','.pt'):
                z.write(f,str(f.relative_to(folder)))
    display(FileLink(str(bundle)))
    print('跨会话续跑还需保存完整 cache；此 ZIP 不包含候选缓存。')
    return folder,table

if __name__=='__main__':
    FUSION_DIR,FUSION_TABLE=run_fusion_training()
