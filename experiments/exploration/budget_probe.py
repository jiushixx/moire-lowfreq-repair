# 将此文件全部复制到现有 Kaggle Notebook 的一个新代码单元格运行。
# 先运行旧 Notebook 第 1～4 节以得到 backbone、val_pairs、ROOT。
# 无需旧缓存，不训练、不读取官方测试集。
import json, hashlib, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity

# ---------- 只改这里；首次可 N_IMAGES=3，跑通后改为 10 ----------
N_IMAGES = 10
REGION = 768       # 原分辨率中心区域，不把整幅图缩到 768
BLOCK = 256        # 不重叠评价块；每块推理额外读取 CONTEXT 像素上下文
CONTEXT = 64
SCALES = (1.0, 0.75)
BASE_SCALE = 0.5
BUDGETS = (0.0, 0.25, 0.5, 1.0)
RANDOM_REPEATS = 20
DIAG_SEED = 2026
DEVICE = 'cuda'
# 本次只测单块替换的可加性收益；融合与边界另设后续实验，不在这里隐藏接缝。

def as_tensor(image):
    a=np.asarray(image).copy()
    return torch.from_numpy(a).permute(2,0,1)[None].float()/255

def low(x):
    return F.avg_pool2d(F.pad(x,(2,2,2,2),mode='replicate'),5,stride=1)

@torch.inference_mode()
def predict_scaled(model, x, scale=1., device='cuda'):
    x=x.to(device)
    h,w=x.shape[-2:]
    size=(max(1,round(h*scale)),max(1,round(w*scale)))
    if size!=(h,w): x=F.interpolate(x,size=size,mode='bilinear',align_corners=False,antialias=True)
    sh,sw=x.shape[-2:]
    x=F.pad(x,(0,(-sw)%32,0,(-sh)%32),mode='replicate')
    result=model(x)
    y=result[0] if isinstance(result,(tuple,list)) else result
    y=y[:,:,:sh,:sw]
    if (sh,sw)!=(h,w): y=F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
    return y.clamp(0,1).cpu()

def timed_predict(model,x,scale):
    if DEVICE=='cuda': torch.cuda.synchronize()
    begin=time.perf_counter()
    y=predict_scaled(model,x,scale,DEVICE)
    if DEVICE=='cuda': torch.cuda.synchronize()
    return y,time.perf_counter()-begin

def locations(h,w,block):
    assert h%block==0 and w%block==0
    return [(t,l,t+block,l+block) for t in range(0,h,block) for l in range(0,w,block)]

def select_blocks(scores,k):
    # 固定预算，对可见特征按分数排序；不使用 GT。
    return np.argsort(-np.asarray(scores),kind='stable')[:k]

def oracle_plan(base,candidates,gt,boxes,k,allowed=(0,1)):
    # 用 GT 计算的诊断上界，仅限当前候选集、块划分和 MSE 目标。
    gains=[]; actions=[]
    for t,l,b,r in boxes:
        target=gt[:,:,t:b,l:r]
        base_error=float((base[:,:,t:b,l:r]-target).square().mean())
        errors=[float((candidates[a][:,:,t:b,l:r]-target).square().mean()) for a in allowed]
        j=int(np.argmin(errors))
        gains.append(base_error-errors[j]); actions.append(allowed[j])
    chosen=[int(i) for i in select_blocks(gains,k) if gains[i]>0]
    return chosen,actions,np.asarray(gains)

def compose(base,candidates,boxes,chosen,actions):
    y=base.clone()
    for i in chosen:
        t,l,b,r=boxes[i]
        y[:,:,t:b,l:r]=candidates[actions[i]][:,:,t:b,l:r]
    return y

def measure(y,g):
    a=y[0].permute(1,2,0).numpy(); b=g[0].permute(1,2,0).numpy()
    return (-10*math.log10(max(float((y-g).square().mean()),1e-12)),
            structural_similarity(a,b,data_range=1.,channel_axis=2))

def visible_scores(x,base,boxes):
    result={'texture':[], 'color_frequency_prior':[]}
    hp=(x-low(x)).abs()
    for t,l,b,r in boxes:
        patch=x[0,:,t:b,l:r]
        rg=patch[0]-patch[1]; yb=(patch[0]+patch[1])/2-patch[2]
        color=torch.sqrt(rg.var(unbiased=False)+yb.var(unbiased=False))
        color+=.3*torch.sqrt(rg.mean().square()+yb.mean().square())
        freq=float(hp[:,:,t:b,l:r].mean())
        result['texture'].append(freq)
        # 简化颜色×高通强度先验，属于 DDA 思路对照，非完整 DDA 复现。
        result['color_frequency_prior'].append(float(color)*freq)
    return result

def run_diagnostic():
    assert 'backbone' in globals() and 'val_pairs' in globals(), '先运行原 Notebook 第 1～4 节'
    assert REGION%BLOCK==0 and N_IMAGES<=len(val_pairs)
    if DEVICE=='cuda': assert torch.cuda.is_available()
    model=backbone.to(DEVICE).eval().requires_grad_(False)
    selected=val_pairs[:N_IMAGES]
    config={'region':REGION,'block':BLOCK,'context':CONTEXT,'scales':SCALES,'base_scale':BASE_SCALE,
            'budgets':BUDGETS,'seed':DIAG_SEED,'items':selected,'protocol':'budget_probe_v1',
            'commit':globals().get('commit','unknown'),'weight_sha':globals().get('weight_sha','unknown')}
    ident=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    folder=Path(ROOT)/('budget_probe_'+ident); folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    # 不复用旧版分块缓存。当前独立缓存保存候选与实测时长。
    predict_scaled(model,torch.zeros(1,3,BLOCK+2*CONTEXT,BLOCK+2*CONTEXT),1.,DEVICE)
    summary=[]; block_rows=[]; scale_rows=[]
    for index,p in enumerate(tqdm(selected,desc='尺度与预算诊断')):
        cache=folder/f'candidate_{index:03d}.npz'
        if not cache.exists():
            image=as_tensor(Image.open(p['x']).convert('RGB'))
            truth=as_tensor(Image.open(p['gt']).convert('RGB'))
            assert image.shape==truth.shape, '输入和 GT 尺寸不同'
            h,w=image.shape[-2:]; assert min(h,w)>=REGION
            t,l=(h-REGION)//2,(w-REGION)//2
            x=image[:,:,t:t+REGION,l:l+REGION]; gt=truth[:,:,t:t+REGION,l:l+REGION]
            base,base_seconds=timed_predict(model,x,BASE_SCALE)
            boxes=locations(REGION,REGION,BLOCK)
            candidates=[torch.empty_like(x) for _ in SCALES]
            seconds=np.zeros((len(SCALES),len(boxes)))
            for i,(t,l,b,r) in enumerate(boxes):
                tt,ll,bb,rr=max(0,t-CONTEXT),max(0,l-CONTEXT),min(REGION,b+CONTEXT),min(REGION,r+CONTEXT)
                for a,scale in enumerate(SCALES):
                    pred,seconds[a,i]=timed_predict(model,x[:,:,tt:bb,ll:rr],scale)
                    candidates[a][:,:,t:b,l:r]=pred[:,:,t-tt:b-tt,l-ll:r-ll]
            tmp=cache.with_suffix('.tmp')
            with open(tmp,'wb') as handle:
                np.savez_compressed(handle,x=x.numpy(),gt=gt.numpy(),base=base.numpy(),
                    candidates=torch.stack(candidates).numpy(),seconds=seconds,base_seconds=base_seconds)
            tmp.replace(cache)
        with np.load(cache) as z:
            x,gt,base=[torch.from_numpy(z[k].copy()) for k in ['x','gt','base']]
            candidates=list(torch.from_numpy(z['candidates'].copy()).unbind(0))
            seconds=z['seconds'].copy(); base_seconds=float(z['base_seconds'])
        boxes=locations(REGION,REGION,BLOCK); n=len(boxes)
        scores=visible_scores(x,base,boxes)
        all_chosen,all_actions,gains=oracle_plan(base,candidates,gt,boxes,n)
        for i,(t,l,b,r) in enumerate(boxes):
            target=gt[:,:,t:b,l:r]
            e0=float((base[:,:,t:b,l:r]-target).square().mean())
            errors=[float((c[:,:,t:b,l:r]-target).square().mean()) for c in candidates]
            block_rows.append({'image':index,'block':i,'best_action':all_actions[i],
                'gain_mse':float(gains[i]),'native_gain_mse':e0-errors[0],
                'texture':scores['texture'][i],'color_frequency_prior':scores['color_frequency_prior'][i]})
        for label,pred in [('base',base)]+[(f'all_scale_{s}',c) for s,c in zip(SCALES,candidates)]:
            psnr,ssim=measure(pred,gt)
            scale_rows.append({'image':index,'method':label,'psnr':psnr,'ssim':ssim})
        for budget in BUDGETS:
            k=int(math.floor(n*budget))
            plans=[]
            for name,values in scores.items(): plans.append((name,select_blocks(values,k),[0]*n))
            for rep in range(RANDOM_REPEATS):
                rng=np.random.default_rng(DIAG_SEED+index*1000+rep)
                plans.append(('random',rng.permutation(n)[:k],[0]*n))
            for name,allowed in [('oracle_native',(0,)),('oracle_two_scales',tuple(range(len(SCALES))))]:
                chosen,actions,_=oracle_plan(base,candidates,gt,boxes,k,allowed)
                plans.append((name,chosen,actions))
            for name,chosen,actions in plans:
                pred=compose(base,candidates,boxes,chosen,actions)
                psnr,ssim=measure(pred,gt)
                # 成本仅为已测单动作耗时之和，不是实际自适应系统端到端耗时。
                estimate=base_seconds+sum(seconds[actions[i],i] for i in chosen)
                summary.append({'image':index,'budget_fraction':budget,'max_blocks':k,'method':name,
                    'selected_blocks':len(chosen),'psnr':psnr,'ssim':ssim,'estimated_seconds':estimate})
            if index<3 and budget==.5:
                chosen,actions,_=oracle_plan(base,candidates,gt,boxes,k)
                best=compose(base,candidates,boxes,chosen,actions)
                fig,axes=plt.subplots(1,5,figsize=(20,4))
                for ax,img,title in zip(axes,[x,base,candidates[0],best,gt],
                       ['Input','Cheap base','All native blocks','Oracle 50% (GT used)','GT']):
                    ax.imshow(img[0].permute(1,2,0)); ax.set_title(title); ax.axis('off')
                plt.tight_layout(); plt.savefig(folder/f'preview_{index}.png',dpi=150); plt.show(); plt.close(fig)
    raw=pd.DataFrame(summary)
    # 先对同一图的随机重复求平均，再对图像求平均，避免重复数影响图像权重。
    per_image=raw.groupby(['image','budget_fraction','method'],as_index=False).mean(numeric_only=True)
    table=per_image.groupby(['budget_fraction','method'])[['psnr','ssim','selected_blocks','estimated_seconds']].mean()
    raw.to_csv(folder/'budget_raw.csv',index=False)
    per_image.to_csv(folder/'budget_per_image.csv',index=False)
    table.to_csv(folder/'budget_summary.csv')
    blocks=pd.DataFrame(block_rows); blocks.to_csv(folder/'block_gains.csv',index=False)
    scales=pd.DataFrame(scale_rows); scales.to_csv(folder/'scale_per_image.csv',index=False)
    display(scales.groupby('method')[['psnr','ssim']].mean())
    display(table)
    print('正收益块比例:',float((blocks.gain_mse>0).mean()))
    print('所有块最佳候选尺度计数（含仍不如 base 的块）:',blocks.best_action.value_counts().to_dict())
    print('特征与额外修复收益的 Spearman 相关（探索性，块并不独立）:')
    display(blocks[['texture','color_frequency_prior','native_gain_mse']].corr(method='spearman'))
    print('结果目录:',folder)
    print('注意：oracle 使用 GT；estimated_seconds 是缓存动作耗时估计，不能声称真实加速。')
    print('所有成绩只对应 768 原分辨率中心区域；不是官方 UHDM 整图成绩。')
    import zipfile
    bundle=folder/'diagnostic_results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in folder.iterdir():
            if f.suffix in ('.csv','.json','.png'): z.write(f,f.name)
    from IPython.display import FileLink
    display(FileLink(str(bundle)))
    return folder,table

if __name__=='__main__':
    DIAG_DIR,BUDGET_TABLE=run_diagnostic()
