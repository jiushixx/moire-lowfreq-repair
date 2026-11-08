# 把全部代码复制到现有 Kaggle Notebook 的新单元格运行。
# A 只需上一轮 candidate_*.npz 缓存；B 还需已经加载的 backbone。
from pathlib import Path
import json, math, hashlib, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity
from IPython.display import display, FileLink

# -------- 配置 --------
PROBE_DIR=Path('/kaggle/working/moire_project/budget_probe_d1957c87f763')
RUN_CONTEXT=True       # 首轮建议 True；只做缓存分析时改 False
CONTEXT_IMAGES=3       # 只给前 3 张生成新候选
AUDIT_BUDGETS=(0.,0.25,0.5,1.)
AUDIT_RANDOM_REPEATS=10
AUDIT_SEED=2026
DEVICE='cuda'
# ----------------------

def mse(a,b): return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
def psnr_from_mse(e): return -10*math.log10(max(float(e),1e-12))
def hwc(a): return a[0].transpose(1,2,0)
def ssim(a,b): return structural_similarity(hwc(a),hwc(b),data_range=1.,channel_axis=2)
def grid_boxes(h,w,block):
    assert h%block==0 and w%block==0
    return [(t,l,t+block,l+block) for t in range(0,h,block) for l in range(0,w,block)]

def error_matrix(base,cands,gt,boxes):
    errors=np.zeros((len(boxes),1+len(cands)))
    for i,(t,l,b,r) in enumerate(boxes):
        for j,pred in enumerate([base]+list(cands)):
            errors[i,j]=mse(pred[:,:,t:b,l:r],gt[:,:,t:b,l:r])
    return errors

def budget_plans(errors,scores,k,seed,repeats):
    # 三类动作：0=保留 base，1=原尺度，2=另一尺度。
    n=len(errors); native_gain=errors[:,0]-errors[:,1]
    best_action=errors[:,1:].argmin(1)+1
    best_gain=errors[:,0]-errors[np.arange(n),best_action]
    def ordered(values): return np.argsort(-np.asarray(values),kind='stable')[:k]
    plans=[]
    for name,values in scores.items():
        chosen=ordered(values)
        actions=np.zeros(n,dtype=int); actions[chosen]=1
        plans.append((name,actions))
        if name=='color_prior':
            # 仅作 GT 辅助诊断，不能作为实际可部署方法。
            rejected=actions.copy(); rejected[native_gain<=0]=0
            plans.append(('color_prior_then_GT_reject',rejected))
    for rep in range(repeats):
        chosen=np.random.default_rng(seed+rep).permutation(n)[:k]
        actions=np.zeros(n,dtype=int); actions[chosen]=1
        plans.append(('random',actions))
    for name,gain,action in [('oracle_native',native_gain,np.ones(n,dtype=int)),
                              ('oracle_scale',best_gain,best_action)]:
        chosen=ordered(gain)
        forced=np.zeros(n,dtype=int); forced[chosen]=action[chosen]
        plans.append((name+'_forced',forced))
        skip=forced.copy(); skip[gain<=0]=0
        plans.append((name+'_allow_skip',skip))
    return plans

def color_align(base,cands,boxes):
    # 固定、无需 GT 的块均值校正。只测试块间色彩偏差是否影响结论。
    result=np.array(cands,copy=True)
    for t,l,b,r in boxes:
        ref=base[:,:,t:b,l:r].mean(axis=(-2,-1),keepdims=True)
        for j in range(len(result)):
            patch=result[j,:,:,t:b,l:r]
            patch+=ref-patch.mean(axis=(-2,-1),keepdims=True)
    return np.clip(result,0,1)

def run_cached_audit(folder):
    assert folder.is_dir(), f'找不到 {folder}。会话若已重启，需挂载上轮完整输出，或先重建候选缓存。'
    cfg=json.loads((folder/'config.json').read_text())
    files=sorted(folder.glob('candidate_*.npz'))
    assert files, '缺少 candidate_*.npz；diagnostic_results.zip 不包含候选缓存。'
    spec={'protocol':'decision_audit_v1','budgets':AUDIT_BUDGETS,'repeats':AUDIT_RANDOM_REPEATS,'seed':AUDIT_SEED}
    ident=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:8]
    out=folder/('audit_'+ident); out.mkdir(exist_ok=True)
    (out/'audit_config.json').write_text(json.dumps(spec,indent=2))
    block=cfg['block']; raw=[]; per_image=[]; wins=[]; candidate_rows=[]
    block_csv=pd.read_csv(folder/'block_gains.csv').set_index(['image','block'])
    for image,f in enumerate(tqdm(files,desc='A：分析已有缓存')):
        with np.load(f) as z:
            x,gt,base,cands=[z[key].copy() for key in ['x','gt','base','candidates']]
        h,w=base.shape[-2:]; boxes=grid_boxes(h,w,block); n=len(boxes)
        features=block_csv.loc[image].sort_index()
        assert len(features)==n
        texture=features.texture.to_numpy(); prior=features.color_frequency_prior.to_numpy()
        scores={'texture':texture,'reverse_texture':-texture,'color_prior':prior,'reverse_color_prior':-prior}
        for variant,cc in [('raw',cands),('mean_aligned',color_align(base,cands,boxes))]:
            errors=error_matrix(base,cc,gt,boxes)
            ebase=float(errors[:,0].mean())
            best_action=errors[:,1:].argmin(1)+1
            gains=errors[:,0]-errors[np.arange(n),best_action]
            for i in range(n):
                wins.append({'image':image,'variant':variant,'block':i,'gain_mse':gains[i],
                             'positive':bool(gains[i]>0),'scale':cfg['scales'][best_action[i]-1]})
            # 全图选一个处理方案的 oracle，与区域动作选择区分开。
            best_whole=float(errors.mean(axis=0).min())
            best_local=float(errors.min(axis=1).mean())
            per_image.append({'image':image,'variant':variant,'base_psnr':psnr_from_mse(ebase),
                'oracle_one_action_psnr':psnr_from_mse(best_whole),
                'oracle_local_psnr':psnr_from_mse(best_local),
                'local_gain_dB':psnr_from_mse(best_local)-psnr_from_mse(ebase),
                'local_vs_whole_dB':psnr_from_mse(best_local)-psnr_from_mse(best_whole)})
            for label,pred in [('base',base)]+[(f'all_scale_{s}',c) for s,c in zip(cfg['scales'],cc)]:
                candidate_rows.append({'image':image,'variant':variant,'method':label,
                    'psnr':psnr_from_mse(mse(pred,gt)),'ssim':ssim(pred,gt)})
            for fraction in AUDIT_BUDGETS:
                k=int(n*fraction)
                for method,actions in budget_plans(errors,scores,k,AUDIT_SEED+image*1000,AUDIT_RANDOM_REPEATS):
                    e=float(errors[np.arange(n),actions].mean())
                    raw.append({'image':image,'variant':variant,'budget':fraction,'max_blocks':k,
                        'method':method,'selected_blocks':int((actions>0).sum()),'psnr':psnr_from_mse(e)})
        if image<3:
            aligned=color_align(base,cands,boxes)
            fig,axes=plt.subplots(1,4,figsize=(16,4))
            for ax,pred,title in zip(axes,[base,cands[0],aligned[0],gt],['Base','Raw local','Mean aligned local','GT']):
                ax.imshow(hwc(pred)); ax.set_title(title); ax.axis('off')
            plt.tight_layout(); plt.savefig(out/f'mean_alignment_{image}.png',dpi=150); plt.show(); plt.close(fig)
    raw=pd.DataFrame(raw)
    avg_image=raw.groupby(['image','variant','budget','method'],as_index=False).mean(numeric_only=True)
    summary=avg_image.groupby(['variant','budget','method'])[['psnr','selected_blocks']].mean()
    raw.to_csv(out/'decisions_raw.csv',index=False)
    avg_image.to_csv(out/'decisions_per_image.csv',index=False)
    summary.to_csv(out/'decisions_summary.csv')
    images=pd.DataFrame(per_image); images.to_csv(out/'oracle_per_image.csv',index=False)
    wr=pd.DataFrame(wins); wr.to_csv(out/'positive_scale_blocks.csv',index=False)
    candidates=pd.DataFrame(candidate_rows); candidates.to_csv(out/'candidate_metrics.csv',index=False)
    print('\nA1：各图潜力，注意是否集中在极少数图上'); display(images)
    print('\nA2：只统计正收益块的尺度计数'); display(wr[wr.positive].groupby(['variant','scale']).size())
    print('\nA3：校正前后的完整候选成绩'); display(candidates.groupby(['variant','method'])[['psnr','ssim']].mean())
    print('\nA4：50% 预算的决策拆分（oracle/GT_reject 不可当实际算法）')
    display(avg_image[avg_image.budget==.5].groupby(['variant','method'])[['psnr','selected_blocks']].mean())
    return out,cfg,files

@torch.inference_mode()
def full_view_predict(model,a,scale,device):
    x=torch.from_numpy(a).to(device); h,w=x.shape[-2:]
    sh,sw=max(1,round(h*scale)),max(1,round(w*scale))
    if (sh,sw)!=(h,w): x=F.interpolate(x,size=(sh,sw),mode='bilinear',align_corners=False,antialias=True)
    x=F.pad(x,(0,(-sw)%32,0,(-sh)%32),mode='replicate')
    result=model(x); y=result[0] if isinstance(result,(tuple,list)) else result
    y=y[:,:,:sh,:sw]
    if (sh,sw)!=(h,w): y=F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
    return y.clamp(0,1).cpu().numpy()

def run_context_audit(out,cfg,files):
    if not RUN_CONTEXT: return
    assert 'backbone' in globals(), 'A 已完成。B 需要加载 backbone；先运行旧 Notebook 的模型加载部分，或设 RUN_CONTEXT=False。'
    if DEVICE=='cuda': assert torch.cuda.is_available()
    model=backbone.to(DEVICE).eval().requires_grad_(False)
    rows=[]; block=cfg['block']
    for image,f in enumerate(tqdm(files[:CONTEXT_IMAGES],desc='B：相同视野尺度检查')):
        with np.load(f) as z: x,gt,base,local=[z[k].copy() for k in ['x','gt','base','candidates']]
        boxes=grid_boxes(*base.shape[-2:],block)
        predictions=[]
        for scale in cfg['scales']:
            save=out/f'full_context_{image}_{scale}.npy'
            try:
                if save.exists(): pred=np.load(save)
                else:
                    pred=full_view_predict(model,x,scale,DEVICE)
                    tmp=save.with_suffix('.tmp')
                    with open(tmp,'wb') as handle: np.save(handle,pred)
                    tmp.replace(save)
                predictions.append((scale,pred))
                rows.append({'image':image,'method':f'full_view_{scale}',
                             'psnr':psnr_from_mse(mse(pred,gt)),'ssim':ssim(pred,gt)})
                j=list(cfg['scales']).index(scale)
                rows.append({'image':image,'method':f'local_view_{scale}',
                             'psnr':psnr_from_mse(mse(local[j],gt)),'ssim':ssim(local[j],gt)})
            except torch.cuda.OutOfMemoryError:
                print(f'图 {image}，尺度 {scale} 显存不足：该项跳过，不缩小评价区域。')
                torch.cuda.empty_cache()
        rows.append({'image':image,'method':'base','psnr':psnr_from_mse(mse(base,gt)),'ssim':ssim(base,gt)})
        if len(predictions)==len(cfg['scales']):
            ee=error_matrix(base,[v for _,v in predictions],gt,boxes)
            rows.append({'image':image,'method':'full_view_local_oracle',
                         'psnr':psnr_from_mse(ee.min(1).mean()),'ssim':np.nan})
        fig,axes=plt.subplots(1,2+len(predictions),figsize=(5*(2+len(predictions)),4),squeeze=False)
        for ax,(label,pred) in zip(axes[0],[('Base',base)]+[(f'Full view {s}',p) for s,p in predictions]+[('GT',gt)]):
            ax.imshow(hwc(pred)); ax.set_title(label); ax.axis('off')
        plt.tight_layout(); plt.savefig(out/f'full_context_{image}.png',dpi=150); plt.show(); plt.close(fig)
    frame=pd.DataFrame(rows); frame.to_csv(out/'context_per_image.csv',index=False)
    print('\nB：相同原始视野下的尺度表现；每种方法的有效图像数必须同时检查')
    display(frame.groupby('method').agg(psnr=('psnr','mean'),ssim=('ssim','mean'),images=('image','nunique')))
    print('如有 OOM，各方法均值可能不是同一批图：请用 context_per_image.csv 对齐后再比较。')

def export_audit(out):
    bundle=out/'second_audit_results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in ('.csv','.json','.png'): z.write(f,f.name)
    display(FileLink(str(bundle)))
    print('输出目录:',out)
    print('全部为验证集探索性诊断；不训练、不使用测试集、不声称实际加速。')

if __name__=='__main__':
    AUDIT_DIR,AUDIT_CONFIG,AUDIT_FILES=run_cached_audit(PROBE_DIR)
    run_context_audit(AUDIT_DIR,AUDIT_CONFIG,AUDIT_FILES)
    export_audit(AUDIT_DIR)
