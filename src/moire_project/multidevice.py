"""Balanced three-device training and fixed test evaluation."""
from . import evaluation as F
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json,random,gc,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from PIL import Image
from huggingface_hub import HfApi,hf_hub_download
from tqdm.auto import tqdm
from IPython.display import display

FROZEN_BUNDLE=None
WEIGHT=None
STEPS=500
EVAL_EVERY=100
SEED=20261108
LR=0.0003
DEVICES=['iphone_1','xiaomi_1','p40_1']

def run():
    assert torch.cuda.is_available(),'请开启GPU'
    F.FROZEN_BUNDLE=FROZEN_BUNDLE
    F.ROOT=Path('/kaggle/working/multidevice_training');F.ROOT.mkdir(parents=True,exist_ok=True)
    out=F.ROOT/datetime.now().strftime('balanced_%Y%m%d_%H%M%S');out.mkdir()
    try:
        frozen,old,band_ck,ck=F.load_bundle()
        I=F.I;H=I.S.C.J.H;J=I.S.C.J
        I.GRID=int(ck['config']['grid']);I.MAX_RESIDUAL=float(ck['config']['max_residual'])
        api=HfApi();plans=[]
        # iPhone沿用原来的12/8划分；新设备各20对，12训练、8验证。
        for r in ck['config']['signature']['samples']:
            if r['domain']=='DCID':
                stem=r['id'].removeprefix('DCID/')
                assert stem.startswith('train/iphone_1/')
                plans.append(dict(device='iphone_1',id=r['id'],split=r['split'],input=stem+'_m.jpg',target=stem+'_gt.jpg',expected_x=r['input_hash'],expected_gt=r['gt_hash']))
        assert len(plans)==20 and sum(r['split']=='train' for r in plans)==12
        for dev in DEVICES[1:]:
            folder='train/'+dev+'/part1'
            files={e.path:int(e.size) for e in api.list_repo_tree(F.HF_REPO,path_in_repo=folder,repo_type='dataset',revision=F.HF_REVISION,recursive=True) if getattr(e,'size',None) is not None}
            chosen=F.choose_pairs(files,20,SEED)
            for idx,p in enumerate(chosen):
                plans.append(dict(device=dev,id='DCID/'+p['target'][:-7],split='train' if idx<12 else 'val',**p))
        config=dict(devices=DEVICES,train_per_device=12,val_per_device=8,steps=STEPS,eval_every=EVAL_EVERY,seed=SEED,lr=LR,
            grid=I.GRID,max_residual=I.MAX_RESIDUAL,tv_weight=0.001,batch='one image per device per step',
            initialization='zero residual',selection='largest mean validation PSNR across devices, including step0',
            repo=F.HF_REPO,revision=F.HF_REVISION,frozen_models=frozen,region=768)
        (out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        (out/'download_plan.json').write_text(json.dumps(plans,indent=2),encoding='utf-8')
        def fetch(r):
            r=dict(r)
            for k in ['input','target']:
                r[k]=hf_hub_download(F.HF_REPO,filename=r[k],repo_type='dataset',revision=F.HF_REVISION,local_dir=str(F.ROOT/'raw'))
                r[k+'_hash']=H.sha_file(r[k])
            if 'expected_x' in r:
                assert r['input_hash']==r['expected_x'] and r['target_hash']==r['expected_gt']
            return r
        with ThreadPoolExecutor(max_workers=4) as executor:
            records=list(tqdm(executor.map(fetch,plans),total=len(plans),desc='下载60对训练/验证图像'))
        hashes=[r[k] for r in records for k in ['input_hash','target_hash']]
        assert len(set(hashes))==len(hashes),'训练/验证出现精确重复图像'
        pd.DataFrame(records).to_csv(out/'manifest.csv',index=False)
        H.ROOT=F.ROOT/'official';H.ROOT.mkdir(exist_ok=True);H.LOCAL_WEIGHT=WEIGHT
        model,meta=H.load_official_model()
        assert meta['weight_sha256']==ck['config']['signature']['meta']['weight_sha256']
        cache=F.ROOT/'cache';cache.mkdir(exist_ok=True)
        def prepare(r):
            key=r['input_hash']+'_'+r['target_hash']+'_'+meta['weight_sha256']
            dest=cache/(key+'.npz')
            if dest.exists():
                with np.load(dest) as d: return dict(r,x=d['x'],g=d['g'],o=d['o'])
            with Image.open(r['input']) as im:a=np.array(im.convert('RGB'))
            with Image.open(r['target']) as im:g=np.array(im.convert('RGB'))
            a,g=J.crop_pair(a,g,r['id'])
            o=H.predict(model,a,1.0).astype(np.float16)
            np.savez(dest,x=a,g=g,o=o)
            return dict(r,x=a,g=g,o=o)
        items=[prepare(r) for r in tqdm(records,desc='缓存冻结主干预测')]
        def tensors(p,flip=0):
            x=J.tensor(p['x']);g=J.tensor(p['g'])
            o=torch.from_numpy(p['o'].astype(np.float32)).permute(2,0,1)[None].cuda()
            dims=([-1] if flip&1 else [])+([-2] if flip&2 else [])
            return [v.flip(dims) if dims else v for v in [x,g,o]]
        def score(net,selection):
            rows=[]
            with torch.inference_mode():
                for p in selection:
                    x,g,o=tensors(p)
                    pred=I.base_for_domain('DCID',x,o,o) if net is None else net(x,o,o)[0].clamp(0,1)
                    a=J.arr(pred).astype(np.float64);b=J.arr(g).astype(np.float64)
                    rows.append(dict(device=p['device'],id=p['id'],psnr=float(-10*np.log10(max(np.mean((a-b)**2),1e-12)))))
            return rows
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        net=I.IndependentRepair('DCID').cuda()
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        val=[p for p in items if p['split']=='val']
        pools={d:[p for p in items if p['device']==d and p['split']=='train'] for d in DEVICES}
        base_rows=score(None,val);best=float(np.mean([r['psnr'] for r in base_rows]));best_step=0
        history=[];val_rows=[dict(r,step=0) for r in base_rows]
        def save_best(step):
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},step=step,config=config,parameters=sum(p.numel() for p in net.parameters())),out/'balanced_best.pt')
        save_best(0);rng=random.Random(SEED);start=time.perf_counter();losses=[]
        for step in tqdm(range(1,STEPS+1),desc='三设备均衡训练'):
            net.train();opt.zero_grad(set_to_none=True)
            samples=[tensors(rng.choice(pools[d]),rng.randrange(4)) for d in DEVICES]
            x,g,o=[torch.cat([s[j] for s in samples]) for j in range(3)]
            y,field=net(x,o,o);mse=TF.mse_loss(y,g);loss=mse+0.001*I.S.tv(field)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();losses.append(float(mse.detach()))
            del samples,x,g,o,y,field,mse,loss
            if step%EVAL_EVERY==0 or step==STEPS:
                net.eval();scored=score(net,val);val_rows.extend(dict(r,step=step) for r in scored)
                per={d:float(np.mean([r['psnr'] for r in scored if r['device']==d])) for d in DEVICES}
                avg=float(np.mean(list(per.values())))
                if avg>best:best=avg;best_step=step;save_best(step)
                history.append(dict(step=step,train_mse=float(np.mean(losses)),val_macro_psnr=avg,best_step=best_step,seconds=time.perf_counter()-start,**per));losses=[]
                pd.DataFrame(history).to_csv(out/'history.csv',index=False)
                pd.DataFrame(val_rows).to_csv(out/'validation_per_image.csv',index=False)
                print('step',step,'验证PSNR',per,'选择步数',best_step)
        # 在训练结束、选择权重后才读取固定测试池，沿用上一轮的目录、种子和抽样规则。
        selected=torch.load(out/'balanced_best.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(selected['model']);net.eval()
        old_net=I.IndependentRepair('DCID').cuda().eval();old_net.load_state_dict(ck['model'])
        (out/'selection.json').write_text(json.dumps(dict(step=best_step,val_macro_psnr=best),indent=2),encoding='utf-8')
        del items;gc.collect();test_rows=[]
        for dev in DEVICES:
            F.HF_SUBDIR='test/'+dev;F.N_DCID=20;F.SEED=SEED
            dest=out/dev;dest.mkdir()
            tests=F.download_target(dest)
            for idx,r in enumerate(tqdm(tests,desc=dev+'固定测试')):
                r['device']=dev;r['split']='test'
                for k in ['input','target']:
                    r[k+'_hash']=H.sha_file(r[k]);assert r[k+'_hash'] not in hashes,'测试与训练/验证精确重复'
                p=prepare(r);x,g,o=tensors(p)
                with torch.inference_mode():
                    outputs={'original':o,'original_mean_anchor':I.base_for_domain('DCID',x,o,o),
                        'iphone_spatial100':old_net(x,o,o)[0].clamp(0,1),'balanced_spatial':net(x,o,o)[0].clamp(0,1)}
                panels=[('Input',J.arr(x))]
                for method,pred in outputs.items():
                    a=J.arr(pred);test_rows.append(dict(device=dev,id=r['id'],method=method,**H.metrics(a,J.arr(g))))
                    if idx<3:
                        pic=dest/'images'/f'{idx:02d}';pic.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(np.round(a*255).astype('uint8')).save(pic/(method+'.png'))
                    panels.append((method,a))
                if idx<3:H.preview(panels+[('GT',J.arr(g))],dest/f'preview_{idx:02d}.jpg')
                pd.DataFrame(test_rows).to_csv(out/'test_per_image.csv',index=False)
            pd.DataFrame(tests).to_csv(dest/'test_manifest.csv',index=False)
        df=pd.DataFrame(test_rows)
        table=df.groupby(['device','method']).agg(psnr=('psnr','mean'),ssim=('ssim','mean'),images=('id','nunique'))
        table.to_csv(out/'test_summary.csv');display(table)
        paired=[]
        for dev,part in df.groupby('device'):
            for metric in ['psnr','ssim']:
                wide=part.pivot(index='id',columns='method',values=metric)
                for ref in ['original_mean_anchor','iphone_spatial100']:
                    delta=wide['balanced_spatial']-wide[ref]
                    paired.append(dict(device=dev,metric=metric,reference=ref,mean_gain=delta.mean(),median_gain=delta.median(),wins=int((delta>0).sum()),images=len(delta)))
        paired=pd.DataFrame(paired);paired.to_csv(out/'paired_comparisons.csv',index=False);display(paired)
    except Exception as error:
        (out/'error.txt').write_text(repr(error),encoding='utf-8');raise
    finally:F.I.S.pack(out)
