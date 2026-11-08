# CPU 可运行。先在 Notebook 单独运行：%pip install -q huggingface_hub pillow
# 然后把本文件全部复制到另一个单元格。默认只列清单，不下载图片。
from pathlib import Path,PurePosixPath
import json,csv,random
from huggingface_hub import HfApi,hf_hub_download
from PIL import Image

REPO='Mrduckk/DCID'
SUBDIR='train/iphone_1/part1'
PAIRS=20
SEED=2026
DO_DOWNLOAD=False   # 先查看总大小，确认后改为 True 再运行
LIMIT_MB=600        # 所选文件总大小上限；超过即停止，不自动下载更多文件
OUT=(Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'work')/'dcid_hf_subset'

def select_pairs(files,n,seed):
    # files: {repo relative path: bytes}; 只配 _gt.jpg 和同目录 _m.jpg。
    found=[]
    for name in sorted(files):
        if name.endswith('_gt.jpg'):
            src=name[:-len('_gt.jpg')]+'_m.jpg'
            if src in files: found.append({'input':src,'target':name})
    random.Random(seed).shuffle(found)
    if len(found)<n:
        raise RuntimeError(f'该目录仅找到 {len(found)} 对，少于要求 {n}；请降低 PAIRS 或修改 SUBDIR。')
    return found[:n]

def run():
    assert PAIRS>0 and SUBDIR.split('/')[0]=='train','本轮只取作者 train，不动 test'
    api=HfApi()
    # 首次解析并记录完整 commit；后续沿用锁文件，避免远程 main 更新改变样本。
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'revision.json'
    if lock.exists():
        info=json.loads(lock.read_text())
        assert info['repo']==REPO,'锁文件属于其他数据集，请另选输出目录'
        revision=info['revision']
    else:
        revision=api.dataset_info(REPO).sha
        lock.write_text(json.dumps({'repo':REPO,'revision':revision},indent=2))
    print('读取文件元数据（此步骤不下载图片）:',SUBDIR)
    files={}
    for item in api.list_repo_tree(REPO,path_in_repo=SUBDIR,repo_type='dataset',revision=revision,recursive=True):
        size=getattr(item,'size',None)
        if size is not None: files[item.path]=int(size)
    chosen=select_pairs(files,PAIRS,SEED)
    total=sum(files[p[k]] for p in chosen for k in ['input','target'])
    print('版本:',revision,'图像对:',len(chosen),'图片数:',len(chosen)*2)
    print(f'所选文件总量: {total/1024**2:.1f} MiB（实际网络传输可能有额外开销）')
    for p in chosen[:5]: print(p)
    label=SUBDIR.replace('/','_')+f'_{PAIRS}_{SEED}'
    plan={'repo':REPO,'revision':revision,'subdir':SUBDIR,'pairs':chosen,'bytes':total,'seed':SEED}
    (OUT/(label+'_plan.json')).write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if total>LIMIT_MB*1024**2:
        raise RuntimeError('超过 LIMIT_MB，尚未下载图片；请减少 PAIRS。')
    if not DO_DOWNLOAD:
        print('当前只生成清单。把 DO_DOWNLOAD 改为 True，重跑即可获取这些图片。')
        return plan
    rows=[]
    for i,p in enumerate(chosen,1):
        local={}
        for key in ['input','target']:
            local[key]=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=p[key],
                revision=revision,local_dir=str(OUT/'images'))
        with Image.open(local['input']) as a,Image.open(local['target']) as b:
            a.load();b.load()
            if a.size!=b.size: raise RuntimeError(f'尺寸不一致：{p}')
        device=PurePosixPath(p['target']).parts[1].split('_')[0]
        rows.append({'sample_id':p['target'][:-len('_gt.jpg')],**local,'domain':'DCID',
                     'device':device,'scene_id':'','split':'train','device_verified':'no','scene_status':'pending'})
        print(f'{i}/{len(chosen)} 对完成')
    dest=OUT/(label+'_manifest_draft.csv')
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('草稿清单:',dest)
    print('原始数据结构已保留；未下载超广角图片，未自动切分验证/测试。')
    return plan

if __name__=='__main__':
    HF_SUBSET_PLAN=run()
