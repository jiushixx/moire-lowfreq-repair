"""无需 GPU。python CPU数据清单审计.py manifest.csv --out audit
CSV 必需列：input,target,domain,device,scene_id,split。
相对路径相对于 CSV 所在目录；场景标识应跨设备/来源保持一致。
本脚本检查精确重复，不能识别近重复、视角变化或判断配准正确性。
"""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import defaultdict
from PIL import Image

def audit(manifest,out):
    manifest=Path(manifest).resolve(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'input','target','domain','device','scene_id','split'}
        assert required<=set(reader.fieldnames or []),f'缺少列：{required-set(reader.fieldnames or [])}'
        rows=list(reader)
    assert rows,'清单为空'
    issues=[]; hashes=defaultdict(set); scenes=defaultdict(set); counts=defaultdict(int); checked=[]
    allowed={'train','val','test','adapt_train','adapt_val','source_train','source_val','source_test'}
    for i,row in enumerate(rows,2):
        for key in required:
            if not row[key].strip(): issues.append({'row':i,'type':'empty_field','detail':key})
        split=row['split'].strip()
        if split not in allowed: issues.append({'row':i,'type':'unexpected_split','detail':split})
        counts[(row['domain'],row['device'],split)]+=1
        if row['scene_id'].strip(): scenes[row['scene_id'].strip()].add(split)
        sizes={}
        for key in ['input','target']:
            p=Path(row[key]);p=p if p.is_absolute() else manifest.parent/p
            try:
                with Image.open(p) as img: img.verify()
                with Image.open(p) as img: sizes[key]=img.size
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                hashes[digest].add(split)
            except Exception as error:
                issues.append({'row':i,'type':'file_error','detail':f'{key}: {error}'})
        if len(sizes)==2 and sizes['input']!=sizes['target']:
            issues.append({'row':i,'type':'size_mismatch','detail':str(sizes)})
        checked.append({'row':i,**row,'input_size':str(sizes.get('input')),'target_size':str(sizes.get('target'))})
    # 任何跨 split 内容均报告，交由人工判断；source_train/adapt_train 并不一定构成泄漏。
    scene_conflicts={k:sorted(v) for k,v in scenes.items() if len(v)>1}
    exact_conflicts={k:sorted(v) for k,v in hashes.items() if len(v)>1}
    report={'rows':len(rows),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'counts':[dict(domain=k[0],device=k[1],split=k[2],images=v) for k,v in counts.items()],
      'issues':issues,'cross_split_scenes':scene_conflicts,'cross_split_exact_files':exact_conflicts,
      'limitation':'精确重复检查不证明场景独立；需人工核对元数据、近重复、配准。'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'checked_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(checked[0]));writer.writeheader();writer.writerows(checked)
    print(json.dumps({'rows':len(rows),'file_or_field_issues':len(issues),
      'cross_split_scenes':len(scene_conflicts),'cross_split_exact_files':len(exact_conflicts),
      'report':str((out/'report.json').resolve())},ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('--out',default='audit')
    args=parser.parse_args();audit(args.manifest,args.out)
