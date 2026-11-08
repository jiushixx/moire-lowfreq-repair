"""Retrain source fusion on the recorded 100/10 split, without old notebook globals."""
import argparse
from pathlib import Path
from moire_project import source_fusion as train, joint, baseline

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uhdm-root', type=Path, required=True)
    p.add_argument('--fusion-bundle', type=Path, required=True)
    p.add_argument('--weight', type=Path)
    p.add_argument('--output', type=Path, default=Path('/kaggle/working/source_fusion'))
    args = p.parse_args()
    joint.FUSION_ZIP = args.fusion_bundle
    protocol, _, _ = joint.load_old_fusion()
    manifest = {}
    for split, rows in protocol['manifest'].items():
        manifest[split] = [dict(row, x=str(joint.source_path(row['x'], args.uhdm_root)),
                                gt=str(joint.source_path(row['gt'], args.uhdm_root))) for row in rows]
    assert len(manifest['train']) == 100 and len(manifest['val']) == 10
    hashed = {split: [dict(gt_hash=baseline.sha_file(r['gt']),
                          input_hash=baseline.sha_file(r['x'])) for r in rows]
              for split, rows in manifest.items()}
    joint.check_split(hashed['train'], hashed['val'])
    # Validate original duplicate/group rules before preserving recorded order.
    train.fusion_manifest(manifest['train'], manifest['val'], 100)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline.ROOT = args.output / 'official'
    baseline.ROOT.mkdir(exist_ok=True)
    baseline.LOCAL_WEIGHT = args.weight
    train.backbone, meta = baseline.load_official_model()
    from moire_project.calibration import fingerprint
    assert fingerprint(train.backbone) == protocol['weight_sha256'], 'Backbone mismatch'
    train.commit = meta.get('commit', 'unknown')
    train.ROOT = args.output
    train.all_train, train.val_pairs = manifest['train'], manifest['val']
    train.fusion_manifest = lambda *args: manifest
    train.run_fusion_training()

if __name__ == '__main__':
    main()
