# 运行说明

## 环境与数据

推荐Kaggle GPU环境，开启Internet，执行`pip install -e .`。

- UHDM：[Kaggle数据集](https://www.kaggle.com/datasets/soumikrakshit/uhdm-dataset)
- DCID：[Hugging Face数据集](https://huggingface.co/datasets/Mrduckk/DCID)
- 测试权重：`moire_frozen_test_models.zip`，单独提供。
- 空间头训练：DCID原图子集`train_iphone_1_part1_20_2026_manifest_draft_with_images.zip`及融合包`fusion_training_results.zip`，单独提供。

## 测试与训练

编辑`configs/frozen_test.json`或`configs/train_repair.json`中的数据与权重路径，然后运行：

```bash
# 冻结模型测试
python scripts/run.py evaluate --config configs/frozen_test.json

# 空间修复训练
python scripts/run.py train-repair --config configs/train_repair.json

# UHDM融合训练
python scripts/train_source_fusion.py --uhdm-root DATA_ROOT --fusion-bundle FUSION_ZIP
```

也可导入`notebooks/01_frozen_evaluation.ipynb`或`02_train_repair.ipynb`，添加相应数据与权重后Run All。测试脚本按需获取DCID测试子集；训练脚本可通过`CACHE_DIR`复用预测缓存。默认输出位于`/kaggle/working/`。
