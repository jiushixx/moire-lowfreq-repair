# 模型权重

| 文件 | 用途 |
|---|---|
| `moire_frozen_test_models.zip` | 分频融合头、DCID第100步空间头与测试配置 |
| `fusion_training_results.zip` | 原融合头及训练清单，用于训练与缓存构建 |

权重包单独提供，校验值见`frozen_protocol.json`。ESDNet主干权重由程序获取，也可通过`WEIGHT`指定。

## 三设备空间修复头

`dcid_balanced300.pt`为本项目三设备均衡训练选出的第300步权重，59,883参数。包含`model`、`step`、`config`与`parameters`字段，使用`IndependentRepair('DCID')`严格加载。对应结果见`results/balanced/`。
