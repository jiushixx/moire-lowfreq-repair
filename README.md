# 轻量去摩尔纹：多尺度融合与空间低频修复

图像处理课程项目。以冻结的 ESDNet 为基础，研究在小样本与有限算力下如何改善去摩尔纹结果；主要方法是**来源条件化的空间低频修复**，UHDM 分频融合为已有源域路线，门控融合为开发对照。

## 方法脉络

1. **UHDM：多尺度分频融合。** 同一中心768视野产生0.5、0.75、1.0尺度预测；高斯滤波拆为低频与残差，轻量网络分别预测融合权重。
2. **DCID：先建立强基线。** 对冻结 ESDNet 输出做输入均值锚定：`clamp(pred + mean(input - pred), 0, 1)`。这是无需GT的整体颜色校正，不适合直接套到UHDM。
3. **DCID：空间修复。** 输入、原始预测、强基线及输入与强基线之差组成12通道，缩至96×96；约5.99万参数小网络预测有界RGB修正场，再上采样至768。零初始化保持强基线，训练MSE加TV平滑。
4. **冻结测试。** DCID使用开发验证选出的第100步模型，UHDM保留原分频融合；测试不训练、不重选模型。数据来源由用户提供，不做未知域自动识别。

## 已取得的结果

作者test目录固定抽样，各20对；原分辨率中心768，逐图PSNR/SSIM取平均。不是官方全量整图成绩。

| 数据 | 方法 | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|
| UHDM | ESDNet | 21.7950 | 0.762888 |
| UHDM | 分频融合（最终） | **21.9399** | **0.764870** |
| DCID | ESDNet | 18.7479 | 0.815882 |
| DCID | 输入均值锚定 | 23.6502 | 0.831743 |
| DCID | 均值锚定 + 空间修复（最终） | **23.8751** | **0.832569** |

UHDM融合相对ESDNet +0.1448 dB，14/20图改善；DCID空间修复相对均值锚定 +0.2249 dB，19/20图改善。DCID的大部分提升来自均值校正，空间头提供额外的小幅改善。测试前登记中均值锚定是DCID主比较，空间头为预先冻结的次比较；此处展示最终保留方法，不改写原登记。

## 代码导航

```text
src/moire_project/    可读模块，不再使用嵌套exec打包
  baseline.py        官方主干加载、数据读取、预测与指标
  joint.py           三尺度融合定义、旧实验数据衔接及联合微调对照
  calibration.py     预测缓存与全局校准对照
  spatial.py         小型空间网络、缓存校验、共享空间/门控对照
  repair.py          独立空间修复头与训练
  evaluation.py      冻结测试、抽样与哈希审计
configs/             可编辑运行设置
scripts/run.py       模块化实验入口
notebooks/           已执行过的独立Kaggle Run All入口
experiments/         源域训练、对照和早期探索
results/             实测CSV与协议，开发/测试分开
docs/                路线、复现步骤、限制与上传指南
models/              权重说明和校验值（不含二进制）
tests/               CPU模型接口、零初始化与数值一致性检查
```

## 最快复现

Kaggle导入 `notebooks/01_frozen_evaluation.ipynb`，打开GPU与Internet，添加UHDM数据和独立提供的冻结模型包，Run All。它不启动训练。需要重训时使用02训练Notebook，所需原图与衔接包见[运行指南](docs/REPRODUCE.md)。

模块化入口（在本仓库根目录）：

```bash
python -m pip install -e .
python scripts/run.py evaluate --config configs/frozen_test.json
python scripts/run.py train-repair --config configs/train_repair.json
python -m unittest discover -s tests -v
```

默认数据发现与输出目录面向Kaggle。普通Linux环境需明确配置路径，官方仓库与权重仍可能需要网络；不宣称任意Windows机器可直接完成GPU训练。

## 证据边界

- DCID只有单一iPhone目录、小样本，不能声称多设备普适或SOTA。
- 哈希排除相同文件，不保证场景或相邻帧独立；开发集已反复用于探索。
- 未做严格匹配的“仅全局可学习头 vs 空间头”消融，不能把全部收益归因为空间结构。
- 门控对照在开发集运行，未补跑最终test，不能说最终test已证明空间头优于门控。
- 没有实测端到端加速结论；缓存训练省成本不等于部署推理加速。

[完整实验路线](docs/EXPERIMENTS.md) · [上传指南](docs/GITHUB_UPLOAD.md) · [第三方来源](THIRD_PARTY.md)
