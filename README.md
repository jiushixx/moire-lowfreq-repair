# 轻量去摩尔纹：多尺度融合与空间低频修复

基于冻结 ESDNet 的图像去摩尔纹项目，结合多尺度分频融合与小样本空间颜色修复，在 UHDM 和 DCID 上开展实验。

## 方法

- **多尺度分频融合**：融合0.5、0.75、1.0尺度预测，分别学习低频与高频残差的融合权重，用于UHDM。
- **均值校正**：利用输入与预测的RGB均值差校正整体色偏，作为DCID修复基线。
- **空间低频修复**：约5.99万参数的网络在96×96网格上预测RGB修正场，上采样后加到校正结果上。主干保持冻结，修复头使用MSE与TV损失训练。

## 测试结果

从作者test目录固定抽取UHDM、DCID各20对图像，评估原分辨率中心768×768区域；DCID取自`test/iphone_1`。指标为逐图平均值，模型在测试前确定。

| 数据 | 方法 | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|
| UHDM | ESDNet | 21.7950 | 0.762888 |
| UHDM | 分频融合 | **21.9399** | **0.764870** |
| DCID | ESDNet | 18.7479 | 0.815882 |
| DCID | 均值校正 | 23.6502 | 0.831743 |
| DCID | 均值校正 + 空间修复 | **23.8751** | **0.832569** |

UHDM分频融合提升0.1448 dB；DCID空间修复在均值校正之上提升0.2249 dB。

## 项目结构

```text
src/moire_project/   模型、训练与评价
configs/             实验配置
scripts/             运行入口
notebooks/           Kaggle Notebook
experiments/         源域融合、对照与探索实验
results/             开发及测试结果
models/              权重说明
tests/               模型接口测试
```

## 运行

安装依赖，在配置文件中填写数据和权重路径：

```bash
pip install -e .
python scripts/run.py evaluate --config configs/frozen_test.json
python scripts/run.py train-repair --config configs/train_repair.json
```

Kaggle可直接使用`notebooks/`中的Notebook。数据与权重单独准备，见[运行说明](docs/REPRODUCE.md)。

[实验设计](docs/EXPERIMENTS.md) · [来源与致谢](THIRD_PARTY.md)
