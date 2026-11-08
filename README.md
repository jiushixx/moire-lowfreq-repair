# 轻量去摩尔纹：多尺度融合与空间低频修复

基于冻结 ESDNet，研究 UHDM 多尺度分频融合，以及 DCID 三设备上的空间低频颜色修复。

## 方法

- UHDM：融合0.5、0.75、1.0尺度预测，为低频和高频残差分别学习权重。
- DCID：先利用输入与预测的RGB均值差校正整体色偏，再用59,883参数的网络预测96×96空间修正场。
- 三设备训练：iPhone、小米、P40各12对训练、8对验证，每个训练批次包含三种设备，主干保持冻结。

## 测试结果

UHDM固定20对，DCID三设备各固定20对，均评价原分辨率中心768×768区域。表中为逐图平均PSNR，单位dB。

| 数据 | ESDNet | 均值校正 | iPhone训练修复头 | 三设备训练修复头 |
|---|---:|---:|---:|---:|
| DCID iPhone | 18.7479 | 23.6502 | 23.8751 | **24.0994** |
| DCID 小米 | 19.6945 | 20.8213 | 20.8273 | **20.9895** |
| DCID P40 | 16.5252 | 23.2040 | 23.1928 | **23.6206** |
| 三设备平均 | 18.3225 | 22.5585 | 22.6317 | **22.9032** |

三设备训练修复头相对均值校正平均提高0.3446 dB。iPhone与P40各17/20张改善，小米8/20张改善。小米的平均SSIM由0.848148变为0.848074，逐图表现见结果记录。

UHDM分频融合的PSNR由21.7950提高至**21.9399 dB**，SSIM由0.762888提高至**0.764870**。

## 代码与运行

```text
src/moire_project/   主干接口、融合、空间修复、三设备训练
configs/             实验配置
scripts/             命令行入口
notebooks/           Kaggle Run All
experiments/         对照与前期探索
results/             开发、初始测试、跨设备与均衡训练结果
models/              修复头权重与模型说明
```

Kaggle导入 `notebooks/04_multidevice_training.ipynb`，添加 `moire_frozen_test_models.zip`，开启GPU与Internet后Run All。数据按需从Hugging Face获取。

```bash
pip install -e .
python scripts/run_balanced.py --config configs/multidevice_train.json
```

[实验设计](docs/EXPERIMENTS.md) · [结果分析](docs/MULTIDEVICE_RESULTS.md) · [运行说明](docs/REPRODUCE.md) · [来源与致谢](THIRD_PARTY.md)

[完整项目报告](docs/项目报告.docx)
