# 实验设计

## 研究路线

围绕去摩尔纹结果中的纹理残留与颜色偏移，先研究多尺度预测融合，再引入整体颜色校正与空间残差修复。

| 方向 | 内容 | 代码位置 |
|---|---|---|
| 多尺度分频融合 | 分别学习低频与高频残差权重 | `src/moire_project/source_fusion.py` |
| 全局颜色校准 | 输入均值校正、ridge校准 | `src/moire_project/calibration.py` |
| 空间低频修复 | 从各域基线学习平滑RGB修正场 | `src/moire_project/repair.py` |
| 门控对照 | 多候选输出的空间加权 | `experiments/controls/` |
| 前期探索 | 分块、视野、尺度与预算选择 | `experiments/exploration/` |

## 训练设置

| 实验 | 训练/验证图像 | 训练设置 |
|---|---|---|
| UHDM融合 | 100 / 10 | 两种融合头各2000步 |
| 独立空间修复 | DCID 12 / 8；UHDM 20 / 10 | 每域500步，每100步验证 |
| 共享空间头与门控 | DCID 12 / 8；UHDM 20 / 10 | 各1000步 |

空间头采用96×96网格、batch size 4、学习率3e-4、TV系数1e-3。DCID使用验证集选出的第100步权重，UHDM采用分频融合。两域使用各自的处理路径。

## 结果记录

开发对照保存在`results/development/`，包括联合微调、全局校准、共享空间头和门控实验。最终测试的逐图指标、抽样清单和配置保存在`results/frozen/`。
