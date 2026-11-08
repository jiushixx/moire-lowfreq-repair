# 详细运行步骤

## A. 只复现最终测试（推荐）
1. 新建Kaggle Notebook，导入notebooks/01_frozen_evaluation.ipynb；启用T4 GPU及Internet。
2. 添加 https://www.kaggle.com/datasets/soumikrakshit/uhdm-dataset （必须含train和test；train用于旧训练文件排重）。
3. 把单独交付的moire_frozen_test_models.zip作为Input。Kaggle自动解压也支持。官方主干权重自动下载，或显式指定WEIGHT。
4. Run All。自动从锁定HF版本的test/iphone_1获取20对，最多200MiB；不下载全库。
5. 输出在/kaggle/working/frozen_moire_test/frozen_*，下载结果ZIP。不要按新成绩改种子或重抽样。
6. 模块化复现时，先安装本项目；编辑configs/frozen_test.json中的FROZEN_BUNDLE、UHDM_ROOT、WEIGHT（明确绝对路径）。None/null只在自动搜索结果唯一时有效。然后执行README中的evaluate命令。

## B. 重跑空间修复训练
1. 导入notebooks/02_train_repair.ipynb；开启GPU和Internet。
2. 添加UHDM数据、train_iphone_1_part1_20_2026_manifest_draft_with_images.zip（DCID20对原图）、fusion_training_results.zip（旧融合头及训练协议）。冻结测试模型包不替代这三个训练输入。
3. 如果已有完整50张预测缓存，指定CACHE_DIR至含signature.json及000.npz等的目录；结果CSV压缩包不是缓存。
4. Run All。代码按历史固定划分训练两域独立头各500步，记录0及每100步评价。最终历史选择是DCID第100步、UHDM原融合；重跑结果有随机性，不能把新test当作挑选依据。
5. 模块化入口编辑configs/train_repair.json后运行train-repair。默认500步用于复现完整学习曲线，不是强制选用第500步。

## C. 源域融合训练与对照
推荐使用独立源域训练入口，读取历史融合包中固定的100/10清单，不需要旧Notebook变量：

```bash
python scripts/train_source_fusion.py --uhdm-root /kaggle/input/你的UHDM目录 --fusion-bundle /kaggle/input/你的目录/fusion_training_results.zip
```

该入口复用原融合训练逻辑，固定历史清单顺序，校验主干与数据分组。默认两种头各2000步。旧experiments/source_fusion脚本与训练单元Notebook仅为历史存档，其中训练单元依赖前置变量；推荐使用上面的独立训练入口。
共享空间/门控、ridge、联合微调的完整独立Notebook在experiments/controls，不是复现最终结果的必跑项。模块入口control-gate、control-calibration、control-joint可用configs/controls.json设置各模块已有大写变量。

## 数据与版本
DCID：https://huggingface.co/datasets/Mrduckk/DCID ，锁定51c674aa3f71c4cbbddb5cfcba9fbe3536eb0722；只用_m与_gt，不用_w。
ESDNet：https://github.com/CVMI-Lab/UHDM 。实际已执行版本及权重信息见结果协议。新增复现实验应记录自己的pip freeze及GPU信息，当前requirements为直接依赖列表而非伪造的历史完整环境锁。

## 验证状态
原始Notebook已由用户在Kaggle完成训练/测试。整理后的模块版在本地进行语法、模型权重加载、零初始化与数值等价检查；未重新在本机跑完整CUDA数据流程。遇到网络下载或挂载路径问题先按错误显式配置，不跳过缺图、重复或审计失败。
