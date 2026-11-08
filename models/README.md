# 权重与数据分发

代码仓库不包含数据、官方主干权重或训练权重二进制。
复现最终结果需要 `moire_frozen_test_models.zip`（随交付文件单独提供），包含旧融合头、DCID 第100步空间头及训练协议。
SHA256 见 frozen_protocol.json；程序加载时校验。旧融合训练衔接包 `fusion_training_results.zip` 用于重建训练缓存。
官方 ESDNet 权重由程序按官方链接获取，或用 WEIGHT 指定。不得用其他主干权重替换后声称复现相同结果。
公开分发前确认相关授权；本仓库不替原作者授予模型或数据使用许可。
