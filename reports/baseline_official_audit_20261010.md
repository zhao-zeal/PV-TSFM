# Baseline 官方来源与配置

本框架保留作者核心计算层，MMSP使用统一24→24数据接口。以下为选定作者脚本/API参考设置，不是MMSP最优超参，也不等于已完成正式复现。当前任务显式使用seed42；模型默认参考seed不同，正式启动必须记录覆盖值。

| 模型 | 官方参考设置与当前适配 |
| --- | --- |
| DLinear | 单功率通道、moving_avg25；ETTh1参考batch32、Adam lr0.005、10轮、patience3、type1 |
| PatchTST | 作者supervised模型、RevIN/BatchNorm；ETTh1参考batch128、dim16/FF128、3层/4头、dropout0.3、patch16/stride8、Adam lr1e-4、100轮、patience100、type3 |
| FusionSF | script三模态guide17，dim64/depth12/heads8、dropout0.4、ctx_mask0.99、TS/context VQ；batch16、AdamW lr0.0016/wd0.05/betas0.9/0.95、L1+VQ、cosine预热5、100轮、验证MAE patience20 |
| Cross-Unet | 作者use_cross_unet.py最终配置；batch128、dim256/FF512、3层/4头、seg12适配24小时、Adam lr1e-4、100轮、patience10、type3 |
| TimeXer | 官方MS分支，功率+历史NWP；batch32、dim512/FF512、2层/8头、Adam lr1e-4、10轮、patience3、type1；patch12适配24步 |
| Chronos-2 | 官方Chronos2Pipeline，冻结、本地权重、median0.5、cross_learning=False、无外部裁剪；四种NWP输入模式分开记录 |
| ChronosX | 官方chronosx分支，Chronos-T5-small+IIB/OIB，只训练injection_block；token CE、AdamW lr0.01/wd0、linear、5000更新步、batch32/累积2、每100步按验证CE保存、20采样 |
| Time-VLM | 官方CLIP full-shot，历史功率生成图像/文本、检索记忆、门控融合；batch32、dim128、Adam lr0.001、10轮、patience5、GPU fp16/CPU fp32 |

来源与版本见 `models/baseline_sources.json`、`layers/`中的LICENSE/NOTICE；原作者配置参考保留在 `scripts/official_reference/`，生效默认值由 `models/official_presets.py` 与 `run.py` 决定。PatchTST/Cross-Unet参考训练中的OneCycle构造器首轮base/25与beta1=0.95行为沿用；后续按选定作者epoch调度设置执行。

DLinear/PatchTST旧三通道适配版本只在legacy_fusionsf中选择；FusionSF guide15、guide17及2/3模态分别记录。当前FusionSF双模态是功率+卫星，不等于历史功率+NWP对照。Time-LLM已移除，不再保留其权重或运行验收记录。

验证/测试保持顺序并覆盖全部窗口，训练drop_last按模型设置。模型输入、训练配置、验证选模依据和最终测试指标须一并保存。Time-VLM记忆库随checkpoint保存，保留作者按forward更新记忆的行为。MMSP适配和数值行为以现有代码为准；一次性验收日志已清理，不能以它们替代后续完整训练/正式评估证据。

FusionSF和Cross-Unet两个协议均已完成正式测试并通过验收；当前获批启动TimeXer两个协议seed42，其余baseline尚未取得当前官方版本的完整两协议实验成绩。数据和指标口径见 `docs/DATA_PROTOCOLS.md`，当前任务与运行位置见 `docs/EXPERIMENT_STATUS.md`。
