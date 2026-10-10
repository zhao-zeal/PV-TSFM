# 实验总结

任务：将 PV-TSFM main 改为 TSLib 风格，保存此前 BasicTS 版，迁入 in_domain 与 zeroshot_v1 两套协议，统一所有模型的数据和运行接口。未启动新训练，优化器更新次数为 0。

## 分支与结构

旧版已提交并推送至 [basicts-snapshot-20261010](https://github.com/zhao-zeal/PV-TSFM/tree/basicts-snapshot-20261010)，提交 72ba44c。main 保留核心分层：`run.py`、`data_provider/`、`exp/`、`models/`、`layers/`、`utils/`、`scripts/`。

用户提到 PVRA-PV，公开检索未定位该名称；按名称相近的光伏项目 [weican1103/PARA-PV](https://github.com/weican1103/PARA-PV) 作为候选参考，已克隆到 `/home/zhaopp/workspace/references/PARA-PV`，提交 158ef9aed8aa7654341b8b6b6379daacc843d4d5。另核对 TSLib 官方源码，克隆到 `/home/zhaopp/workspace/references/Time-Series-Library`，提交 4e938a1767106324dd753b2a44832bf870a0252e。只参考分层与接口，没有引入 PARA-PV 模型和数据。

所有模型共用 `data_provider(args, flag)` 和 `Dataset_MMSP`，固定返回：

```python
seq_x, seq_y, seq_x_mark, seq_y_mark, covariates
```

前四项沿用 TSLib 序列接口；第五项提供固定键名的历史/未来 NWP、卫星图像/坐标、站点坐标、FusionSF 时间编码、Cross-Unet 历史相关块，以及站点/预测时间元信息。未加载模态为可 collate 的空张量，Dataset 不按模型分叉。模型统一为 `Model(configs)`，接受 `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=...)`。实验层只将历史 label 段和零填充未来功率送入 decoder，未来真实功率仅用于损失/指标。

main 的运行不依赖 BasicTS/EasyTorch，旧 `src/pvtsfm` 已迁出 main。此前复制的 `vendor/BasicTS` 继续保留为源码归档；DLinear 使用的移动平均计算层及许可证已复制到 `layers/`。本项目内 datasets/pretrained/checkpoints/outputs/reports 继续保留，默认路径均在项目内。

## 配置与结果

MMSP，24→24，label_len=12。in_domain 训练/验证/测试站点为 0–9；zeroshot_v1 训练/验证站点为 10–19，测试为 0–9，测试复用训练 scaler。时间比例 60%/20%/20%，仅按完整目标时间分配窗口，保留原有归一化。每协议训练窗口 76,570、验证窗口 25,450、测试窗口 25,450。DLinear/PatchTST 完整复评 seed=42，使用已有验证最优权重，GPU 0、CPU 0–3，batch=64。

| 模型 | 旧 MAE | 新入口 MAE | 旧 RMSE | 新入口 RMSE | MAE 绝对/相对变化 | RMSE 绝对/相对变化 |
|---|---:|---:|---:|---:|---|---|
| DLinear | 0.0775414185108 | 0.0775414185108 | 0.1239254257096 | 0.1239254257096 | 0 / 0% | 0 / 0% |
| PatchTST | 0.0638854546691 | 0.0638854546128 | 0.1155919995611 | 0.1155919996993 | -5.6342e-11 / -8.8192e-8% | +1.3818e-10 / +1.1954e-7% |

DLinear 完整预测逐元素一致；PatchTST 最大单点差异 1.4305e-6，通过 atol=1e-6、rtol=1e-5 的联合容差，指标变化仅为浮点误差。两模型目标、站点 ID、预测时间戳完全一致。PatchTST 最佳：in_domain、seed=42、24→24、验证最优 epoch=34，原训练 batch=16、初始学习率 0.0002，patch=16、stride=8、d_model=512、3 层、8 头；新入口 MAE=0.0638854546、RMSE=0.1155919997、R²=0.7808967460。相对 DLinear，MAE 减少约 0.0136559639（17.61%），RMSE 减少约 0.0083334260（6.72%）。

## 核对与修复

- 两协议 × 四种模态 × train/val/test 共 24 组接口核对，覆盖首/末窗口与站点边界：窗口数量、历史功率、未来标签、坐标模型实际 float32 输入、已加载模态和 scaler 均一致。固定 covariates 键名和五项 batch 默认 collate 通过。
- Cross-Unet 相关历史块、历史/未来天气，TimeXer 日历特征映射对照通过。
- DLinear、PatchTST、Cross-Unet、TimeXer、FusionSF 双模态 experiment、三模态 experiment、三模态 script 共 7 组模型配置：相同权重下前向与梯度差异均为 0，梯度有限，改变未来真实标签不改变预测；优化器与调度器构建通过。
- Chronos-2 两协议 × 四种输入各核对窗口 [0,2544,2545,25449]，最大差异 1.7881e-7；TimesFM 3.0 同样核对四个窗口，最大差异 3.2783e-7。均满足联合容差，预训练模型保持冻结；冻结复核仅使用最多 4 个 CPU 核，DataLoader 0 子进程。
- 对齐旧实验的 TF32 关闭、cuDNN benchmark 设置，修正第一次 PatchTST 复评的数值设置差异；最终完整指标如上。数据验收脚本按模型实际使用的 float32 坐标比较，避免误将原始 float64 坐标精度变化当成输入错误。
- 清除旧 build 目录中的过期包文件后重新构建 0.2.0 wheel；隔离导入 10 个入口/模型模块，检查 wheel 包含当前源码全部模块、许可证，且没有旧 pvtsfm/BasicTS 模块。CLI 帮助、配置查看、可编辑安装与 shell 语法检查通过。

## 保存位置与限制

根目录：`/home/zhaopp/workspace/PV-TSFM`。

- 运行说明与接口字段：`README.md`。
- 完整验收预测、指标、配置、scaler、站点与时间元信息、日志：`outputs/in_domain/tslib_acceptance_DLinear_20261010/`、`outputs/in_domain/tslib_acceptance_PatchTST_20261010/`。
- 复评用的原模型权重：`checkpoints/in_domain/legacy_20261010/DLinear/`、`checkpoints/in_domain/legacy_20261010/PatchTST/`。未修改原权重。
- 验收证据：`reports/tslib_migration_20261010.json`、`reports/tslib_frozen_migration_20261010.json` 及对应 `.log`；包验收为 `reports/tslib_package_acceptance_20261010.json`，构建日志为 `reports/tslib_package_build_20261010.log`。
- 分协议结果表：`reports/in_domain/results.csv`、`reports/zeroshot_v1/results.csv`。已有结果索引：`reports/legacy_experiment_index.csv`。本次是重构一致性验收，没有生成新的性能图；历史图表仍保存在 `reports/legacy/`。
- 包：`reports/pv_tsfm-0.2.0-py3-none-any.whl`。尚未验证另一台全新机器上的完整环境安装。

TimeXer 已接入通用训练/测试流程，默认新增训练设置为 Adam/MSE/lr=0.0001，尚无正式训练验证；此次只验证输入和计算一致性。ChronosX 来源已确认但未迁入，Ours 仍为待设计方法。FusionSF、Cross-Unet 等此前缺失的正式结果没有因目录迁移自动补齐。

旧数据的未来 NWP 发布时间缺失及插值限制继续存在；TimesFM 已有异常预测质量结论继续保留。没有重跑冻结模型全部测试窗口，没有启动新训练，也没有改动原项目的后台训练。

Git 分支保存代码及验收清单；大数据、预训练权重、实验权重和完整预测仍通过本机项目目录保存。完整转发必须保留被 Git 忽略的这些目录。

结论：TSLib 核心分层、统一数据接口与两套协议已迁入 main，已有模型计算与结果在验收范围内保持一致。后续最小行动是使用新入口检查所需模型配置，再按用户明确指定的任务运行实验。
