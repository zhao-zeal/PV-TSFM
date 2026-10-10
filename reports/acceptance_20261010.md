# 实验总结

日期：2026-10-10。任务：验收现有实验和 PV-TSFM 工程迁移，核对指定模型清单，将后续所需框架、数据、权重和已有结果实际复制到 PV-TSFM。结论：现有工程迁移通过；指定模型清单和正式实验结果尚未齐全。

## 模型覆盖与结果状态

| 模型 | 项目内代码与入口 | 正式结果状态 |
|---|---|---|
| DLinear | 模型、训练及已有 checkpoint 评估入口齐全 | in_domain 完整测试已复核；zeroshot_v1 未完成 |
| PatchTST | 模型、训练及已有 checkpoint 评估入口齐全 | in_domain 完整测试已复核；zeroshot_v1 未完成 |
| FusionSF | 双模态、三模态及 experiment/script 预设；训练入口齐全 | 已归档三模态验证最优权重快照，无最终测试结果 |
| Cross-Unet | 官方模型、适配器、训练入口齐全 | 数据、前向/反向、训练初始化与保存恢复通过；无正式结果 |
| Chronos-2 | 冻结推理入口及本地权重齐全 | 两协议各四种输入的已有完整结果已归档；迁移后抽样推理通过 |
| ChronosX | 已确认论文和官方 chronosx 分支；尚未接入 | 无正式结果 |
| TimeXer | 模型及输入、配置接口已有；入口只打印配置 | 无训练入口及正式结果，不能算完整基线 |
| Ours | 自有方法，尚未设计确定；不计入基线 | 待用户后续确定方法 |

用户补充确认：ChronosX 指 [ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables](https://proceedings.mlr.press/v258/arango25a.html)（AISTATS 2025），官方代码位于 [amazon-science/chronos-forecasting 的 chronosx 分支](https://github.com/amazon-science/chronos-forecasting/tree/chronosx)。该分支提供历史/未来外生变量适配，README 示例基于 `amazon/chronos-t5-small`，不能将当前 Chronos-2 加 NWP 的结果视为 ChronosX 结果。Ours 为待设计自有方法，不是待补齐的已有基线。

额外保留已有 TimesFM 3.0 官方接口、权重和结果。上述清单不构成启动新模型或新实验的授权。

## 配置与核心指标

MMSP，历史 24 小时预测未来 24 小时。完整复评：in_domain，站点 0–9，按时间 60%/20%/20% 划分，seed=42，训练窗口 76,570、验证窗口 25,450、测试窗口 25,450，每份预测形状 (25450, 24, 1)。归一化仅拟合训练站点和训练时段。当前结果为加载迁移前最优 checkpoint 的评估，优化器更新次数为 0。

| 模型 | 迁移前 MAE | 项目内复评 MAE | 迁移前/复评 RMSE | 迁移前/复评 R² | MAE、RMSE 绝对/相对变化 |
|---|---:|---:|---:|---:|---|
| DLinear | 0.0775414185 | 0.0775414185 | 0.1239254257 | 0.7481661456 | 均为 0 / 0% |
| PatchTST | 0.0638854547 | 0.0638854547 | 0.1155919996 | 0.7808967465 | 均为 0 / 0% |

两份完整预测逐元素完全一致。当前完整复评的两个监督模型中，PatchTST 最佳：seed=42，24→24，batch=16，初始学习率 0.0002，patch=16、stride=8、d_model=512、3 层、8 头，AdamW/MSE，验证 MAE 最优 epoch=34。相对 DLinear，MAE 降低 0.0136559638（17.61%），RMSE 降低 0.0083334261（6.72%），R² 增加 0.0327306009。DLinear 配置为 batch=64、初始学习率 0.0016、moving_avg=13，最优 epoch=9。

已有归档结果中，in_domain 最低 MAE 为 Chronos-2 的 power_history_future_nwp：MAE=0.0609095408、RMSE=0.1177711910、R²=0.7725576115，seed=42、24→24。相对同模型纯功率输入，MAE 降低 0.0052162228（7.89%），RMSE 降低 0.0192365306（14.04%）。其输入包含 NWP，与纯功率模型的横向比较存在输入条件差异。最低 RMSE 为 PatchTST。Chronos-2 本次只进行迁移后抽样推理，没有重新运行全部测试窗口。

## 验收依据与修复

- 56 个 Python 源文件语法通过；数据协议、各模态输入及 scaler 与原工程抽样一致，覆盖站点边界。DLinear、PatchTST、FusionSF、Cross-Unet、TimeXer 的模型前向与梯度对照最终差异为 0；目标值变化不影响预测。
- 最终实际导入框架为项目内 `vendor/BasicTS/src/basicts`，使用未修改的上游 BasicTS；项目的 PVRunner 承担 Plateau 验证后调度和标准预测导出。
- DLinear、PatchTST 的真实命令行完整评估通过。DLinear、PatchTST、FusionSF、Cross-Unet 的训练初始化、优化器/调度器状态保存恢复通过，没有执行训练更新。
- Chronos-2 两协议四种输入分别核对首窗口、站点边界及末窗口；最大预测差异 1.7881e-7。TimesFM 3.0 同样核对四个窗口，最大差异 3.2783e-7，均在 atol=1e-6、rtol=1e-5 内，预训练参数保持冻结。
- 修复独立评估未保存 cfg.json 的问题；DLinear、PatchTST/FusionSF 基线入口增加 `--checkpoint`，支持明确只评估已有权重。
- 最终 wheel 构建通过，隔离导入 9 个入口模块，包含全部 55 个包内 Python 模块及相应许可证/声明。尚未验证另一台干净机器上的完整安装。
- GPU 检查后仅使用空闲 GPU 0；计算绑定 CPU 0–3。冻结模型 CPU 推理最多 4 线程，DataLoader 子进程为 0。未启动新的训练任务，也未改动原工程正在运行的实验。

## 实际复制与后续保存

项目根目录：`/home/zhaopp/workspace/PV-TSFM`。以下均为实际文件，数据和预训练权重不是符号链接，并与来源逐字节核对一致。

| 内容 | 项目内路径 | 说明 |
|---|---|---|
| 所需 BasicTS 框架源码及许可证 | `vendor/BasicTS/` | 上游提交 c2bb6e31e591167e84459775a21a62e70a5893ce，来源记入 UPSTREAM_REVISION |
| MMSP 功率、NWP、卫星数据 | `datasets/MMSP/data/` | 8 文件，1,469,517,843 字节 |
| 已用预训练模型 | `pretrained/chronos-2/`、`pretrained/timesfm-3.0-pytorch/` | 合计约 1.7 GiB |
| 监督模型权重、配置、日志、预测 | `checkpoints/in_domain/legacy_20261010/` | DLinear/PatchTST 最优 checkpoint；FusionSF 验证快照另有 archive_note.json |
| 10 份冻结模型历史结果 | `outputs/{protocol}/legacy_20261010/` | 含两协议 Chronos-2 共 8 份和 TimesFM 共 2 份 |
| 迁移后完整评估结果 | `outputs/in_domain/local_acceptance_20261010/` | DLinear、PatchTST 的 cfg.json、test_metrics.json、预测数组等 |
| 结果表 | `reports/in_domain/results.csv`、`reports/zeroshot_v1/results.csv` | 已有结果和迁移复评；复评不是独立训练 |
| 已有实验索引 | `reports/legacy_experiment_index.csv` | 12 份完成结果、4 份未完成记录；result_dir 指向本项目，source_result_dir 保留历史来源 |
| 原有总结及图表 | `reports/legacy/` | 包含 Chronos-2、TimesFM 的 JSON/CSV/Markdown/PNG 图表和日志 |
| 验收证据 | `reports/acceptance_20261010.json`、`local_runtime_acceptance_20261010.json`、`frozen_acceptance_20261010.json`、`local_copy_20261010.json`、`package_acceptance_20261010.json` | 详细核对记录；相应 .log 保存在同一 reports 目录 |
| 包构建产物 | `reports/pv_tsfm-0.1.0-py3-none-any.whl` | 配合项目内框架与数据使用 |

共归档 12 份已完成实验结果。历史配置中的原绝对路径保留作来源记录，当前默认加载与新实验保存路径均在本项目内。项目 AGENTS.md 已规定后续光伏实验在 PV-TSFM 内执行，使用内部框架、数据与权重。

完整转发请复制项目目录，包括被 Git 忽略的 datasets、pretrained、checkpoints、outputs、reports；排除 .venv、.git、__pycache__，按 README 重新安装环境。单独转发代码仓库或 wheel 不包含数据、预训练权重和结果。

## 缺失与异常

指定的 7 个基线中，5 个具备运行入口，TimeXer 尚未具备训练入口，ChronosX 来源已确认但尚未接入。Ours 为另外的待设计方法，不计为缺失基线。FusionSF 的快照仅代表已有验证状态，不能充当最终测试结果；Cross-Unet 和 TimeXer 没有正式结果，监督模型 zeroshot_v1 三份结果仍缺失。

已有未来 NWP 数据缺少发布时间，双向插值涉及 29 个缺失单元，严格在线可用性尚未证明。TimesFM 3.0 已有结果 R² 约 -0.524，存在较大异常预测；迁移一致性通过不意味着预测质量通过。验收脚本早期随机遮蔽及 VQ 初始化对照失败已通过统一随机状态、同步初始化后的权重修正，最终模型对照通过。

Git origin 已连接 `https://github.com/zhao-zeal/PV-TSFM.git`，main 跟踪 origin/main；只读远端核对确认连接，当前重构及本次迁移相关代码仍未提交、未推送。Git 忽略的数据与结果需要完整目录转发保存。

结论：PV-TSFM 已完成现有范围的本地迁移和运行验收，七个指定基线仍待接入 ChronosX、补齐 TimeXer 训练入口；Ours 保留为待设计方法。后续由用户指定要补齐的模型任务。
