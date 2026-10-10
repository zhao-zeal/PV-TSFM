# 实验协议、进度与待办

更新时间：2026-10-10（Asia/Shanghai）。本文件是后续 Codex 恢复工作的入口；运行状态以实际进程、日志与产物为准。长期模型清单不构成执行授权。

## 最新决定与授权

- 只做 `in_domain` 和 `zeroshot_v1`；取消后续 Site1 实验，保留其代码和历史异常记录。
- 仅 seed 42，不做多种子。此前三个种子的方案已被本次指令覆盖。
- 逐模型完成两个协议并验收；优先顺序 FusionSF → Cross-Unet → TimeXer。
- 用户本次批准：文档更新及 FusionSF 两协议 seed42 的训练与测试。最多两张空闲 GPU，合计最多四个 CPU 核，DataLoader=0。本次不启动后续模型、不自动调参、不修改模型结构、不发邮件。
- 每个模型完成后向用户报告、等待验收和下一模型审批。发现异常先核对并讨论，不将异常输出当作可信 baseline。
- 活跃 baseline：DLinear、PatchTST、FusionSF、Cross-Unet、Chronos-2、ChronosX、TimeXer、Time-VLM。Time-LLM 已移除；Ours 尚未设计。

## 固定数据协议

| 协议 | 训练站点 | 验证站点 | 测试站点 | 训练/验证/测试窗口数 |
| --- | --- | --- | --- | --- |
| in_domain | #0–#9 | #0–#9 | #0–#9 | 76570 / 25450 / 25450 |
| zeroshot_v1 | #10–#19 | #10–#19 | #0–#9 | 76570 / 25450 / 25450 |

- MMSP 小时数据，历史24步→未来24步，label_len=12，单功率目标。
- 站内时间60/20/20；完整目标窗口归属对应时间分区。验证/测试的历史上下文可来自分区之前的已知历史。
- 目标范围（左闭右开）：训练 [2021-01-02, 2021-11-19)，验证 [2021-11-19, 2022-03-06)，测试 [2022-03-06, 2022-06-21)。
- 所有模型共用 `Dataset_MMSP` 和五项 TSLib 接口，不建立专用 Dataset，不改变预处理或指标口径。
- `chronological_target_split_v1`；scaler 仅由训练站点及训练时段拟合，zeroshot 测试复用训练 scaler。测试集不参与配置或 checkpoint 选择。
- Dataset 提供15个 NWP 变量；FusionSF 官方 guide17 在适配层添加两列经纬度。保留历史 guide15 兼容，不混入新配置结果。
- `paper_main_v1` 是全量 scaler 的历史论文兼容协议，不是本项目正式主协议 `in_domain`，不得互换命名。
- 保留已记录的限制：NWP CSV 缺少发布时间，未来气象按协议假定可用；既有双向插值的在线可用性未验证，不据此声称严格在线零泄漏。
- 测试指标按全部预测元素统一计算 MAE/RMSE/R²（全局 R²）；不额外标准化功率。每次保存协议和 scaler。

## 已完成与当前进度

| 项目 | 状态与证据 |
| --- | --- |
| TSLib 核心迁移、统一数据接口、官方模型接入 | 已有实现与基础验收记录；全框架冻结尚未最终验收 |
| 两个完整数据协议 | 已核对站点、窗口和 scaler；见 `reports/stage0_20261010/protocols/` |
| Site1 协议 | 已实现和验收；当前不再运行 |
| 旧 FusionSF guide15 的框架回归 | 新旧全部预测完全一致；测试 MAE=0.04780034881873366，RMSE=0.10953583251576295，R²=0.8032540956219328；绝对/相对变化均为0 |
| FusionSF Site1 guide17 两次训练 | lr=0.0016 和 0.00016 均出现全零预测，已停止；无可信正式测试结果 |
| FusionSF 两协议 guide17、seed42 | 两个协议已启动并行训练；仅 seed42；尚无测试指标 |
| Cross-Unet / TimeXer 两协议实验 | 未启动，等待 FusionSF 验收和下一模型批准 |
| 其余 baseline / Ours | 不在本次执行范围 |

旧 guide15 结果只作历史参照，不替代当前 guide17 的正式运行。Site1 降学习率后验证 MAE/RMSE 变化均为0，诊断样本预测头梯度为0；根因尚未确定，不能假定增加站点会解决。旧完成实验使用的 VQ 包版本与当前环境不同，训练行为等价性未验证；本次不据此自动更换依赖。

## 当前 FusionSF 配置与运行表

官方 script 三模态，guide17；dim64/depth12/heads8，decoder_dim128/depth4/heads6，dropout0.4，ctx_mask0.99；TS/CTX VQ 开启，guide VQ 关闭，保留官方 ReLU。

seed42、batch/eval_batch16、FP32、AdamW（lr0.0016，betas0.9/0.95，weight_decay0.05）、L1+VQ、cosine+5轮warmup、最多100轮、patience20，以验证MAE选最优。现有调度器第一轮实际lr=0，不能将该轮训练损失下降等同于参数优化收敛。

运行名：`FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full`。

| 协议 | GPU | CPU 绑定 | PID | 状态 |
| --- | --- | --- | --- | --- |
| in_domain | 3 | 0,7 | 2635253 | 运行中（exec session 74427） |
| zeroshot_v1 | 0 | 9,10 | 2642044 | 运行中（exec session 93008） |

首次 nohup 启动未保留进程，未执行训练；随后 GPU0/1 被其他用户占用，已改为保留的训练会话在 GPU3 启动 in_domain；GPU0 再次空闲后已启动 zeroshot_v1。两条实际 Python 进程均已核对，分别约占12GB显存。不得使用仍有其他进程的卡。

运行记录：`reports/FusionSF_two_protocols_seed42_20261010/runs.json`。恢复工作先核对 PID 命令行、GPU 和日志，不盲目重启；只启动本次两条运行，没有多种子或后续模型队列。

## 待办与验收

1. 启动并观察当前两协议 seed42，检查训练/验证输出是否正常；再现全零预测则暂停受影响运行、核对输出/预测头梯度并讨论，不自动调参。
2. 正常完成时加载验证 MAE 最优 checkpoint，固定测试，检查预测、指标、站点/时间索引及配置；分别汇总 MAE/RMSE/R²。
3. 保存日志、模型、结果表与必要图表，报告历史参照及配置差异；缺失或异常必须明确标记。向用户提交 FusionSF 验收后停止。
4. 用户批准后才进入 Cross-Unet；之后同样审批 TimeXer。不进行其他模型或多种子实验。

## 产物位置与保存说明

- 已有协议及回归验收：`reports/stage0_20261010/report.md`、`protocols/{in_domain,zeroshot_v1}/data_protocol.json`。
- Site1 异常：`reports/site1_FusionSF_20261010/` 和 `reports/site1_FusionSF_lr0p00016_20261010/`。
- 历史完成结果：`outputs/in_domain/legacy_completed_20261010/FusionSF/ca862c2f74fb5f4eb788a861875a39e1/`。
- 本次报告和启动日志：`reports/FusionSF_two_protocols_seed42_20261010/`。
- 本次最佳权重：`checkpoints/{protocol}/FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full/checkpoint.pth`。
- 本次配置、日志、协议/scaler、预测与指标：`outputs/{protocol}/FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full/`；测试结束才生成预测与 metrics.json。
- 结果表：`reports/{protocol}/results.csv`；汇总与图表保存在本次报告目录。
- 所有新实验只写 PV-TSFM。本地大文件/多数 reports 被 Git 忽略，完整转发需要复制项目数据、权重与产物；用户已于2026-10-10批准本轮代码及文档推送至GitHub main；当前版本以本地和远端提交核对为准。

GitHub：推送前 main 远端与本地 HEAD 均为 `01e9ff4`；用户已批准本轮提交与推送。数据、预训练权重、运行checkpoint和大部分reports不随Git同步。

完成时间仅为估计：沿用已有速度、并行运行且约30–40轮早停时约8–12小时；跑满100轮可能超过一天。两个完整协议尚未完成首轮，须按实测耗时校准。

## BasicTS 旧运行核对

只读观察时间：2026-10-10T14:14:55+08:00。旧协调进程PID1785089，脚本`/tmp/mmsp_two_protocol_experiments.py`；六个旧任务中五个已完成，仅FusionSF zeroshot_v1尚在运行。旧协调脚本没有后续新模型或多种子任务。

| 协议 | 模型 | 状态 | 最佳验证轮次 | 测试 MAE | 测试 RMSE | 测试 R² |
| --- | --- | --- | --- | --- | --- | --- |
| in_domain | DLinear（旧配置） | 已完成 | 9 | 0.07754142 | 0.12392543 | 0.74816615 |
| in_domain | PatchTST（旧配置） | 已完成 | 34 | 0.06388545 | 0.11559200 | 0.78089675 |
| in_domain | FusionSF（旧配置） | 已完成 | 16 | 0.04780035 | 0.10953583 | 0.80325410 |
| zeroshot_v1 | DLinear（旧配置） | 已完成 | 9 | 0.07811652 | 0.12445281 | 0.74601814 |
| zeroshot_v1 | PatchTST（旧配置） | 已完成 | 18 | 0.06457397 | 0.11509320 | 0.78278361 |
| zeroshot_v1 | FusionSF（旧guide15） | 完成7轮，正在8/100 | 7（按日志） | 未测试 | 未测试 | 未测试 |

仍运行的旧FusionSF：PID2195967，GPU2，seed42、24→24、guide15、lr0.0016、batch16、最多100轮、patience20。第2–6轮验证MAE约0.1121，第7轮改善至约0.0396、RMSE约0.0936（日志四位小数）。早期停滞不证明永久失效，亦未核对其逐样本输出；不得将这项观察解释为本次guide17已经正常。

与PV-TSFM跨站运行在数据协议、种子和任务上重叠，guide通道与环境存在差异。用户最新决定：暂时保留两边，先讨论，继续监控；不得自行终止、重启或调整任一运行。本次仅核对旧任务，没有新启动或调整它。当前合计三个主训练进程：BasicTS旧guide15跨站一条，加PV-TSFM新guide17两个协议；这些是不同配置的独立结果，不合并种子统计。旧任务DataLoader4且未绑定四核，不符合当前执行偏好；当前PV-TSFM两个进程仍各绑定两核、DataLoader0。

旧跨站最低测试MAE（已完成者）为PatchTST，最佳验证epoch18、lr0.0002、batch16；相较旧DLinear：
- mae: -0.0135425464，相对变化 -17.34%（仅旧配置同协议比较）。
- rmse: -0.0093596143，相对变化 -7.52%（仅旧配置同协议比较）。
- r2: +0.0367654737，相对变化 +4.93%（仅旧配置同协议比较）。

旧状态/最佳checkpoint/产物索引：`/home/zhaopp/workspace/BasicTS/reports/two_protocol_reproduction_20261010.json`；旧Fusion日志：`/home/zhaopp/workspace/BasicTS/checkpoints/zeroshot_v1/reproduction_20261010/FusionSF/f0ba7f2fda78657013614576eff71728/training_log_20261010122154.log`。旧数据/权重仍在BasicTS，本次未复制全部旧产物；guide15历史in_domain产物已按此前验收复制到PV-TSFM。
