# 项目工作约定

- main 使用 TSLib 核心分层：`run.py`、`exp/`、`data_provider/`、`models/`、`layers/`、`utils/`；当前 BasicTS 版保存在 `basicts-snapshot-20261010` 分支。
- 后续光伏实验在本项目目录运行，使用 `datasets/MMSP/data` 和 `pretrained/` 中的本地资源。main 的训练和评估不依赖 BasicTS；`vendor/BasicTS` 保留为此前复制的源码归档。
- 所有模型共用 `Dataset_MMSP` 和 `data_provider(args, flag)`；站点、时间窗口和 scaler 由 `site1_v1`、`in_domain`、`zeroshot_v1` 协议统一决定，不建立模型专用 Dataset。历史 `paper_main_v1` 保留原含义，不作为无全量 scaler 泄漏的正式协议。
- 按协议将模型、预测及报告保存到本项目的 `checkpoints/`、`outputs/`、`reports/`，不再向相邻 BasicTS 或 FusionSF 项目写入新实验。
- 只执行用户当前明确授权的任务；模型清单不构成启动全部实验的授权。
- 每个阶段修改或运行前须经用户审批；本阶段完成后停止，验收并等待下一阶段审批。已验收内容跳过；发现冲突或含糊之处先讨论。当前只做 `in_domain` 和 `zeroshot_v1`，取消后续 Site1 实验；只运行 seed 42，不做多种子。按 FusionSF → Cross-Unet → TimeXer 逐模型推进，每个模型完成两个协议并验收后，再申请下一模型审批；Ours 单独审批。
- 当前 baseline 共8个：DLinear、PatchTST、FusionSF、Time-VLM、Cross-Unet、ChronosX、TimeXer、Chronos-2。用户已移除 Time-LLM 并要求删除 Llama 本地权重；历史验收报告保留原始记录。
- 长期工作先读 `docs/EXPERIMENT_STATUS.md`，其中记录协议、最新授权、进度、异常、待办和产物路径；恢复任务时先核对已有进程与 checkpoint，避免重复启动。用户于 2026-10-10 明确批准 FusionSF 两协议 seed42 实验和文档更新；没有批准启动后续模型、多种子、调参或架构修改。
- 本次最多使用两张空闲 GPU；两个并行进程合计最多 4 个 CPU 核，每个进程绑定两个核心并限制计算线程为 2，DataLoader 使用 0 个子进程。实验总结在项目与会话中保存，不自动发邮件。
