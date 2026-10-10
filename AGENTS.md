# 项目工作约定

- main 使用 TSLib 核心分层：`run.py`、`exp/`、`data_provider/`、`models/`、`layers/`、`utils/`；当前 BasicTS 版保存在 `basicts-snapshot-20261010` 分支。
- 后续光伏实验在本项目目录运行，使用 `datasets/MMSP/data` 和 `pretrained/` 中的本地资源。main 的训练和评估不依赖 BasicTS；`vendor/BasicTS` 保留为此前复制的源码归档。
- 所有模型共用 `Dataset_MMSP` 和 `data_provider(args, flag)`；站点、时间窗口和 scaler 由两套协议统一决定，不建立模型专用 Dataset。
- 按协议将模型、预测及报告保存到本项目的 `checkpoints/`、`outputs/`、`reports/`，不再向相邻 BasicTS 或 FusionSF 项目写入新实验。
- 只执行用户当前明确授权的任务；模型清单不构成启动全部实验的授权。
