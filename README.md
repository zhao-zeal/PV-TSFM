# PV-TSFM

基于 BasicTS 的独立光伏预测实验项目。BasicTS 提供训练、评估、配置和指标；本项目维护 MMSP 数据协议、模型适配器及实验配置。所需 BasicTS 源码、MMSP 数据和已用预训练权重均保存在项目目录内，后续实验统一在本项目运行。

## 安装

项目内 `vendor/BasicTS` 保存 BasicTS 1.1.0 的原始框架源码，来源提交记录在 `vendor/BasicTS/UPSTREAM_REVISION`。使用 Python 3.11 或更高版本安装：

```bash
cd PV-TSFM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e vendor/BasicTS -e .
```

BasicTS 的旧版依赖声明包含 setuptools 和 transformers 的固定版本；先安装本项目运行依赖，再使用 `--no-deps` 安装框架，避免覆盖 Chronos 所需版本。已有兼容 PyTorch 环境时也可使用 `--system-site-packages` 创建虚拟环境。Chronos-2 和 TimesFM 3.0 分别对应 `chronos-forecasting==2.2.2` 和 `timesfm==3.0.2`（提供 `timesfm3` 模块）。

## 结构与入口

```text
src/pvtsfm/data/mmsp/    数据加载、站点划分和模型输入接口
src/pvtsfm/models/       FusionSF、DLinear、PatchTST、Cross-Unet、TimeXer、冻结模型适配器
src/pvtsfm/experiments/  各模型的独立配置与运行入口
src/pvtsfm/runner.py     BasicTS 扩展：验证后更新 Plateau 调度器、标准 .npy 导出
run.py                  选择一个明确指定的任务
vendor/BasicTS/          项目内框架源码及上游许可证
datasets/MMSP/data/      实际复制的 MMSP 数据，不使用目录链接
pretrained/             Chronos-2、TimesFM 3.0 本地预训练权重
checkpoints/            模型权重及历史已完成模型的归档
outputs/                本项目评估预测及历史结果归档
reports/                验收报告、分协议结果表、历史实验索引
```

仅查看配置或帮助：

```bash
python run.py --help
python run.py crossunet --protocol in_domain --print-config
python run.py timexer --protocol in_domain
python run.py baseline --model DLinear --protocol in_domain --gpus 0 --print-config
python run.py fusionsf --protocol in_domain --gpus 0 --print-config
python run.py zeroshot --help
```

TimeXer 目前只准备接口，入口只打印配置，不启动训练。Cross-Unet 保留官方实现和脚本实际生效的模型、优化器、学习率调度配置；MMSP 的 24→24 数据设置由本项目单独指定。FusionSF 保留 `experiment` 与官方 `script` 两种预设。

数据默认读取本项目 `datasets/MMSP/data`；可用 `PVTSFM_MMSP_DIR` 或入口的 `--data-dir` 覆盖。冻结模型默认读取本项目 `pretrained/chronos-2` 和 `pretrained/timesfm-3.0-pytorch`，也可使用 `--model-path`。预测长度为 24；`in_domain` 和 `zeroshot_v1` 协议的结果保存在各自目录。

训练需要明确选择空闲 GPU，单次最多 4 张；入口会检查占用。CPU 零样本评估限制为 4 核、DataLoader 0 个子进程。配置查看不启动实验。数据、预训练权重、`checkpoints/`、`outputs/`、`reports/` 不纳入 Git。转发和本地保存时复制整个项目并保留这些目录，排除 `.venv/`、`.git/` 和 `__pycache__/`，接收方重新安装环境。

## 当前模型范围与验收

DLinear、PatchTST、FusionSF、Cross-Unet、Chronos-2 具有运行入口；TimeXer 当前仅提供配置与输入接口。TimesFM 3.0 保留为已有额外模型。ChronosX 已确认指 [ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables](https://proceedings.mlr.press/v258/arango25a.html)，官方代码位于 [chronos-forecasting 的 chronosx 分支](https://github.com/amazon-science/chronos-forecasting/tree/chronosx)，尚未接入本项目。Ours 为用户尚未确定的自有方法，属于待设计内容，不计为缺失基线。当前七个指定基线中，仍需接入 ChronosX、补齐 TimeXer 训练入口。

本机验收报告位于 `reports/acceptance_20261010.md`；分协议结果位于 `reports/in_domain/results.csv` 和 `reports/zeroshot_v1/results.csv`。历史结果索引位于 `reports/legacy_experiment_index.csv`。工程迁移验收不代表全部模型的正式实验已完成。

## GitHub

`origin`：https://github.com/zhao-zeal/PV-TSFM.git，分支 `main` 跟踪 `origin/main`。沿用远端历史，旧 Hydra/Lightning 工程在本地重构为上述结构；远端更新需另行提交和推送。
