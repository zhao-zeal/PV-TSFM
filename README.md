# PV-TSFM

TSLib 风格的光伏预测工程。main 保留 `run.py → exp → data_provider/models → layers/utils` 核心分层；所有模型使用同一 MMSP 数据接口。此前通过验收的 BasicTS 版已保存在 GitHub 分支 [`basicts-snapshot-20261010`](https://github.com/zhao-zeal/PV-TSFM/tree/basicts-snapshot-20261010)。

## 安装与文件

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

main 的运行不依赖 BasicTS。`vendor/BasicTS` 保留为此前实际复制的上游源码归档；当前只保留了 DLinear 所需的移动平均层及许可证到 `layers/`。参考目录与接口来自 [TSLib](https://github.com/thuml/Time-Series-Library) 和 [PARA-PV](https://github.com/weican1103/PARA-PV)，没有引入 PARA-PV 的模型、数据或额外研究路线。

```text
run.py                         TSLib 命令行入口
exp/exp_basic.py                构建 models/<模型>.py 中的 Model(configs)
exp/exp_long_term_forecasting.py 统一训练、验证、测试及结果保存
data_provider/data_factory.py  data_provider(args, flag)
data_provider/data_loader.py   唯一公开数据接口 Dataset_MMSP
data_provider/protocols.py     协议、训练/测试存储及 scaler 元信息
data_provider/mmsp_storage.py  原有 MMSP 功率、NWP、卫星读取逻辑
data_provider/split_utils.py   目标时间划分及窗口记录
models/                        各模型统一 forward 接口
layers/                        保留的模型计算层及上游声明
utils/                         指标、损失、调度器、权重恢复与 GPU 检查
scripts/run_mmsp.sh             在项目根目录执行 run.py
datasets/MMSP/data/             实际复制的数据
pretrained/                    Chronos-2、Chronos-T5、Llama、CLIP 等本地权重
checkpoints/{protocol}/         训练权重、配置和 scaler
outputs/{protocol}/            预测、指标、站点/时间元信息和运行日志
reports/                       两协议结果表、历史归档及迁移验收
```

## 统一数据接口

所有模型都通过 `data_provider(args, flag)` 获取同一个 `Dataset_MMSP`，`flag` 为 train/val/test。每个样本固定返回五项：

```python
seq_x, seq_y, seq_x_mark, seq_y_mark, covariates = dataset[index]
```

| 字段 | 单样本形状与含义 |
|---|---|
| seq_x | `[seq_len, 1]`，历史功率 |
| seq_y | `[label_len + pred_len, 1]`，历史标签段 + 未来评估标签 |
| seq_x_mark / seq_y_mark | `[时间长度, 4]`，归一化 hour/dayofweek/day/dayofyear |
| covariates.site_coords | `[2]`，站点经纬度 |
| covariates.historical_nwp | `[seq_len, 15]`，历史 NWP |
| covariates.future_nwp | `[pred_len, 15]`，未来 NWP |
| covariates.satellite | `[seq_len, 1, 64, 64]`，历史卫星图像 |
| covariates.satellite_coords | `[2, 64, 64]`，卫星网格坐标 |
| covariates.fusion_time | `[seq_len, 3, 64, 64]`，保留 FusionSF 使用的月/日/小时 |
| covariates.correlation_history | `[seq_len, 1]`，保留 Cross-Unet 使用的历史相关块 |
| covariates 元信息 | site_id、forecast_timestamps、输入/预测起止时间、split_id |

`--modalities power/power_nwp/satellite/all` 控制加载哪些模态；未加载字段为形状 `[0]` 的空张量，键名始终一致，可直接使用 PyTorch 默认 collate。默认根据模型所需输入选择加载范围，但 Dataset 不按模型分叉。`--modalities all` 可将完全相同的一份批次交给不同模型。

所有模型统一接受：

```python
model(x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=covariates)
```

实验层将 decoder 的未来功率部分置零，只保留历史 label 段；未来真实功率只用于损失和指标，不交给模型。模型返回 `[B, pred_len, 1]`，或含 prediction 与训练辅助损失的字典。NWP 和卫星归一化沿用已验收的数据逻辑，不另行标准化功率。

## 两套协议

- `in_domain`：站点 0–9 同站点按时间 60%/20%/20% 划分；窗口按完整目标时间归属。
- `zeroshot_v1`：训练和验证使用站点 10–19，测试使用站点 0–9；时间划分仍为 60%/20%/20%，测试复用训练 scaler。对监督模型表示跨站点泛化协议，并不表示模型从未训练。

默认 24→24、label_len=12。每协议保存 data_protocol.json 与 scalers.npz。不得在模型层改变站点划分、窗口或重新拟合 scaler。未来 NWP 的 CSV 缺少发布时间，在线可用性限制详见历史验收报告。

## 运行

只检查配置，不加载模型或启动实验：

```bash
python run.py --model DLinear --protocol in_domain --print_config
python run.py --model TimeXer --protocol zeroshot_v1 --print_config
```

训练并测试（GPU 必须由执行者明确选定为空闲卡）：

```bash
python run.py --task_name long_term_forecast --is_training 1 \
  --model DLinear --model_id DLinear_seed42 --data MMSP --protocol in_domain \
  --seq_len 24 --label_len 12 --pred_len 24 --seed 42 --use_gpu 1 --gpu 0
```

只评估已有权重，接受旧 BasicTS checkpoint 或新入口的 checkpoint.pth：

```bash
python run.py --is_training 0 --model DLinear --baseline_config legacy_fusionsf --protocol in_domain \
  --checkpoint checkpoints/in_domain/legacy_20261010/DLinear/power_coordinates/f892b16805555e2fb6b6bf47dbc034ef/FusionSFDLinear_best_val_MAE.pt \
  --model_id DLinear_existing_eval --use_gpu 1 --gpu 0
```

冻结模型测试：

```bash
python run.py --is_training 0 --model Chronos2 --protocol zeroshot_v1 \
  --nwp_mode history_future --model_id Chronos2_history_future --use_gpu 0
```

`--model_id` 区分独立运行；`--root_path`、`--model_path`、`--llm_path`、`--vlm_path` 可覆盖项目内默认资源。九个 baseline 均有统一入口：DLinear、PatchTST、FusionSF、CrossUnet、Chronos2、ChronosX、TimeXer、TimeLLM、TimeVLM；也支持 Cross-Unet、Chronos-2、Time-LLM、Time-VLM 名称。额外保留已有 TimesFM3；Ours 为待设计方法。Chronos-2 的 NWP 结果不代表 ChronosX。

默认 `--baseline_config official`，采用作者核心代码和指定官方脚本训练设置，详见 [核对报告](reports/baseline_official_audit_20261010.md) 与 `models/official_presets.py`。MMSP 保留 24→24、单功率目标、两套站点/时间/scaler 协议；这些数据适配项不等于作者原数据集的完整实验配置。旧 DLinear/PatchTST 权重需要 `--baseline_config legacy_fusionsf`，历史指标仍属旧配置。FusionSF 支持 `--preset script/experiment`、`--fusion_modalities 2/3`；script 仅支持三模态。FusionSF、Cross-Unet 在 MMSP 接口中要求 seq_len==pred_len。

FusionSF 的官方三模态 NWP 分支输入为 17 通道（两列经纬度 + 15 个气象变量）；模型适配层从统一 covariates 组合它们，Dataset 仍返回 15 个气象通道。此前 15 通道实现的权重可明确指定 `--guide_channels 15`。

ChronosX 使用 [官方 chronosx 分支](https://github.com/amazon-science/chronos-forecasting/tree/chronosx) 的 Chronos-T5-small + IIB/OIB，冻结骨干，训练 token 交叉熵；默认 5000 optimizer steps、梯度累积 2、每 100 步按验证 CE 保存最优模型。Time-LLM 使用冻结 Llama-7B（32 层）和 bf16；可通过 `--llm_model GPT2/BERT` 选择作者支持的其他骨干并指定本地权重。Time-VLM 使用官方 full-shot CLIP、历史序列生成的图像与文本、检索记忆和门控融合，GPU 训练 fp16，CPU 回退 fp32。其官方记忆库会在前向中更新，评估按数据顺序执行；迁入后记忆库随 checkpoint 保存和恢复。预训练权重全部从本地加载，运行入口不自动下载。

```bash
python run.py --model ChronosX --protocol in_domain --print_config
python run.py --model Time-LLM --protocol in_domain --print_config
python run.py --model Time-VLM --protocol zeroshot_v1 --print_config
```

本次 Time-LLM 的运行验收采用作者支持的 GPT-2 完整12层，权重保存到 `pretrained/gpt2/`；参考脚本的默认 Llama-7B 权重尚未下载完成，正式 Llama 骨干运行待核验。选择本地 GPT-2：

```bash
python run.py --model Time-LLM --llm_model GPT2 --protocol in_domain --print_config
```

运行绑定最多 4 个 CPU 核；CPU 模式固定 DataLoader 0 子进程。GPU 入口检查所选卡空闲，只使用一张指定卡。未启用自动实验队列。

## 保存与 GitHub

`checkpoints/{protocol}/{model_id}/checkpoint.pth` 保存验证最优模型、优化器和调度器；评估保存在 `outputs/{protocol}/{model_id}/`：pred.npy、true.npy、site_ids.npy、forecast_timestamps.npy、metrics.json、config.json、data_protocol.json、scalers.npz、run.log。结果索引仍为 `reports/{protocol}/results.csv`，历史索引为 `reports/legacy_experiment_index.csv`。

数据、预训练权重、实验权重与大部分结果不纳入 Git；目录仍在本机 PV-TSFM 内。完整转发需复制整个项目并保留 datasets/pretrained/checkpoints/outputs/reports，排除 .venv/.git/__pycache__，接收方重建环境。快照分支保存代码状态与验收清单，大文件通过本地完整目录保存。

origin：[zhao-zeal/PV-TSFM](https://github.com/zhao-zeal/PV-TSFM)。旧版：[basicts-snapshot-20261010](https://github.com/zhao-zeal/PV-TSFM/tree/basicts-snapshot-20261010)。迁移验收：`reports/tslib_migration_20261010.md`；此前验收：`reports/acceptance_20261010.md`。
