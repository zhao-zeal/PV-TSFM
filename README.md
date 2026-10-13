# PV-TSFM

统一的光伏预测实验框架，采用 TSLib 分层和统一 MMSP 数据接口。历史实验提供研究线索；在本框架完成并验收的正式评估提供论文 Metric。

## 目录

```text
run.py             统一运行入口
data_provider/     数据读取、时间窗口、站点协议和 scaler
models/            模型适配，统一 Model(configs) 与 forward
layers/            模型核心计算层及上游许可证
exp/               统一训练、验证、测试和产物保存
utils/             指标、损失、调度器、资源检查和权重恢复
scripts/           运行脚本与官方配置参考
datasets/          本地数据
pretrained/        本地预训练权重
checkpoints/       按协议和运行名保存训练权重
outputs/           配置、协议、预测、标签、索引和运行日志
reports/           运行记录、验收说明和正式结果索引
docs/              数据协议与当前实验状态
```

运行代码不依赖旧 BasicTS 或 FusionSF 目录。模型核心代码与许可证已保留在 layers/；不保存整套上游仓库或内部 BasicTS 归档。`.venv/` 是本机环境，数据和实验大文件不随 Git 同步。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

## 数据与模型

当前正式协议为 `in_domain` 和 `zeroshot_v1`，MMSP 小时数据，24→24，seed42。`site1_v1`用于Site #1单站点探索，已完成数据协议核对；新训练需单独批准。站点、时间窗口、输入模态、归一化和指标定义见 [数据协议](docs/DATA_PROTOCOLS.md)。

所有模型通过 `data_provider(args, flag)` 使用 `Dataset_MMSP`：

```python
seq_x, seq_y, seq_x_mark, seq_y_mark, covariates = dataset[index]
model(x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=covariates)
```

历史功率为 `[seq_len, 1]`；seq_y包括label_len=12历史段和pred_len=24目标段。实验层将decoder未来功率置零。covariates统一提供历史/未来NWP、卫星、坐标、日历及站点/时间索引；未加载字段为空张量。各模型消费的模态不同，结果必须记录实际输入。

支持 DLinear、PatchTST、FusionSF、Cross-Unet、Chronos-2、ChronosX、TimeXer、Time-VLM；保留既有 TimesFM3 入口。Time-LLM已移除，Ours尚未设计。默认 `--baseline_config official`；官方来源及配置参照见 [配置说明](reports/baseline_official_audit_20261010.md)、`models/baseline_sources.json` 和 `models/official_presets.py`。接入代码不代表完成性能复现。

## 运行

查看配置，仅读取参数，不加载模型或运行实验：

```bash
python run.py --model FusionSF --protocol in_domain --seed 42 --print_config
```

训练并正式测试；须先确认GPU空闲，示例不构成自动运行授权：

```bash
python run.py --is_training 1 --model FusionSF --protocol in_domain \
  --model_id FusionSF_seed42 --seed 42 --seq_len 24 --pred_len 24 \
  --num_workers 0 --use_gpu 1 --gpu 0
```

仅评估本框架已有权重：

```bash
python run.py --is_training 0 --model FusionSF --protocol in_domain \
  --checkpoint checkpoints/in_domain/FusionSF_seed42/checkpoint.pth \
  --model_id FusionSF_seed42_eval --seed 42 --num_workers 0 --use_gpu 0
```

冻结Chronos-2推理：

```bash
python run.py --is_training 0 --model Chronos2 --protocol zeroshot_v1 \
  --nwp_mode history_future --eval_batch_size 64 --model_id Chronos2_seed42 \
  --seed 42 --num_workers 0 --use_gpu 0
```

`model_id`区分独立运行，不覆盖已有实验。旧DLinear/PatchTST版本可用 `legacy_fusionsf`，旧FusionSF权重须匹配guide15/17、preset和模型结构；兼容入口不表示这些旧成绩可作当前官方版本结果。

## 记录与验收

- [实验总表](docs/EXPERIMENT_STATUS.md#实验总表) / [下载CSV](reports/experiment_summary.csv)：models、第二列input及两个MMSP协议的MAE、RMSE、R²，seed42、24→24，正式测试成绩与用户明确批准展示的历史登记成绩均注明证据状态；DLinear/PatchTST zeroshot原产物缺失，未完成当前完整验收。
- [Chronos-2四种输入Metric](docs/EXPERIMENT_STATUS.md#chronos-2四种输入metric) / [下载CSV](reports/chronos2_input_metrics.csv)：第二列input，两个协议的MAE/RMSE/R²均保留六位小数；主表固定history_future。
- `checkpoints/{protocol}/{model_id}/`：配置、协议/scaler与验证最优checkpoint。
- `outputs/{protocol}/{model_id}/`：训练期间保存run.log；正式测试保存配置、协议/scaler、pred.npy、true.npy、site_ids.npy、forecast_timestamps.npy与metrics.json。
- `reports/{protocol}/results.csv`：当前新框架实验的测试结果索引；验收状态仍以对应实验记录为准，存在一行不等于已验收。
- [实验状态](docs/EXPERIMENT_STATUS.md)：当前授权、运行位置、异常和待办。每个模型完成两个协议后提交验收，等待下一模型授权。

代码检查、单批次计算、完整训练、正式评估分别记录。论文引用只使用协议/输入明确且经完整测试与验收的结果。历史Metric默认用于参照；用户批准且通过完整证据验收的Chronos-2及DLinear/PatchTST旧版本结果复用已登记，来源分别见reports/Chronos2_two_protocols_seed42_20261011/runs.json和reports/DLinear_PatchTST_migration_seed42_20261011/runs.json，不称为本轮重新推理或训练。DLinear/PatchTST主表采用FusionSF仓库旧实现/旧配置，不是原作者官方版本；其zeroshot按本次明确授权填入历史登记成绩，并标注原产物缺失、未完成当前完整验收，不进入分协议正式测试索引。输入或协议不一致则重新运行后验收。

GitHub保存代码、实验指标、必要配置和训练日志、验收记录及分析；结果入口为[实验总表](reports/experiment_summary.csv)、[FusionSF输入结果](reports/fusionsf_input_metrics.csv)、[Adapter结果](reports/chronos2_adapter_metrics.csv)和[实验状态](docs/EXPERIMENT_STATUS.md)。相同配置、协议和日志只保留必要副本；重复副本、缓存清单及旧监控/协调材料不再跟踪，本机文件仍保留。数据集、预训练权重、模型权重及预测/特征数组未上传；完整重放实验需另外保存datasets/pretrained/checkpoints/outputs/reports并重建环境。记录中的绝对路径指向原实验机器，部分依赖仅保存在本机。

GitHub：[zhao-zeal/PV-TSFM](https://github.com/zhao-zeal/PV-TSFM)。旧代码快照分支：`basicts-snapshot-20261010`。

Chronos-2 Adapter研究版本：统一入口`run.py --model Chronos2Adapter --adapter_version b131_linear_v1`（历史E11）或`b131_mlp63_v1`（历史E12）。两阶段结构/原训练设置保留，统一标准化NWP输入下重新训练；这两个是研究变体，不属于官方Chronos-2基线或已定Ours。版本与验收见[实验状态](docs/EXPERIMENT_STATUS.md)，两协议指标见[Adapter指标CSV](reports/chronos2_adapter_metrics.csv)；当前seed42四组已完成训练、正式测试和完整产物验收。
