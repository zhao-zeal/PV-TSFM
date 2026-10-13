"""Accept the two existing TimeVLM runs; never starts or repeats training."""

import argparse
import ast
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from run import parse_args, configure_resources
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric
from utils.tools import load_model_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', type=int, help='Use one idle GPU for both sequential checkpoint replays')
parser.add_argument('--report', type=Path, default=ROOT / 'reports/TimeVLM_two_protocols_seed42_20261011/runs.json')
options = parser.parse_args()
report = options.report
record = json.loads(report.read_text())
if record.get('acceptance', {}).get('status') == 'passed_full_artifact_acceptance':
    raise SystemExit(0)
raw = pd.read_csv(ROOT / 'datasets/MMSP/data/solar_power/solar_power.csv', parse_dates=['datetime'])
accepted = []
for run in record['runs']:
    out, ckpt = Path(run['output_dir']), Path(run['checkpoint_dir'])
    saved = json.loads((out / 'metrics.json').read_text())
    config = json.loads((out / 'config.json').read_text())
    assert config == json.loads((ckpt / 'config.json').read_text())
    protocol = json.loads((out / 'data_protocol.json').read_text())
    train_protocol = json.loads((ckpt / 'data_protocol.json').read_text())
    for key in ['version', 'train_sites', 'validation_sites', 'test_sites', 'windows', 'target_time_ranges', 'scaler_fit']:
        assert protocol[key] == train_protocol[key], key
    assert protocol['scaler_fit'] == {}, 'Power-only TimeVLM fits no NWP/satellite scalers'
    assert protocol['windows'] == {'train': 76570, 'validation': 25450, 'test': 25450}
    assert protocol['train_sites'] == list(range(10) if run['protocol'] == 'in_domain' else range(10, 20))
    assert protocol['test_sites'] == list(range(10))
    arrays = {name: np.load(out / f'{name}.npy') for name in ['pred', 'true', 'site_ids', 'forecast_timestamps']}
    pred, truth, sites, times = [arrays[k] for k in ['pred', 'true', 'site_ids', 'forecast_timestamps']]
    assert pred.shape == truth.shape == (25450, 24, 1)
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    reference = ROOT / 'outputs' / run['protocol'] / 'TimeXer_official_seed42_sl24_pl24_power_nwp_full'
    for name in ['true', 'site_ids', 'forecast_timestamps']:
        assert np.array_equal(arrays[name], np.load(reference / f'{name}.npy')), name
    for site in range(10):
        mask = sites == site
        assert mask.sum() == 2545 and np.any(pred[mask] != 0)
        group = raw[raw.site == site]
        raw_times = group.datetime.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        idx = np.searchsorted(raw_times, times[mask])
        assert np.array_equal(raw_times[idx], times[mask])
        assert np.array_equal(group.power.to_numpy(dtype=np.float32)[idx], truth[mask, :, 0])
    recomputed = metric(pred, truth)
    for key in ['mae', 'rmse', 'r2']:
        assert abs(recomputed[key] - saved[key]) < 1e-14, key
    lines = Path(run['log']).read_text().splitlines()
    prior_lines = []
    if run.get('source_log'):
        for line in Path(run['source_log']).read_text().splitlines():
            match = re.search(r'Epoch (\d+) ', line)
            if match and int(match[1]) <= run['resume_epoch']:
                prior_lines.append(line)
    epochs = []
    for line in prior_lines + lines:
        match = re.search(r'Epoch (\d+) train_loss=(\S+) validation=(\{.*\}) lr=', line)
        if match:
            assert np.isfinite(float(match[2]))
            validation = ast.literal_eval(match[3])
            assert all(np.isfinite(v) for v in validation.values())
            epochs.append((int(match[1]), validation))
    logged_test = [ast.literal_eval(line.split('Complete test metrics: ', 1)[1]) for line in lines if 'Complete test metrics: ' in line]
    assert logged_test == [saved]
    checkpoint = torch.load(ckpt / 'checkpoint.pth', map_location='cpu', weights_only=True)
    assert checkpoint['config'] == config
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    best = min(epochs, key=lambda item: item[1]['mse'])
    assert checkpoint['epoch'] == best[0] and checkpoint['validation_metrics'] == best[1]
    assert epochs[-1][0] == config['train_epochs'] or epochs[-1][0] - best[0] == config['patience']
    state = checkpoint['model_state_dict']
    assert state['patch_memory_bank.patches'].shape == (100, 128)
    assert 0 <= int(state['patch_memory_bank._pointer']) <= 100
    rows = list(csv.DictReader((ROOT / 'reports' / run['protocol'] / 'results.csv').open()))
    matched = [row for row in rows if row['result_dir'] == str(out.resolve())]
    assert len(matched) == 1
    for key in ['mae', 'rmse', 'r2']:
        assert float(matched[0][key]) == saved[key]
    del checkpoint, state
    args = parse_args(run['argv'] + ['--train_epochs', str(config['train_epochs']), '--patience', str(config['patience'])] +
                      (['--gpu', str(options.gpu)] if options.gpu is not None else []))
    if not accepted:
        configure_resources(args)
    elif options.gpu is None:
        from utils.resources import require_idle_gpus
        require_idle_gpus(str(args.gpu))
    experiment = Exp_Long_Term_Forecast(args)
    load_model_checkpoint(experiment.model, ckpt / 'checkpoint.pth', experiment.device)
    experiment.model.eval()
    _, loader = experiment._get_data('test')
    replay = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            prediction, _, _ = experiment._forward_batch(batch)
            replay.append(prediction.float().cpu().numpy())
            if i == 1:
                break
    replay = np.concatenate(replay)
    difference = float(np.max(np.abs(replay - pred[:len(replay)])))
    assert difference <= 1e-5, difference
    del experiment, loader, replay
    torch.cuda.empty_cache()
    baseline = json.loads((reference / 'metrics.json').read_text())
    comparison = {key: {'absolute_change': saved[key] - baseline[key], 'relative_change_percent': (saved[key] / baseline[key] - 1) * 100} for key in ['mae', 'rmse', 'r2']}
    evidence = {name: str(out / f'{name}.npy') for name in arrays}
    evidence.update(metrics=str(out / 'metrics.json'), config=str(out / 'config.json'), protocol=str(out / 'data_protocol.json'), checkpoint=str(ckpt / 'checkpoint.pth'), log=run['log'])
    accepted.append({'protocol': run['protocol'], 'status': 'passed_full_artifact_acceptance', 'completed_epochs': len(epochs), 'best_epoch': best[0], 'selection': 'minimum validation MSE; checkpoint includes memory state', 'metrics': saved, 'checks': ['all labels match raw CSV and accepted test reference', 'all site/time indices match accepted protocol', 'config/checkpoint/protocol match; no modality scaler is required', 'predictions/weights finite; every station nonzero', 'minimum validation MSE checkpoint selected', 'independent metric recomputation/log/unique index agree', 'first64 chronological test predictions replayed from checkpoint memory on idle GPU fp16'], 'first64_checkpoint_replay_max_absolute_difference': difference, 'checkpoint_replay_gpu': args.gpu, 'comparison_to_TimeXer_same_protocol': comparison, 'comparison_limit': 'TimeVLM consumes power only and generated image/constant text; TimeXer additionally consumes historical NWP; difference does not isolate architecture effect', 'negative_prediction_fraction': float(np.mean(pred < 0)), 'prediction_range': [float(pred.min()), float(pred.max())], 'evidence': evidence})
    if run.get('source_output_dir'):
        original_metrics = json.loads((Path(run['source_output_dir']) / 'metrics.json').read_text())
        accepted[-1]['comparison_to_original_10epoch_TimeVLM'] = {
            key: {'absolute_change': saved[key] - original_metrics[key],
                  'relative_change_percent': (saved[key] / original_metrics[key] - 1) * 100}
            for key in ['mae', 'rmse', 'r2']}
        accepted[-1]['continued_epochs'] = len(epochs) - run['resume_epoch']
        accepted[-1]['resume_epoch'] = run['resume_epoch']
    run.update(state='completed_artifact_acceptance_passed', current_test_metrics=saved)
    print(run['protocol'], saved, flush=True)
record.update(state='completed_artifact_acceptance_passed', completed_at=datetime.now().astimezone().isoformat(), acceptance={'status': 'passed_full_artifact_acceptance', 'checked_at': datetime.now().astimezone().isoformat(), 'runs': accepted, 'new_training': True, 'acceptance_full_reinference': False, 'acceptance_replay': 'first64 chronological samples per protocol only; preserves stateful memory; metrics from original full formal test'})
report.write_text(json.dumps(record, indent=2) + '\n')
table = ROOT / 'reports/experiment_summary.csv'
with table.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle)
    fields, rows = reader.fieldnames, list(reader)
for row in rows:
    if row['models'] == 'Time-VLM':
        row['input'] = '历史功率+衍生图像/固定文本'
        for item in accepted:
            prefix = 'MMSP in_domain' if item['protocol'] == 'in_domain' else 'MMSP zeroshot'
            for label, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')]:
                row[f'{prefix} {label}'] = f'{item["metrics"][key]:.6f}'
        markdown_row = '| ' + ' | '.join(row[f] for f in fields) + ' |'
with table.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
text = doc.read_text()
text = text.replace('## Time-VLM复现（当前任务）', '## Time-VLM复现（已完成）')
text = re.sub(r'^\| Time-VLM \|.*$', markdown_row, text, flags=re.MULTILINE)
if record.get('resume_training'):
    lines = ['TimeVLM续训两协议已完成正式测试及完整产物验收；总表更新为200轮/patience20预算版本，原10轮结果及产物完整保留。']
    for item in accepted:
        m = item['metrics']
        lines.append(f'{item["protocol"]}本次续训{item["continued_epochs"]}轮，结束epoch={item["completed_epochs"]}，最佳epoch={item["best_epoch"]}，MAE/RMSE/R²={m["mae"]:.6f}/{m["rmse"]:.6f}/{m["r2"]:.6f}。')
    lines.append(f'证据：`{report.relative_to(ROOT)}`；原10轮的随机数/AMP状态及第10轮末状态缺失，从最佳第9轮分支续训，不声称完整轨迹恢复。后续按用户授权在GPU0串行运行ChronosX。')
    text = text.replace('<!-- TIMEVLM_CONTINUATION_STATUS -->', '\n\n'.join(lines))
    doc.write_text(text)
    raise SystemExit(0)
text = text.replace('正式Metric尚未生成，主表保持空白。', '两个协议已完成训练、正式测试与完整产物验收，主表已同步。')
text = text.replace('Time-VLM两协议已按用户最新任务启动，等待完整训练、正式测试与产物验收；', 'Time-VLM两协议已完成训练、正式测试与完整产物验收并填表；')
details = '\n\n完整产物验收通过：全部25,450×24标签与原始CSV、已验收测试参考一致，站点/时间索引、配置/协议匹配；预测/权重有限、各站非全零；统一全量Metric重算与指标、日志及唯一分协议索引一致。各协议从保存权重及记忆状态顺序重放首64个样本进行身份检查，不重做完整推理。纯功率输入不拟合NWP/卫星scaler。\n'
for item in accepted:
    m = item['metrics']
    deltas = item['comparison_to_TimeXer_same_protocol']
    comparison_text = '、'.join(f'{label} {deltas[key]["absolute_change"]:+.6f}（{deltas[key]["relative_change_percent"]:+.2f}%）' for label, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R²', 'r2')])
    details += f'\n{item["protocol"]}完成{item["completed_epochs"]}轮，最低验证MSE权重第{item["best_epoch"]}轮；测试MAE/RMSE/R²={m["mae"]:.6f}/{m["rmse"]:.6f}/{m["r2"]:.6f}。同协议对照TimeXer：{comparison_text}。'
details += '\n\nTime-VLM只消费历史功率及衍生图像/固定文本，TimeXer还消费历史NWP，差异不能单独归因于架构。仅seed42；官方文本截断、图像无梯度及顺序记忆更新行为保留，未开展消融或修复。两协议验收后按用户新授权串行执行FusionSF三个输入版本。唯一验收证据为本轮runs.json的acceptance；未生成新图表或重复Markdown报告。\n'
text = text.replace('\n## Chronos-2 Adapter迁移与复现（已完成）', details + '\n## Chronos-2 Adapter迁移与复现（已完成）')
doc.write_text(text)
