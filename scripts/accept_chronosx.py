"""Accept the authorized ChronosX full-test artifacts, including sampled prediction replay."""

import argparse
import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--protocol', choices=['in_domain', 'zeroshot_v1'], help='Accept and register one completed protocol while the other trains')
options = parser.parse_args()
options.report = options.report.resolve()
import numpy as np
import pandas as pd
import torch
from run import parse_args, configure_resources
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric
from utils.tools import load_model_checkpoint

record = json.loads(options.report.read_text())
if record.get('acceptance', {}).get('status') == 'passed_full_artifact_acceptance':
    raise SystemExit(0)
baseline_record = json.loads((ROOT / 'reports/FusionSF_two_protocols_seed42_20261010/runs.json').read_text())
chronos2_record = json.loads((ROOT / 'reports/Chronos2_two_protocols_seed42_20261011/runs.json').read_text())
raw = pd.read_csv(ROOT / 'datasets/MMSP/data/solar_power/solar_power.csv', parse_dates=['datetime'])
accepted = []
resources_configured = False
for run in record['runs']:
    if options.protocol and run['protocol'] != options.protocol:
        continue
    cached_path = options.report.parent / f'acceptance_{run["protocol"]}.json'
    if cached_path.exists():
        cached = json.loads(cached_path.read_text())
        saved_metrics = json.loads((Path(run['output_dir']) / 'metrics.json').read_text())
        assert cached['status'] == 'passed_full_artifact_acceptance'
        assert cached['metrics'] == saved_metrics and cached['output_dir'] == run['output_dir']
        assert cached['checkpoint'] == str(Path(run['checkpoint_dir']) / 'checkpoint.pth')
        accepted.append(cached)
        run.update(state='completed_artifact_acceptance_passed', current_test_metrics=saved_metrics)
        continue
    out, ckpt = Path(run['output_dir']), Path(run['checkpoint_dir'])
    config = json.loads((out / 'config.json').read_text())
    assert config == json.loads((ckpt / 'config.json').read_text())
    assert config['model'] == 'ChronosX' and config['seed'] == 42
    assert config['train_epochs'] == 200 and config['patience'] == 20
    steps_per_epoch = math.ceil(math.ceil(76570 / config['batch_size']) / config['gradient_accumulation_steps'])
    assert config['eval_steps'] == steps_per_epoch
    assert config['max_steps'] == steps_per_epoch * config['train_epochs']
    expected = parse_args(run['argv'])
    for key in ('batch_size', 'eval_batch_size', 'precision', 'gradient_accumulation_steps', 'learning_rate', 'num_samples'):
        assert config[key] == getattr(expected, key), key
    assert config['baseline_config'] == 'extended_budget'
    protocol = json.loads((out / 'data_protocol.json').read_text())
    training = json.loads((ckpt / 'data_protocol.json').read_text())
    for key in ('version', 'train_sites', 'validation_sites', 'test_sites', 'windows', 'target_time_ranges', 'scaler_fit'):
        assert protocol[key] == training[key], key
    assert protocol['version'] == run['protocol']
    assert protocol['windows'] == dict(train=76570, validation=25450, test=25450)
    assert protocol['train_sites'] == list(range(10) if run['protocol'] == 'in_domain' else range(10, 20))
    assert protocol['test_sites'] == list(range(10)) and protocol['modalities'] == 'power_nwp'
    reference = Path(next(x for x in baseline_record['acceptance']['runs'] if x['protocol'] == run['protocol'])['output_dir'])
    with np.load(ckpt / 'scalers.npz') as a, np.load(out / 'scalers.npz') as b, np.load(reference / 'scalers.npz') as c:
        assert set(a.files) == set(b.files) == {'nwp_mean', 'nwp_scale'}
        for key in a.files:
            assert np.array_equal(a[key], b[key]) and np.array_equal(a[key], c[key]), key
    arrays = {name: np.load(out / f'{name}.npy') for name in ('pred', 'true', 'site_ids', 'forecast_timestamps')}
    pred, truth, sites, times = [arrays[k] for k in ('pred', 'true', 'site_ids', 'forecast_timestamps')]
    assert pred.shape == truth.shape == (25450, 24, 1)
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    for name in ('true', 'site_ids', 'forecast_timestamps'):
        assert np.array_equal(arrays[name], np.load(reference / f'{name}.npy')), name
    for site in range(10):
        mask = sites == site
        assert mask.sum() == 2545 and np.any(pred[mask] != 0)
        group = raw[raw.site == site]
        raw_times = group.datetime.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        indices = np.searchsorted(raw_times, times[mask])
        assert np.array_equal(raw_times[indices], times[mask])
        assert np.array_equal(group.power.to_numpy(dtype=np.float32)[indices], truth[mask, :, 0])
    saved = json.loads((out / 'metrics.json').read_text())
    for key, value in metric(pred, truth).items():
        assert abs(value - saved[key]) < 1e-14
    lines = Path(run['log']).read_text().splitlines()
    epochs = []
    for line in lines:
        match = re.search(r'Epoch (\d+) train_loss=(\S+) validation=(\{.*\}) lr=', line)
        if match:
            validation = ast.literal_eval(match[3])
            assert np.isfinite(float(match[2])) and np.isfinite(validation['token_ce'])
            epochs.append((int(match[1]), validation))
    assert [ast.literal_eval(line.split('Complete test metrics: ', 1)[1]) for line in lines if 'Complete test metrics: ' in line] == [saved]
    checkpoint = torch.load(ckpt / 'checkpoint.pth', map_location='cpu', weights_only=True)
    assert checkpoint['config'] == config
    assert all(torch.isfinite(x).all() for x in checkpoint['model_state_dict'].values())
    best = min(epochs, key=lambda x: x[1]['token_ce'])
    assert checkpoint['epoch'] == best[0] and checkpoint['validation_metrics'] == best[1]
    assert checkpoint['step'] == best[0] * steps_per_epoch
    assert len(epochs) == 200 or len(epochs) - best[0] == 20
    with (ROOT / 'reports' / run['protocol'] / 'results.csv').open() as handle:
        matching = [x for x in csv.DictReader(handle) if x['result_dir'] == str(out.resolve())]
    assert len(matching) == 1 and matching[0]['baseline_config'] == 'extended_budget'
    for key in ('mae', 'rmse', 'r2'):
        assert float(matching[0][key]) == saved[key]
    args = parse_args(run['argv'] + ['--gpu', str(options.gpu)])
    if not resources_configured:
        configure_resources(args)
        resources_configured = True
    experiment = Exp_Long_Term_Forecast(args)
    load_model_checkpoint(experiment.model, ckpt / 'checkpoint.pth', experiment.device)
    experiment.model.eval()
    _, loader = experiment._get_data('test')
    rng = torch.load(out / 'prediction_rng_state.pth', map_location='cpu', weights_only=True)
    torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state(rng['cuda'], experiment.device)
    with torch.no_grad():
        replay = experiment._forward_batch(next(iter(loader)))[0].cpu().numpy()
    difference = float(np.max(np.abs(replay - pred[:len(replay)])))
    assert difference < 1e-5, difference
    del experiment, loader, replay, checkpoint
    torch.cuda.empty_cache()
    baseline = next(x['current_test_metrics'] for x in chronos2_record['runs'] if x['protocol'] == run['protocol'] and x['nwp_mode'] == 'history_future')
    item = dict(protocol=run['protocol'], status='passed_full_artifact_acceptance', metrics=saved,
                completed_epochs=len(epochs), best_epoch=best[0], selection='minimum validation token CE',
                checks=['all labels match raw CSV/reference', 'sites/timestamps/config/protocol agree',
                        'train-only NWP scaler unchanged', 'finite predictions/weights; each station nonzero',
                        'minimum validation CE checkpoint and patience20/max200', 'full metrics/log/unique index agree',
                        'first sampled batch replayed using saved CPU/CUDA RNG'],
                sampled_checkpoint_replay_max_absolute_difference=difference,
                comparison_to_frozen_Chronos2_history_future={key: dict(absolute_change=saved[key]-baseline[key],
                    relative_change_percent=(saved[key]/baseline[key]-1)*100) for key in ('mae', 'rmse', 'r2')},
                comparison_limit='Same MMSP power/history/future NWP; different pretrained backbone and adapters; seed42 only',
                output_dir=str(out), checkpoint=str(ckpt / 'checkpoint.pth'), log=run['log'])
    item['accepted_at'] = datetime.now().astimezone().isoformat()
    cached_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
    accepted.append(item)
    run.update(state='completed_artifact_acceptance_passed', current_test_metrics=saved)
if not options.protocol:
    record.update(state='completed_artifact_acceptance_passed', completed_at=datetime.now().astimezone().isoformat(),
                  acceptance=dict(status='passed_full_artifact_acceptance', runs=accepted, full_reinference=False))
    options.report.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
table = ROOT / 'reports/experiment_summary.csv'
with table.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle)
    fields, rows = reader.fieldnames, list(reader)
for row in rows:
    if row['models'] == 'ChronosX':
        row['input'] = '历史功率+历史/未来NWP'
        for item in accepted:
            prefix = 'MMSP in_domain' if item['protocol'] == 'in_domain' else 'MMSP zeroshot'
            for label, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')]:
                row[f'{prefix} {label}'] = f'{item["metrics"][key]:.6f}'
        markdown = '| ' + ' | '.join(row[key] or '—' for key in fields) + ' |'
with table.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
text = re.sub(r'^\| ChronosX \|.*$', markdown, doc.read_text(), flags=re.MULTILINE)
details = ['ChronosX已完成协议的正式测试指标经过完整产物验收后登记到主表。' if options.protocol else 'ChronosX两协议已完成200/20预算下的正式训练、测试和产物验收，主表已同步。']
for item in accepted:
    m = item['metrics']
    details.append(f'{item["protocol"]}完成{item["completed_epochs"]}轮，最低验证CE第{item["best_epoch"]}轮；MAE/RMSE/R²={m["mae"]:.6f}/{m["rmse"]:.6f}/{m["r2"]:.6f}。')
if options.protocol:
    details.append(f'本协议验收证据：`{cached_path.relative_to(ROOT)}`；另一协议尚未登记，完成后由既有队列自动验收补齐。两协议整体状态仍以原runs.json为准。')
else:
    details.append(f'唯一整体验收证据：`{options.report.relative_to(ROOT)}`；20次采样中位数、仅seed42，未验证多种子稳定性。此队列完成后停止，无后续实验。')
block = '<!-- CHRONOSX_STATUS -->\n\n' + '\n\n'.join(details) + '\n\n<!-- /CHRONOSX_STATUS -->'
if '<!-- /CHRONOSX_STATUS -->' in text:
    text = re.sub(r'<!-- CHRONOSX_STATUS -->.*?<!-- /CHRONOSX_STATUS -->', lambda _: block, text, flags=re.DOTALL)
else:
    text = text.replace('<!-- CHRONOSX_STATUS -->', block)
doc.write_text(text)
print('Accepted and registered:', ', '.join(item['protocol'] for item in accepted))
