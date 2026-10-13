"""Verify the approved single Site1 FusionSF run after training finishes."""

import ast
import csv
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
cores = sorted(os.sched_getaffinity(0))[:4]
os.sched_setaffinity(0, cores)
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = str(len(cores))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from utils.metrics import metric

torch.set_num_threads(len(cores))
torch.set_num_interop_threads(1)
DEST = ROOT / 'reports/site1_FusionSF_lr0p00016_20261010'
SETTING = 'FusionSF_official_script3_guide17_seed42_lr0p00016_site1'
output = ROOT / 'outputs/site1_v1' / SETTING
checkpoint = ROOT / 'checkpoints/site1_v1' / SETTING / 'checkpoint.pth'
metrics = json.loads((output / 'metrics.json').read_text())
config = json.loads((output / 'config.json').read_text())
protocol = json.loads((output / 'data_protocol.json').read_text())
assert config['guide_channels'] == 17 and config['seed'] == 42
assert config['learning_rate'] == 0.00016
assert config['checkpoint'] is None and config['is_training'] == 1
assert protocol['train_sites'] == protocol['validation_sites'] == protocol['test_sites'] == [1]
assert protocol['windows'] == {'train': 7657, 'validation': 2545, 'test': 2545}
prediction = np.load(output / 'pred.npy')
targets = np.load(output / 'true.npy')
sites = np.load(output / 'site_ids.npy')
times = np.load(output / 'forecast_timestamps.npy')
assert prediction.shape == targets.shape == (2545, 24, 1)
assert np.isfinite(prediction).all() and np.isfinite(targets).all()
assert np.all(sites == 1) and times.shape == (2545, 24)
raw = pd.read_csv(ROOT / 'datasets/MMSP/data/solar_power/solar_power.csv', parse_dates=['datetime'])
site = raw.loc[raw['site'] == 1].set_index('datetime')['power']
expected = site.loc[pd.to_datetime(times.reshape(-1))].to_numpy(dtype=np.float32).reshape(targets.shape)
assert np.array_equal(targets, expected)
recomputed = metric(prediction, targets)
assert all(abs(recomputed[key] - metrics[key]) < 1e-12 for key in ('mae', 'rmse', 'r2'))

rows = []
pattern = r'Epoch (\d+) train_loss=(\S+) validation=(\{.*\}) lr=(\S+)'
for match in re.finditer(pattern, (output / 'run.log').read_text()):
    validation = ast.literal_eval(match.group(3))
    rows.append({'epoch': int(match.group(1)), 'train_l1_plus_vq': float(match.group(2)),
                 'validation_mae': validation['mae'], 'validation_rmse': validation['rmse'],
                 'learning_rate_next_epoch': float(match.group(4))})
assert rows and np.isfinite([list(row.values()) for row in rows]).all()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
best = min(rows, key=lambda row: row['validation_mae'])
assert saved['epoch'] == best['epoch']
assert saved['validation_metrics']['mae'] == best['validation_mae']
assert all(torch.isfinite(tensor).all().item() for tensor in saved['model_state_dict'].values())
with (DEST / 'training_curve.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot([row['epoch'] for row in rows], [row['train_l1_plus_vq'] for row in rows])
axes[0].set(xlabel='Epoch', ylabel='Training L1 + VQ loss', title='FusionSF Site #1, seed42')
axes[1].plot([row['epoch'] for row in rows], [row['validation_mae'] for row in rows])
axes[1].axvline(best['epoch'], color='tab:red', linestyle='--', label=f"Best epoch {best['epoch']}")
axes[1].set(xlabel='Epoch', ylabel='Validation MAE', title='Validation selects the checkpoint')
axes[1].legend()
for axis in axes:
    axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(DEST / 'training_curve.png', dpi=180)
plt.close(fig)

report = {'status': 'passed', 'model': 'FusionSF', 'data_protocol': 'site1_v1',
          'seed': 42, 'input_len': 24, 'output_len': 24, 'guide_channels': 17,
          'learning_rate': 0.00016,
          'epochs_completed': rows[-1]['epoch'], 'best_epoch': best['epoch'],
          'best_validation_mae': best['validation_mae'], 'metrics': metrics,
          'targets_match_original_csv': True, 'model_parameters_finite': True,
          'prediction_min': float(prediction.min()), 'prediction_max': float(prediction.max()),
          'negative_prediction_fraction': float(np.mean(prediction < 0)),
          'target_min': float(targets.min()), 'target_max': float(targets.max()),
          'comparison': {'previous_independent_site1_result': None,
                         'absolute_change': None, 'relative_change_percent': None,
                         'reason': 'No previously accepted independent Site1 run; ten-site metrics are not a comparable reference.'},
          'checkpoint': str(checkpoint), 'output_dir': str(output),
          'training_log': str(DEST / 'training.log'), 'curve': str(DEST / 'training_curve.png'),
          'limitations': ['Single seed Site1 exploration; one user-approved learning-rate retry after zero-output failure.',
                         'NWP issue times are absent; online availability is unverified.']}
(DEST / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
with (DEST / 'results.csv').open('w', newline='') as handle:
    fields = ['model', 'data_protocol', 'seed', 'input_len', 'output_len', 'guide_channels',
              'best_epoch', 'best_validation_mae', 'test_windows', 'mae', 'rmse', 'r2']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerow({key: report.get(key, metrics.get(key)) for key in fields})
print(json.dumps(report, indent=2), flush=True)
