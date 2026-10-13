"""Occasional checks of the two retained PV-TSFM runs; never controls training."""
import ast
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np

ROOT = Path('/home/zhaopp/workspace/PV-TSFM')
REPORT = Path(__file__).resolve().parent
RUN = 'FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full'
JOBS = [
    ('PVTSFM_in_domain_guide17', 2635253, REPORT / 'in_domain_training.log', ROOT / 'outputs/in_domain' / RUN / 'metrics.json', 'in_domain', False),
    ('PVTSFM_zeroshot_guide17', 2642044, REPORT / 'zeroshot_v1_training.log', ROOT / 'outputs/zeroshot_v1' / RUN / 'metrics.json', 'zeroshot_v1', False),
]

# Compute the all-zero reference from validation targets only, preserving
# float32 labels and complete 24-step windows from the accepted protocol.
power = {}
with (ROOT / 'datasets/MMSP/data/solar_power/solar_power.csv').open() as file:
    for row in csv.DictReader(file):
        if '2021-11-19' <= row['datetime'] < '2022-03-06':
            power.setdefault(int(row['site']), []).append(float(row['power']))
references = {}
for protocol in ('in_domain', 'zeroshot_v1'):
    meta = json.loads((ROOT / 'checkpoints' / protocol / RUN / 'data_protocol.json').read_text())
    labels = np.concatenate([
        np.lib.stride_tricks.sliding_window_view(np.array(power[site], dtype=np.float32), 24)
        for site in meta['validation_sites']
    ]).astype(np.float64)
    if len(labels) != meta['windows']['validation']:
        raise ValueError('Validation zero-reference window count differs from accepted protocol')
    references[protocol] = {'mae': float(np.abs(labels).mean()), 'rmse': float(np.sqrt(np.square(labels).mean()))}
(REPORT / 'validation_zero_reference.json').write_text(json.dumps(references, indent=2) + '\n')

seen = set()
print(json.dumps({'checker_pid': os.getpid(), 'interval_seconds': [600, 1200], 'zero_references': references}), flush=True)
while True:
    states = []
    for name, pid, log, metrics, protocol, rounded in JOBS:
        text = log.read_text(errors='replace')
        issues = []
        progress = []
        if rounded:
            progress = [(i + 1, float(a), float(b)) for i, (a, b) in enumerate(re.findall(r'Result <val>:.*?val/MAE: ([\d.eE+-]+), val/RMSE: ([\d.eE+-]+)', text))]
        else:
            for epoch, validation in re.findall(r'Epoch (\d+) train_loss=\S+ validation=(\{.*?\}) lr=', text):
                if re.search(r'\b(?:nan|inf)\b', validation, re.I):
                    issues.append('nonfinite_validation')
                    continue
                values = ast.literal_eval(validation)
                progress.append((int(epoch), values['mae'], values['rmse']))
        if 'Traceback (most recent call last)' in text:
            issues.append('traceback_in_log')
        if re.search(r'(?:train_loss=|train/loss: )[-+]?(?:nan|inf)\b', text, re.I):
            issues.append('nonfinite_training_loss')
        recent = progress[-3:]
        tolerance = 0.000051 if rounded else 1e-8
        zero = references[protocol]
        if len(recent) >= 2 and all(abs(mae - zero['mae']) < tolerance and abs(rmse - zero['rmse']) < tolerance for _, mae, rmse in recent[-2:]):
            issues.append('suspected_zero_predictions_two_validations; output_check_required')
        elif len(recent) == 3 and all(mae == recent[0][1] and rmse == recent[0][2] for _, mae, rmse in recent):
            issues.append('unchanged_validation_three_epochs; output_check_required')
        alive = Path(f'/proc/{pid}').exists()
        complete = metrics.exists()
        if not alive and not complete:
            issues.append('process_exited_without_test_metrics')
        state = {'run': name, 'pid': pid, 'alive': alive, 'test_complete': complete, 'latest_validation': progress[-1] if progress else None, 'issues': issues}
        states.append(state)
        event = (name, tuple(issues), complete)
        if (issues or complete) and event not in seen:
            seen.add(event)
            record = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), **state}
            with (REPORT / 'random_check_events.jsonl').open('a') as file:
                file.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    delay = random.randint(600, 1200)
    (REPORT / 'random_check_status.json').write_text(json.dumps({'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), 'checker_pid': os.getpid(), 'next_check_seconds': delay, 'runs': states}, indent=2) + '\n')
    if not any(state['alive'] for state in states):
        break
    time.sleep(delay)
