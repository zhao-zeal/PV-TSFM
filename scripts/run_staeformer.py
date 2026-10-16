"""Run STAEformer training, formal testing and acceptance on both MMSP protocols."""

import argparse
import csv
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/STAEformer_two_protocols_seed42_20261016'
INDEX = REPORT / 'runs.json'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare():
    REPORT.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    runs = []
    for protocol in ('in_domain', 'zeroshot_v1'):
        model_id = f'STAEformer_official_seed42_sl24_pl24_power_time_{protocol}_{stamp}'
        runs.append(dict(
            protocol=protocol, model_id=model_id, state='pending',
            log=str(REPORT / f'{protocol}.log'),
            checkpoint_dir=str(ROOT / 'checkpoints' / protocol / model_id),
            output_dir=str(ROOT / 'outputs' / protocol / model_id),
        ))
    save(INDEX, dict(
        state='prepared',
        authorization='User: 将调参任务和STAEformer加入待办列表，等有GPU了开始运行',
        scope='STAEformer in_domain and zeroshot_v1; seed42; 24->24; train, formal test, acceptance',
        config=dict(batch_size=16, eval_batch_size=16, train_epochs=100, patience=10,
                    monitor='rmse', learning_rate=.001, precision='fp32', num_workers=0,
                    input='historical power + hour + weekday', num_nodes=1),
        architecture='Published temporal/spatial attention and adaptive embedding; single-node spatial attention degenerates',
        order='in_domain then zeroshot_v1; Cross-Unet already accepted',
        resources='One idle GPU; first two allowed CPU cores; two threads; workers0',
        failure_policy='Stop STAEformer queue on first failure; no automatic retry',
        runs=runs,
    ))


def idle_gpu(selected_gpu):
    from utils.resources import require_idle_gpus
    while True:
        for gpu in (selected_gpu,):
            try:
                require_idle_gpus(str(gpu))
                return gpu
            except ValueError:
                pass
        time.sleep(60)


def preflight(gpu, protocol, root_path):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from run import configure_resources, parse_args
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.losses import forecasting_loss

    args = parse_args(['--model', 'STAEformer', '--protocol', protocol, '--is_training', '1',
                       '--seed', '42', '--use_gpu', '1', '--gpu', str(gpu), '--num_workers', '0',
                       '--root_path', str(root_path)])
    configure_resources(args)
    torch.manual_seed(42); np.random.seed(42)
    experiment = Exp_Long_Term_Forecast(args)
    data, _ = experiment._get_data('train')
    batch = default_collate([data[0], data[len(data) - 1]])
    prediction, target, _ = experiment._forward_batch(batch)
    loss = forecasting_loss(prediction, target, args.model)
    loss.backward()
    gradients = [p.grad for p in experiment.model.parameters() if p.requires_grad and p.grad is not None]
    if (prediction.shape != target.shape or prediction.shape[1:] != (24, 1)
            or not torch.isfinite(prediction).all() or not torch.isfinite(loss)
            or not gradients or not all(torch.isfinite(value).all() for value in gradients)
            or sum(value.detach().abs().sum().item() for value in gradients) <= 0):
        raise RuntimeError('STAEformer real-GPU preflight failed')
    result = dict(status='passed_real_gpu_forward_backward', gpu=gpu,
                  sample_indices=[0, len(data) - 1], prediction_shape=list(prediction.shape),
                  loss=loss.item(), train_windows=len(data), test_accessed=False)
    del experiment, data, batch, prediction, target, loss, gradients
    gc.collect(); torch.cuda.empty_cache()
    return result


def sync_summary(status):
    table = ROOT / 'reports/experiment_summary.csv'
    with table.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle); fields, rows = reader.fieldnames, list(reader)
    row = next((item for item in rows if item['models'] == 'STAEformer'), None)
    if row is None:
        row = dict.fromkeys(fields, '')
        row.update(models='STAEformer', input='历史功率+小时/星期')
        rows.append(row)
    for run in status['runs']:
        prefix = 'MMSP in_domain' if run['protocol'] == 'in_domain' else 'MMSP zeroshot'
        for label, metric in (('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')):
            row[f'{prefix} {label}'] = f'{run["metrics"][metric]:.6f}'
    with table.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    document = ROOT / 'docs/EXPERIMENT_STATUS.md'
    text = document.read_text()
    old = next(line for line in text.splitlines() if line.startswith('| STAEformer |'))
    text = text.replace(old, '| ' + ' | '.join(row[field] or '—' for field in fields) + ' |', 1)
    text += ('\nSTAEformer两个协议完整训练、正式测试及产物验收完成；原始配置、日志、权重、预测和指标见'
             f'{INDEX.relative_to(ROOT) if INDEX.is_relative_to(ROOT) else INDEX}记录的路径。单节点空间注意力退化限制保留。\n')
    document.write_text(text)


def coordinate(selected_gpu, root_path):
    status = json.loads(INDEX.read_text())
    if status['state'] != 'prepared':
        raise ValueError(f'Unexpected state: {status["state"]}')
    try:
        status['order'] = 'in_domain then zeroshot_v1; Cross-Unet already accepted'
        for run in status['runs']:
            status.update(state='waiting_for_idle_gpu', active_protocol=run['protocol'])
            save(INDEX, status)
            gpu = idle_gpu(selected_gpu)
            run.update(state='preflight', gpu=gpu)
            save(INDEX, status)
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--check-protocol', run['protocol'], '--gpu', str(gpu),
                            '--root-path', str(root_path), '--report-dir', str(REPORT)],
                           cwd=ROOT, check=True)
            run['preflight'] = json.loads((REPORT / f'{run["protocol"]}_preflight.json').read_text())
            command = [sys.executable, '-u', str(ROOT / 'run.py'),
                       '--model', 'STAEformer', '--protocol', run['protocol'], '--is_training', '1',
                       '--seed', '42', '--seq_len', '24', '--pred_len', '24', '--batch_size', '16',
                       '--eval_batch_size', '16', '--train_epochs', '100', '--patience', '10',
                       '--monitor', 'rmse', '--learning_rate', '0.001', '--lradj', 'plateau',
                       '--precision', 'fp32', '--use_gpu', '1', '--gpu', str(gpu), '--num_workers', '0',
                       '--model_id', run['model_id'], '--root_path', str(root_path)]
            run['command'] = command
            run['state'] = 'running'; run['started_at'] = datetime.now().isoformat()
            save(INDEX, status)
            with Path(run['log']).open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            output = Path(run['output_dir'])
            acceptance = json.loads((output / 'acceptance.json').read_text())
            if acceptance['status'] != 'passed':
                raise RuntimeError(f'{run["protocol"]} acceptance: {acceptance["status"]}')
            run.update(state='accepted', completed_at=datetime.now().isoformat(),
                       metrics=json.loads((output / 'metrics.json').read_text()), acceptance=acceptance)
            save(INDEX, status)
        status.update(state='completed_formal_test_acceptance', active_protocol=None)
        sync_summary(status)
    except Exception as error:
        status.update(state='failed_stopped', error=repr(error))
        raise
    finally:
        save(INDEX, status)


if __name__ == '__main__':
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='Idle GPU index visible to nvidia-smi')
    parser.add_argument('--root-path', type=Path, default=ROOT / 'datasets/MMSP/data')
    parser.add_argument('--report-dir', type=Path,
                        default=ROOT / ('reports/STAEformer_two_protocols_seed42_' + datetime.now().strftime('%Y%m%dT%H%M%S')))
    parser.add_argument('--check-protocol', choices=['in_domain', 'zeroshot_v1'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    REPORT = args.report_dir.resolve()
    INDEX = REPORT / 'runs.json'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    if args.check_protocol:
        save(REPORT / f'{args.check_protocol}_preflight.json',
             preflight(args.gpu, args.check_protocol, args.root_path.resolve()))
    else:
        prepare()
        if not args.prepare:
            coordinate(args.gpu, args.root_path.resolve())
