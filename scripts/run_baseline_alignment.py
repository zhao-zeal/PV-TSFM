"""The approved 100/10, prediction-RMSE baseline task; no hyperparameter search."""

import argparse
import ast
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ('TimeVLM', 'TimeXer', 'ChronosX')
PROTOCOLS = ('in_domain', 'zeroshot_v1')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(report):
    report.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    originals = {}
    for protocol in PROTOCOLS:
        rows = csv.DictReader((ROOT / 'reports' / protocol / 'results.csv').open())
        for row in rows:
            if row['model'] in (*MODELS, 'CrossUnet'):
                originals[(row['model'], protocol)] = row['result_dir']
    convergence = []
    for model in ('TimeVLM', 'TimeXer'):
        for protocol in PROTOCOLS:
            output = Path(originals[(model, protocol)])
            epochs = [dict(epoch=int(epoch), train_loss=float(loss), **ast.literal_eval(val),
                           next_epoch_lr=float(lr)) for epoch, loss, val, lr in re.findall(
                r'Epoch (\d+) train_loss=(\S+) validation=(\{.*?\}) lr=(\S+)',
                (output / 'run.log').read_text())]
            convergence.append(dict(model=model, protocol=protocol, source=str(output),
                best_epoch=min(epochs, key=lambda x: x['rmse'])['epoch'], epochs=epochs,
                conclusion=('Reached epoch10 cap one epoch after best; convergence not established'
                            if model == 'TimeVLM' else 'Best epoch4, patience3 stop at7; train loss falls but validation RMSE does not')))
    save(report / 'convergence_before.json', convergence)
    reused = []
    for protocol in PROTOCOLS:
        output = Path(originals[('CrossUnet', protocol)])
        config = json.loads((output / 'config.json').read_text())
        assert (config['train_epochs'], config['patience'], config['monitor']) == (100, 10, 'mse')
        reused.append(dict(protocol=protocol, output_dir=str(output),
            metrics=json.loads((output / 'metrics.json').read_text()),
            reason='Existing accepted max100/patience10; argmin MSE equals argmin RMSE',
            acceptance_record=str(ROOT / 'reports/CrossUnet_two_protocols_seed42_20261010/runs.json')))
    for model in MODELS:
        directory = report / model
        directory.mkdir()
        runs = []
        for protocol, cores in zip(PROTOCOLS, ([9, 10], [0, 7])):
            name = f'{model}_unified100_pat10_rmse_seed42_sl24_pl24_{stamp}'
            argv = ['--model', model, '--protocol', protocol, '--is_training', '1',
                '--baseline_config', 'unified_prediction', '--seed', '42', '--seq_len', '24',
                '--pred_len', '24', '--train_epochs', '100', '--patience', '10', '--monitor', 'rmse',
                '--num_workers', '0', '--use_gpu', '1', '--gpu', '0', '--model_id', name]
            if model == 'ChronosX':
                argv += ['--batch_size', '1024', '--eval_batch_size', '64', '--precision', 'bf16',
                         '--gradient_accumulation_steps', '1', '--learning_rate', '0.01', '--num_samples', '20']
            else:
                argv += ['--batch_size', '32', '--eval_batch_size', '32', '--lradj', 'plateau',
                         '--lr_patience', '2', '--learning_rate', '0.001' if model == 'TimeVLM' else '0.0001']
            runs.append(dict(protocol=protocol, model_id=name, argv=argv, state='pending', cpu_cores=cores,
                log=str(directory / f'{protocol}_training.log'),
                output_dir=str(ROOT / 'outputs' / protocol / name),
                checkpoint_dir=str(ROOT / 'checkpoints' / protocol / name),
                previous_output_dir=originals[(model, protocol)]))
        save(directory / 'runs.json', dict(model=model, state='prepared', baseline_config='unified_prediction',
            train_epochs=100, patience=10, monitor='rmse', seed=42, seq_len=24, pred_len=24,
            schedule_note='TimeVLM/TimeXer ReduceLROnPlateau patience2 triggers after 3 bad epochs; factor0.5; initial LR unchanged',
            runs=runs))
    save(report / 'runs.json', dict(state='prepared', crossunet_reused=reused,
        models={model: str(report / model / 'runs.json') for model in MODELS},
        input_controls='Original inputs retained and disclosed; no architecture-only comparison or extra input ablations',
        scope='Three changed baselines x two protocols; CrossUnet reuse; no additional seeds or search'))


def idle_gpu(exclude=()):
    from utils.resources import require_idle_gpus
    for index in range(8):
        if index in exclude:
            continue
        try:
            require_idle_gpus(str(index))
            return index
        except ValueError:
            pass
    raise RuntimeError('No idle GPU; leave remaining runs pending')


def environment():
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '2'
    return env


def sync_timexer(record):
    table = ROOT / 'reports/experiment_summary.csv'
    with table.open(encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    row = next(row for row in rows if row['models'] == 'TimeXer')
    for run in record['runs']:
        metrics = json.loads((Path(run['output_dir']) / 'metrics.json').read_text())
        prefix = 'MMSP in_domain' if run['protocol'] == 'in_domain' else 'MMSP zeroshot'
        for label, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')]:
            row[f'{prefix} {label}'] = f'{metrics[key]:.6f}'
    with table.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
    markdown = '| ' + ' | '.join(row[key] or '—' for key in fields) + ' |'
    doc.write_text(re.sub(r'^\| TimeXer \|.*$', markdown, doc.read_text(), flags=re.MULTILINE))


def execute(report):
    from utils.resources import require_idle_gpus
    root_record = json.loads((report / 'runs.json').read_text())
    if root_record['state'] != 'prepared':
        raise RuntimeError('This task has already started; inspect existing processes and artifacts')
    root_record.update(state='running', coordinator_pid=os.getpid())
    save(report / 'runs.json', root_record)
    try:
        for model in MODELS:
            path = report / model / 'runs.json'
            record = json.loads(path.read_text())
            processes = []
            for run in record['runs']:
                if run['state'] != 'pending':
                    raise RuntimeError('Refusing to repeat an existing run')
                gpu = idle_gpu(exclude=[item['gpu'] for item, _ in processes])
                run['argv'][run['argv'].index('--gpu') + 1] = str(gpu)
                run.update(gpu=gpu, state='training', started_at=datetime.now().isoformat())
                require_idle_gpus(str(gpu))
                command = ['taskset', '-c', ','.join(map(str, run['cpu_cores'])), sys.executable,
                           '-u', str(ROOT / 'run.py'), *run['argv']]
                with Path(run['log']).open('x') as handle:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=handle, stderr=subprocess.STDOUT)
                run['pid'] = process.pid
                processes.append((run, process))
                record['state'] = 'running'
                save(path, record)
            for run, process in processes:
                run.update(returncode=process.wait(), completed_at=datetime.now().isoformat())
                run['state'] = 'complete_pending_acceptance' if run['returncode'] == 0 else 'failed'
                save(path, record)
            if any(run['returncode'] for run, _ in processes):
                raise RuntimeError(f'{model} worker failed; see logs')
            if model in ('TimeVLM', 'ChronosX'):
                gpu = idle_gpu()
                with (path.parent / 'acceptance.log').open('x') as handle:
                    subprocess.run(['taskset', '-c', '9,10', sys.executable, '-u',
                        str(ROOT / 'scripts' / ('accept_timevlm.py' if model == 'TimeVLM' else 'accept_chronosx.py')),
                        '--report', str(path), '--gpu', str(gpu)], cwd=ROOT, env=environment(),
                        stdout=handle, stderr=subprocess.STDOUT, check=True)
                record = json.loads(path.read_text())
            else:
                for run in record['runs']:
                    receipt = json.loads((Path(run['output_dir']) / 'acceptance.json').read_text())
                    assert receipt['status'] == 'passed'
                    run.update(state='completed_artifact_acceptance_passed', acceptance=receipt)
                record['state'] = 'completed_artifact_acceptance_passed'
                sync_timexer(record)
            for run in record['runs']:
                previous = json.loads((Path(run['previous_output_dir']) / 'metrics.json').read_text())
                current = json.loads((Path(run['output_dir']) / 'metrics.json').read_text())
                run['comparison_to_previous'] = {key: dict(previous=previous[key], current=current[key],
                    absolute=current[key] - previous[key], relative_percent=100 * (current[key] / previous[key] - 1))
                    for key in ('mae', 'rmse', 'r2')}
            save(path, record)
        root_record.update(state='completed_artifact_acceptance_passed', finished_at=datetime.now().isoformat())
    except Exception as error:
        root_record.update(state='failed_stopped', error=str(error))
        raise
    finally:
        save(report / 'runs.json', root_record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else execute)(args.report.resolve())
