"""Prepare A05 commands, or run explicitly selected variants serially on one GPU."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=['a', 'b', 'c'], default=['a'])
    parser.add_argument('--protocols', nargs='+', choices=['in_domain', 'zeroshot_v1'],
                        default=['in_domain', 'zeroshot_v1'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--root-path', type=Path, default=ROOT / 'datasets/MMSP/data')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'pretrained/chronos-2')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--run', action='store_true', help='Execute training and formal tests; default: prepare only')
    args = parser.parse_args()
    cores = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, cores)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(len(cores))
    from run import parse_args

    stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
    report = (args.report_dir or ROOT / f'reports/A05_seed42_{stamp}').resolve()
    trials, missing = [], []
    receipt = ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json'
    for path in (args.root_path, args.model_path, receipt):
        if not path.exists():
            missing.append(str(path))
    for protocol in dict.fromkeys(args.protocols):
        base_id = 'Chronos2Adapter_b131_linear_v1_seed42_sl24_pl24_history_future'
        checkpoint = ROOT / 'checkpoints' / protocol / base_id / 'checkpoint.pth'
        for path in (checkpoint, *(ROOT / 'outputs' / protocol / base_id / name
                                  for name in ('pred.npy', 'true.npy', 'site_ids.npy', 'forecast_timestamps.npy'))):
            if not path.is_file():
                missing.append(str(path))
        for variant in dict.fromkeys(args.variants):
            model_id = f'Chronos2MCDecoder_A05{variant}_seed42_sl24_pl24_{stamp}'
            command = [sys.executable, '-u', str(ROOT / 'run.py'),
                       '--model', 'Chronos2MCDecoder', '--protocol', protocol,
                       '--is_training', '1', '--a05_variant', variant, '--seed', '42',
                       '--seq_len', '24', '--pred_len', '24', '--batch_size', '128',
                       '--eval_batch_size', '256', '--cache_batch_size', '64',
                       '--train_epochs', '100', '--patience', '10', '--monitor', 'rmse',
                       '--learning_rate', '0.0001', '--precision', 'fp32', '--num_workers', '0',
                       '--use_gpu', '1', '--gpu', str(args.gpu), '--root_path', str(args.root_path.resolve()),
                       '--model_path', str(args.model_path.resolve()),
                       '--adapter_base_checkpoint', str(checkpoint), '--model_id', model_id,
                       '--a05_cache_id', f'Chronos2MCDecoder_native_cache_{stamp}']
            parse_args(command[3:])
            trials.append(dict(protocol=protocol, variant=variant, model_id=model_id,
                               command=command, state='pending'))
    report.mkdir(parents=True, exist_ok=False)
    status = dict(state='prepared', cpu_cores=cores, missing_resources=missing, trials=trials,
                  cache_note='Shared by variants within each protocol; allow about 48 GiB for two protocols')

    def save():
        (report / 'runs.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')

    save()
    print(f'Run index: {report / "runs.json"}', flush=True)
    print(f'{len(trials)} commands prepared; {len(missing)} missing resources.', flush=True)
    for path in missing:
        print(f'MISSING: {path}', flush=True)
    if not args.run:
        return
    if missing:
        status['state'] = 'blocked_missing_resources'
        save()
        raise FileNotFoundError('Upload the listed A01 resources before running A05')
    from utils.resources import require_idle_gpus
    try:
        for trial in trials:
            require_idle_gpus(str(args.gpu))
            trial['state'] = status['state'] = 'running'
            save()
            log = report / f'{trial["protocol"]}_{trial["variant"]}.log'
            trial['log'] = str(log)
            with log.open('x') as handle:
                subprocess.run(trial['command'], cwd=ROOT, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            output = ROOT / 'outputs' / trial['protocol'] / trial['model_id']
            acceptance = json.loads((output / 'acceptance.json').read_text())
            if acceptance['status'] != 'passed':
                raise RuntimeError(f'A05 acceptance failed: {output}')
            trial.update(state='accepted', output_dir=str(output),
                         metrics=json.loads((output / 'metrics.json').read_text()))
            save()
        status['state'] = 'completed_formal_test_acceptance'
    except Exception as error:
        status.update(state='failed_stopped', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
