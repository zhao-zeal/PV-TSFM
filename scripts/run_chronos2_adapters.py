"""Run the two explicitly approved adapter versions serially within one protocol."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['in_domain', 'zeroshot_v1'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    report = ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011' / args.protocol
    report.mkdir(parents=True, exist_ok=True)
    status = {'protocol': args.protocol, 'gpu': args.gpu, 'pid': os.getpid(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'seed': 42,
              'versions': [], 'status': 'running'}
    status_path = report / 'runs.json'

    def save():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')

    save()
    for version in ('b131_linear_v1', 'b131_mlp63_v1'):
        model_id = f'Chronos2Adapter_{version}_seed42_sl24_pl24_history_future'
        command = [sys.executable, str(ROOT / 'run.py'), '--model', 'Chronos2Adapter',
                   '--adapter_version', version, '--is_training', '1',
                   '--protocol', args.protocol, '--seed', '42', '--use_gpu', '1',
                   '--gpu', str(args.gpu), '--num_workers', '0', '--model_id', model_id]
        row = {'version': version, 'model_id': model_id, 'command': command,
               'checkpoint_dir': str(ROOT / 'checkpoints' / args.protocol / model_id),
               'output_dir': str(ROOT / 'outputs' / args.protocol / model_id), 'status': 'running'}
        status['versions'].append(row)
        save()
        with (report / f'{version}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            save()
            row['returncode'] = process.wait()
        if row['returncode']:
            row['status'] = status['status'] = 'failed'
            save()
            return row['returncode']
        row['status'] = 'complete_pending_acceptance'
        row['metrics'] = json.loads((Path(row['output_dir']) / 'metrics.json').read_text())
        save()
        if version == 'b131_linear_v1':
            # CUDA teardown and nvidia-smi utilization sampling lag process exit.
            time.sleep(3)
    status['status'] = 'complete_pending_acceptance'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
