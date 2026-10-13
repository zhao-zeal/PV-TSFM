"""Run and accept the two authorized ChronosX protocols on one idle GPU."""

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now().astimezone().isoformat()


def write(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def monitor_gpu(gpu, path, stop):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'gpu_utilization_percent', 'memory_mib'])
        while not stop.is_set():
            row = subprocess.check_output([
                'nvidia-smi', f'--id={gpu}', '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'], text=True).strip().split(',')
            writer.writerow([now(), *[int(x) for x in row]])
            handle.flush()
            stop.wait(5)


def wait_for_gpu(report, record, preferred, stage):
    while True:
        devices = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        occupied = set(subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
            text=True).splitlines())
        rows = [line.split(',') for line in devices.splitlines()]
        candidates = []
        snapshot = []
        for index, uuid, memory, utilization in rows:
            index, memory, utilization = int(index), int(memory), int(utilization)
            busy = uuid.strip() in occupied
            snapshot.append(dict(gpu=index, memory_mib=memory, utilization_percent=utilization,
                                 has_compute_process=busy))
            if not busy and memory <= 1024 and utilization == 0:
                candidates.append(index)
        record.update(state='waiting_for_idle_gpu', current_blockers=['Waiting for a completely idle GPU'], waiting_stage=stage,
                      last_gpu_check_at=now(), gpu_snapshot=snapshot)
        write(report, record)
        eligible = candidates if preferred < 0 else [gpu for gpu in candidates if gpu == preferred]
        if eligible:
            selected = eligible[0]
            from utils.resources import require_idle_gpus
            try:
                require_idle_gpus(str(selected))
            except ValueError:
                continue
            record.update(gpu=selected, current_blockers=[], waiting_stage=None, selected_gpu_at=now())
            write(report, record)
            print(f'{now()} selected idle GPU{selected} for {stage}', flush=True)
            return selected
        print(f'{now()} waiting for a fully idle GPU: {stage}', flush=True)
        threading.Event().wait(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=-1, help='-1: wait for any fully idle GPU')
    options = parser.parse_args()
    report = options.report.resolve()
    record = json.loads(report.read_text())
    if record.get('acceptance', {}).get('status') == 'passed_full_artifact_acceptance':
        return
    os.sched_setaffinity(0, record['cpu_cores'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(record['threads'])
    lock = (report.parent / 'chronosx.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        record.update(coordinator_pid=os.getpid(), coordinator_started_at=now())
        write(report, record)
        for protocol in [item['protocol'] for item in record['runs']]:
            run = next(item for item in record['runs'] if item['protocol'] == protocol)
            if run['state'] == 'complete_pending_acceptance':
                continue
            if run['state'] != 'pending' or Path(run['output_dir']).exists() or Path(run['checkpoint_dir']).exists():
                raise RuntimeError(f'Refusing to repeat {run["protocol"]} {run["model_id"]}')
            if record.get('preflight', {}).get('config', {}).get('precision') != 'bf16':
                gpu = wait_for_gpu(report, record, options.gpu, 'ChronosX BF16 GPU preflight')
                record['previous_preflight'] = record.get('preflight')
                write(report, record)
                subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/check_budget_models.py'),
                                '--report', str(report), '--gpu', str(gpu)], cwd=ROOT, check=True)
                record = json.loads(report.read_text())
                run = next(item for item in record['runs'] if item['protocol'] == run['protocol'])
            gpu = wait_for_gpu(report, record, options.gpu, f'{run["protocol"]} train/test')
            argv = list(run['argv'])
            argv[argv.index('--gpu') + 1] = str(gpu)
            run.update(argv=argv, gpu=gpu, state='training', started_at=now())
            record['state'] = 'running'
            utilization = report.parent / f'{run["protocol"]}_gpu_utilization.csv'
            run['gpu_utilization_log'] = str(utilization)
            stop = threading.Event()
            monitor = threading.Thread(target=monitor_gpu, args=(gpu, utilization, stop))
            with Path(run['log']).open('w') as handle:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run.py'), *argv],
                                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                run['pid'] = process.pid
                write(report, record)
                monitor.start()
                try:
                    returncode = process.wait()
                finally:
                    stop.set()
                    monitor.join()
            run.update(returncode=returncode, finished_at=now(),
                       state='complete_pending_acceptance' if returncode == 0 else 'failed')
            write(report, record)
            if returncode:
                raise RuntimeError(f'ChronosX {run["protocol"]} exited {returncode}')
        gpu = wait_for_gpu(report, record, options.gpu, 'ChronosX full artifact acceptance')
        record.update(state='accepting_artifacts', acceptance_gpu=gpu)
        write(report, record)
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/accept_chronosx.py'),
                        '--report', str(report), '--gpu', str(gpu)], cwd=ROOT, check=True)
    except Exception as error:
        record.update(state='failed_stopped', error=str(error), stopped_at=now())
        write(report, record)
        raise


if __name__ == '__main__':
    main()
