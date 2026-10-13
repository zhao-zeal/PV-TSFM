"""GPU0: resume both TimeVLM protocols, accept, then reproduce both ChronosX protocols."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUEUE = ROOT / 'reports/TimeVLM_ChronosX_queue_20261011'
REPORTS = [ROOT / 'reports/TimeVLM_resume200_pat20_seed42_20261011/runs.json',
           ROOT / 'reports/ChronosX_two_protocols_seed42_20261011/runs.json']


def now():
    return datetime.now().astimezone().isoformat()


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def prepare():
    old = json.loads((ROOT / 'reports/TimeVLM_two_protocols_seed42_20261011/runs.json').read_text())
    for model, report in zip(('TimeVLM', 'ChronosX'), REPORTS):
        if report.exists():
            continue
        runs = []
        for protocol in ('in_domain', 'zeroshot_v1'):
            model_id = f'{model}_ep200_pat20_seed42_sl24_pl24_' + ('power_resume9' if model == 'TimeVLM' else 'history_future_IIBOIB')
            argv = ['--model', model, '--protocol', protocol, '--is_training', '1',
                    '--seed', '42', '--seq_len', '24', '--pred_len', '24',
                    '--train_epochs', '200', '--patience', '20', '--num_workers', '0',
                    '--use_gpu', '1', '--gpu', '0', '--model_id', model_id]
            run = dict(protocol=protocol, model_id=model_id, argv=argv, gpu=0,
                       cpu_cores=[9, 10], threads=2, num_workers=0, state='pending',
                       output_dir=str(ROOT / 'outputs' / protocol / model_id),
                       checkpoint_dir=str(ROOT / 'checkpoints' / protocol / model_id),
                       log=str(report.parent / f'{protocol}_training.log'))
            if model == 'TimeVLM':
                source = next(x for x in old['runs'] if x['protocol'] == protocol)
                run.update(resume_epoch=9, source_log=source['log'], source_output_dir=source['output_dir'],
                           source_checkpoint=str(Path(source['checkpoint_dir']) / 'checkpoint.pth'))
                argv.extend(['--resume', run['source_checkpoint']])
            runs.append(run)
        record = dict(model=model, state='prepared_not_started', created_at=now(), gpu=0,
                      authorization='User: TimeVLM200/20 continuation, then ChronosX200/20 on the same GPU; preserve other existing runs',
                      train_epochs=200, patience=20, seed=42, seq_len=24, pred_len=24,
                      cpu_cores=[9, 10], threads=2, num_workers=0,
                      resume_training=model == 'TimeVLM', runs=runs)
        if model == 'TimeVLM':
            record.update(continuation_limit='Source contains best epoch9 model/optimizer/scheduler/memory; epoch10 terminal state and original RNG/AMP scaler unavailable; resume starts epoch10 with fresh RNG/scaler',
                          restored_lr=3.90625e-6, scheduler='original type1 retained; halve each epoch',
                          original_results_record=str(ROOT / 'reports/TimeVLM_two_protocols_seed42_20261011/runs.json'))
        else:
            record.update(input='historical power + train-standardized historical/future NWP; 15 weather + 15 missing indicators',
                          architecture='Frozen Chronos-T5-small, IIB+OIB; train injection blocks only; hidden256/layer1',
                          training='AdamW lr0.01 wd0; token CE; fp32; batch32/eval8; accumulation2; clip1; linear LR over epoch-derived optimizer steps; min validation CE; sampled20 median prediction',
                          official_budget_adaptation='Author fixed5000 steps/eval100 changed by user to max200 epochs/patience20; validate at each full epoch; max239400 steps at current batch/accumulation')
        write(report, record)
    QUEUE.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    os.sched_setaffinity(0, [9, 10])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    prepare()
    if options.prepare_only:
        return
    lock = (QUEUE / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state='running', pid=os.getpid(), gpu=0, cpu_cores=[9, 10], started_at=now(), stages=[str(x) for x in REPORTS])
    write(QUEUE / 'status.json', status)
    from scripts.run_fusionsf_inputs import idle_gpu
    try:
        for report in REPORTS:
            record = json.loads(report.read_text())
            if record.get('acceptance', {}).get('status') == 'passed_full_artifact_acceptance':
                continue
            if record.get('preflight', {}).get('status') != 'passed_real_batch_forward_backward_and_prediction':
                idle_gpu(0)
                subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/check_budget_models.py'),
                                '--report', str(report), '--gpu', '0'], cwd=ROOT, check=True)
            for index in range(2):
                record = json.loads(report.read_text())
                run = record['runs'][index]
                if run['state'] == 'complete_pending_acceptance':
                    continue
                resuming = run['state'] == 'resume_pending'
                if not resuming and (run['state'] != 'pending' or Path(run['output_dir']).exists() or Path(run['checkpoint_dir']).exists()):
                    raise RuntimeError(f'Refusing to repeat an existing run: {run["model_id"]} {run["protocol"]}')
                idle_gpu(0)
                run.update(state='training', started_at=now())
                record['state'] = 'running'
                with Path(run['log']).open('a' if resuming else 'w') as log:
                    process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run.py'), *run['argv']],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    run['pid'] = process.pid
                    write(report, record)
                    returncode = process.wait()
                run.update(returncode=returncode, state='complete_pending_acceptance' if returncode == 0 else 'failed')
                write(report, record)
                if returncode:
                    raise RuntimeError(f'{record["model"]} {run["protocol"]} exited {returncode}')
            idle_gpu(0)
            helper = 'accept_timevlm.py' if record['model'] == 'TimeVLM' else 'accept_chronosx.py'
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / helper), '--report', str(report), '--gpu', '0'], cwd=ROOT, check=True)
            record = json.loads(report.read_text())
            assert record['acceptance']['status'] == 'passed_full_artifact_acceptance'
        status.update(state='completed_artifact_acceptance_passed', completed_at=now())
        write(QUEUE / 'status.json', status)
    except Exception as error:
        status.update(state='failed_stopped', error=str(error), stopped_at=now())
        write(QUEUE / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
