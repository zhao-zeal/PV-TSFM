"""Authorized four-run gate queue; validation only, stop on any failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/A01_gate_two_protocols_seed42_20261015'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def accept(run, device='cpu'):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from data_provider.data_factory import data_provider
    from scripts.run_a01_dynamic_gate import setup, raw_batch
    from utils.metrics import metric
    directory = ROOT / run['report_dir']
    status = json.loads((directory / 'runs.json').read_text())
    assert status['state'] == 'completed_validation_only' and status['test_accessed'] is False
    output, checkpoint = Path(status['output_dir']), Path(status['checkpoint_dir'])
    arrays = {k: np.load(output / 'validation' / f'{k}.npy')
              for k in ('pred', 'true', 'gates', 'site_ids', 'forecast_timestamps')}
    assert arrays['pred'].shape == arrays['true'].shape == (25450, 24, 1)
    assert all(np.isfinite(v).all() for v in arrays.values())
    metrics = metric(arrays['pred'], arrays['true'])
    for key in ('mae', 'rmse', 'r2'):
        assert abs(metrics[key] - status['metrics'][key]) < 1e-10
    weights, stages = {}, {}
    for stage, file in [('selector_b131', 'selector_b131.pth'), ('linear', 'checkpoint.pth')]:
        rows = json.loads((output / f'{stage}_history.json').read_text())
        w = torch.load(checkpoint / file, map_location='cpu', weights_only=True)
        assert w['epoch'] == min(rows, key=lambda row: row['mae'])['epoch']
        assert len(rows) == 100 or len(rows) - w['epoch'] == 10
        assert all(np.isfinite(row[k]) for row in rows for k in ('mae', 'rmse', 'train_mae'))
        weights[stage] = w
        stages[stage] = dict(completed_epochs=len(rows), best_epoch=w['epoch'])
    for key in ('adapter_state_dict', 'selector_state_dict'):
        assert all(torch.equal(v, weights['linear'][key][k])
                   for k, v in weights['selector_b131'][key].items())
    config = json.loads((output / 'config.json').read_text())
    assert config == weights['linear']['config']
    assert config['protocol'] == run['protocol'] and config['selector_lr'] == .00016
    args, model = setup(device, run['gate_mode'], config['fixed_gate_input'], run['protocol'])
    for module, key in [('adapter', 'adapter_state_dict'), ('selector', 'selector_state_dict'),
                        ('readout', 'readout_state_dict')]:
        getattr(model, module).load_state_dict(weights['linear'][key])
    model.stage = 'b131_linear_v1'
    model.eval()
    dataset, _ = data_provider(args, 'val')
    # All validation labels, sites and times checked without accessing test data.
    for i in range(len(dataset)):
        sample = dataset[i]
        assert np.array_equal(sample[1][-24:].numpy(), arrays['true'][i])
        assert np.array_equal(sample[4]['site_id'].numpy(), arrays['site_ids'][i])
        assert np.array_equal(sample[4]['forecast_timestamps'].numpy(), arrays['forecast_timestamps'][i])
    indices = [32, len(dataset) - 1]
    batch_size = config['eval_microbatch']
    starts = sorted({(i // batch_size) * batch_size for i in indices})
    replay_batches = [list(range(start, min(start + batch_size, len(dataset)))) for start in starts]
    error = gate_error = 0.
    with torch.no_grad():
        for batch_indices in replay_batches:
            power, history, future, target = raw_batch(
                default_collate([dataset[i] for i in batch_indices]), device)
            gates = model.gates(power, history, future)
            pred = model.forward_features(*model.encode_inputs(power, history, future, gates)).cpu().numpy()
            error = max(error, float(np.max(np.abs(pred - arrays['pred'][batch_indices]))))
            gate_error = max(gate_error, float(np.max(np.abs(gates.cpu().numpy() - arrays['gates'][batch_indices]))))
    assert error < 1e-5
    assert gate_error < 1e-6
    if run['gate_mode'] == 'fixed':
        assert np.array_equal(arrays['gates'], np.broadcast_to(arrays['gates'][:1, :1], arrays['gates'].shape))
    result = dict(status='passed_validation_artifact_acceptance', metrics=metrics,
                  stages=stages, all_validation_labels_sites_times_aligned=True,
                  selector_b131_frozen_in_stage2=True, checkpoint_replay_indices=indices,
                  replay_batches=replay_batches, replay_device=device,
                  replay_precision='fp32', prediction_tolerance=1e-5, gate_tolerance=1e-6,
                  torch_version=torch.__version__,
                  matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  prediction_max_abs_error=error, gate_max_abs_error=gate_error,
                  test_accessed=False, limitation='Single-seed validation only; no target-test generalization claim')
    save(directory / 'validation_review.json', result)
    status['acceptance'] = dict(status=result['status'], evidence=str(directory / 'validation_review.json'))
    save(directory / 'runs.json', status)
    return result


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def worker(protocol, gpu, cores):
    import time
    from utils.resources import require_idle_gpus
    os.sched_setaffinity(0, set(map(int, cores.split(','))))
    path = REPORT / protocol / 'queue.json'
    status = json.loads(path.read_text())
    assert status['state'] == 'approved_resume_remaining'
    status.update(state='running', coordinator_pid=os.getpid())
    save(path, status)
    def idle_card():
        while True:
            candidates = [gpu] if gpu >= 0 else [i for i in range(8) if i != 6]
            for candidate in candidates:
                try:
                    require_idle_gpus(str(candidate))
                    return candidate
                except ValueError:
                    pass
            time.sleep(60)
    try:
        for run in status['runs']:
            directory = ROOT / run['report_dir']
            if run['state'] == 'accepted_formal_test':
                receipt = json.loads((directory / 'test_acceptance.json').read_text())
                assert receipt['status'] == 'passed_formal_test_artifact_acceptance'
                continue
            if (REPORT / 'STOP').exists():
                raise RuntimeError('Other protocol failed; do not start another run')
            assert run['gate_mode'] == 'dynamic' and run['state'] == 'prepared_not_started'
            assert not (directory / 'runs.json').exists()
            for category in ('outputs', 'checkpoints'):
                assert not (ROOT / category / protocol / run['run_id']).exists()
            selected_gpu = idle_card()
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_a01_dynamic_gate.py'),
                       '--protocol', protocol, '--gate-mode', run['gate_mode'],
                       '--gpu-index', str(selected_gpu), '--selector-lr', '.00016',
                       '--run-id', run['run_id'], '--report-dir', str(directory)]
            with (directory / 'launcher.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                run.update(state='running_or_waiting_for_idle_gpu', pid=child.pid, gpu=selected_gpu, command=command)
                save(path, status)
                code = child.wait()
            if code:
                raise RuntimeError(f'{protocol}/dynamic exited {code}')
            for phase in ('validation_acceptance', 'formal_test'):
                selected_gpu = idle_card()
                run['state'] = phase
                save(path, status)
                command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_a01_gate.py'),
                           '--protocol', protocol, '--mode', 'dynamic', '--gpu', str(selected_gpu)]
                if phase == 'validation_acceptance':
                    command.append('--validate-only')
                with (directory / (phase + '_20261016.log')).open('x') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            run['test_acceptance'] = json.loads((directory / 'test_acceptance.json').read_text())
            run['state'] = 'accepted_formal_test'
            save(path, status)
        status.update(state='completed_formal_test_acceptance', finished_at=datetime.now().isoformat())
    except Exception as error:
        status.update(state='failed_stopped', error=repr(error))
        (REPORT / 'STOP').touch()
        raise
    finally:
        save(path, status)


def main():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['in_domain', 'zeroshot_v1'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cores')
    options = parser.parse_args()
    os.chdir(ROOT)
    if options.worker:
        return worker(options.worker, options.gpu, options.cores)
    path = REPORT / 'runs.json'
    status = json.loads(path.read_text())
    assert status['state'] == 'approved_resume_after_fixed_test'
    status.update(state='running_parallel', coordinator_pid=os.getpid(), training_started=True)
    save(path, status)
    children = []
    for protocol, gpu, cores in [('in_domain', 6, '9,10'), ('zeroshot_v1', -1, '0,7')]:
        with (REPORT / protocol / 'coordinator_resume_20261016.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--worker', protocol,
                                      '--gpu', str(gpu), '--cores', cores], stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
    while True:
        queues = {p: json.loads((REPORT / p / 'queue.json').read_text())
                  for p in ('in_domain', 'zeroshot_v1')}
        status['protocol_queues'] = {p: dict(state=q['state'], path=str(REPORT / p / 'queue.json'))
                                     for p, q in queues.items()}
        status['runs'] = [r for q in queues.values() for r in q['runs']]
        if any(child.poll() not in (None, 0) for child in children):
            (REPORT / 'STOP').touch()
            status['state'] = 'failed_no_new_runs_other_active_run_may_finish'
        if all(child.poll() is not None for child in children):
            status['state'] = ('completed_formal_test_acceptance' if all(child.returncode == 0 for child in children)
                               else 'failed_stopped')
            if status['state'] == 'completed_formal_test_acceptance':
                from scripts.evaluate_a01_gate import sync_table
                sync_table('dynamic')
            save(path, status)
            break
        save(path, status)
        time.sleep(60)


if __name__ == '__main__':
    main()
