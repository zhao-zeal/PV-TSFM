"""Bounded real-step batch comparison on GPU4; never updates formal experiment weights."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/FusionSF_inputs_two_protocols_seed42_20261011'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=8.)
    options = parser.parse_args()
    a02 = json.loads((ROOT / 'reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012/runs.json').read_text())
    assert a02['acceptance']['status'] == 'passed_full_training_test_artifact_acceptance'
    os.sched_setaffinity(0, [0, 7])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    import torch
    from data_provider.data_factory import data_provider
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from run import configure_resources, parse_args
    from utils.losses import forecasting_loss
    from torch.utils.data import DataLoader

    record = json.loads((REPORT / 'runs.json').read_text())
    selected_runs = [run for run in record['runs'] if run['protocol'] == 'in_domain']
    base_args = parse_args(selected_runs[0]['argv'] + ['--gpu', str(options.gpu)])
    configure_resources(base_args)
    device = torch.device(f'cuda:{options.gpu}')
    evidence = dict(diagnostic_only=True, formal_training_or_test=False,
        gpu=options.gpu, cpu_cores=[0, 7], threads=2, num_workers=0,
        created_at=datetime.now().astimezone().isoformat(), seed=42, precision='fp32',
        candidates=[16, 64, 256], measurement_seconds=options.seconds,
        selection='maximum measured end-to-end training windows/second among feasible batches',
        sampling='nvidia-smi every 0.5s during measured real data/forward/backward/AdamW steps',
        runs=[])
    evidence_path = REPORT / 'batch_throughput_20261012.json'
    for run in selected_runs:
        torch.manual_seed(42)
        args = parse_args(run['argv'] + ['--gpu', str(options.gpu)])
        experiment = Exp_Long_Term_Forecast(args)
        source = Path(run['checkpoint_dir']) / 'checkpoint.pth'
        source_checkpoint = torch.load(source, map_location='cpu', weights_only=True) if source.exists() else None
        if source_checkpoint is not None:
            experiment.model.load_state_dict(source_checkpoint['model_state_dict'], strict=True)
        original = {key: value.detach().cpu().clone() for key, value in experiment.model.state_dict().items()}
        dataset, _ = data_provider(args, 'train')
        variants = []
        for batch_size in evidence['candidates']:
            experiment.model.load_state_dict(original, strict=True)
            torch.manual_seed(42)
            experiment.model.train()
            optimizer = experiment._select_optimizer()
            if source_checkpoint is not None:
                optimizer.load_state_dict(source_checkpoint['optim_state_dict'])
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(42))
            iterator = iter(loader)
            samples, stop, sampler = [], threading.Event(), None
            batch = result = targets = loss = None

            def sample_utilization():
                while not stop.is_set():
                    try:
                        values = subprocess.check_output(['nvidia-smi', f'--id={options.gpu}',
                            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                        memory, utilization = [int(value.strip()) for value in values.strip().split(',')]
                        samples.append(dict(time=time.time(), memory_mib=memory, utilization_percent=utilization))
                    except subprocess.CalledProcessError:
                        pass
                    stop.wait(0.5)

            def step():
                nonlocal batch, result, targets, loss
                batch = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                result, targets, _ = experiment._forward_batch(batch)
                loss = forecasting_loss(result, targets, 'FusionSF')
                loss.backward()
                optimizer.step()
                return len(targets)

            try:
                for _ in range(3):
                    step()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(parameter.grad).all() for parameter in experiment.model.parameters()
                           if parameter.grad is not None)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                sampler = threading.Thread(target=sample_utilization, daemon=True)
                sampler.start()
                start = time.perf_counter()
                windows = steps = 0
                while time.perf_counter() - start < options.seconds:
                    windows += step()
                    steps += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                stop.set()
                sampler.join()
                assert torch.isfinite(loss)
                item = dict(batch_size=batch_size, status='passed_real_training_steps',
                    steps=steps, windows=windows, seconds=elapsed, windows_per_second=windows / elapsed,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    mean_utilization_percent=sum(row['utilization_percent'] for row in samples) / len(samples),
                    peak_utilization_percent=max(row['utilization_percent'] for row in samples),
                    utilization_samples=samples, final_diagnostic_loss=float(loss.detach()))
                variants.append(item)
                print(run['variant'], batch_size, f'{item["windows_per_second"]:.2f} windows/s',
                      f'{item["mean_utilization_percent"]:.2f}% GPU', flush=True)
            except torch.OutOfMemoryError as error:
                stop.set()
                if sampler is not None:
                    sampler.join()
                variants.append(dict(batch_size=batch_size, status='capacity_rejected', error=str(error)))
                print(run['variant'], batch_size, 'capacity rejected', flush=True)
            finally:
                del optimizer, iterator, loader, batch, result, targets, loss
                torch.cuda.empty_cache()
        best = max((item for item in variants if item['status'] == 'passed_real_training_steps'),
                   key=lambda item: item['windows_per_second'])
        experiment.model.eval()
        eval_batch_size = 64
        while True:
            batch = result = targets = None
            try:
                batch = next(iter(DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)))
                with torch.no_grad():
                    result, targets, _ = experiment._forward_batch(batch)
                assert result['prediction'].shape == targets.shape == (eval_batch_size, 24, 1)
                assert torch.isfinite(result['prediction']).all()
                break
            except torch.OutOfMemoryError:
                if eval_batch_size == 16:
                    raise
                eval_batch_size = 16
            finally:
                del batch, result, targets
                torch.cuda.empty_cache()
        evidence['runs'].append(dict(variant=run['variant'], protocol='in_domain', candidates=variants,
            selected_batch_size=best['batch_size'], selected_eval_batch_size=eval_batch_size,
            eval_forward_finite=True, resume_source=str(source) if source_checkpoint is not None else None,
            source_epoch=source_checkpoint['epoch'] if source_checkpoint is not None else None,
            checkpoint_not_modified=True, training_trajectory_equivalent=False))
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
        del experiment, dataset, original, source_checkpoint
        torch.cuda.empty_cache()
    evidence['status'] = 'passed_three_inputs_real_steps_and_eval_capacity'
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(evidence_path, flush=True)


if __name__ == '__main__':
    main()
