"""Real-batch correctness and throughput checks for the authorized TimeVLM/ChronosX queue."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, [9, 10])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    from run import parse_args, configure_resources
    record = json.loads(options.report.read_text())
    run = record['runs'][0]
    args = parse_args(run['argv'] + ['--gpu', str(options.gpu)])
    configure_resources(args)
    import numpy as np
    import torch
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.tools import load_model_checkpoint
    from utils.losses import forecasting_loss
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    experiment = Exp_Long_Term_Forecast(args)
    train_data, train_loader = experiment._get_data('train')
    val_data, val_loader = experiment._get_data('val')
    assert len(train_data) == 76570 and len(val_data) == 25450
    optimizer = experiment._select_optimizer()
    evidence = dict(status='passed_real_batch_forward_backward_and_prediction',
                    gpu=options.gpu, cpu_cores=[9, 10], performance_reproduction_complete=False)
    if args.resume:
        checkpoint = load_model_checkpoint(experiment.model, args.resume, experiment.device)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler = experiment._select_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        evidence.update(resume_epoch=checkpoint['epoch'], resumed_lr=optimizer.param_groups[0]['lr'],
                        scheduler_epoch=scheduler.last_epoch, model_and_memory_strict_load=True,
                        old_rng_and_amp_scaler_not_saved=True)
        assert evidence['resume_epoch'] == evidence['scheduler_epoch'] == 9
        assert evidence['resumed_lr'] == 3.90625e-6
    iterator = iter(train_loader)
    train_seconds = []
    for _ in range(3):
        experiment.model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        batch = next(iterator)
        if args.model == 'ChronosX':
            x, _, _, _, cov, target = experiment._prepare_batch(batch)
            with experiment._autocast():
                loss = experiment.model.token_loss(x, target, cov, augment=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(experiment.model.parameters(), 1.)
            optimizer.step()
        else:
            scaler = torch.amp.GradScaler('cuda')
            output, target, _ = experiment._forward_batch(batch)
            loss = forecasting_loss(output, target, args.model)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all(torch.isfinite(p.grad).all() for p in experiment.model.parameters() if p.grad is not None)
            scaler.step(optimizer)
            scaler.update()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in experiment.model.parameters() if p.grad is not None)
        torch.cuda.synchronize()
        train_seconds.append(time.perf_counter() - begin)
    experiment.model.eval()
    validation_seconds = []
    iterator = iter(val_loader)
    with torch.no_grad():
        for _ in range(3):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            batch = next(iterator)
            if args.model == 'ChronosX':
                x, _, _, _, cov, target = experiment._prepare_batch(batch)
                with experiment._autocast():
                    validation = experiment.model.token_loss(x, target, cov)
                assert torch.isfinite(validation)
            else:
                prediction, target, _ = experiment._forward_batch(batch)
                assert torch.isfinite(prediction).all()
            torch.cuda.synchronize()
            validation_seconds.append(time.perf_counter() - begin)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        prediction, target, _ = experiment._forward_batch(batch)
        torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - begin
    if isinstance(prediction, dict):
        prediction = prediction['prediction']
    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
    evidence.update(train_microbatch_seconds=train_seconds, validation_batch_seconds=validation_seconds,
                    prediction_batch_seconds=prediction_seconds, prediction_shape=list(prediction.shape),
                    loss=float(loss.detach()), config=vars(args))
    if args.model == 'ChronosX':
        parameters = list(experiment.model.named_parameters())
        assert all(not p.requires_grad for name, p in parameters if 'injection_block' not in name)
        assert any(p.grad is not None for name, p in parameters if 'injection_block' in name)
        evidence.update(frozen_t5_verified=True, trainable_parameters=sum(p.numel() for _, p in parameters if p.requires_grad),
                        total_parameters=sum(p.numel() for _, p in parameters),
                        samples_per_window=args.num_samples, historical_future_nwp_shape=list(batch[4]['historical_nwp'].shape))
    record['preflight'] = evidence
    options.report.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
