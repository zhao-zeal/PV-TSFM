"""Single-seed A01 dynamic soft-gate development on train/validation only."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/A01_site0_dynamic_gate_seed42_20261014'


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def setup(device, gate_mode='dynamic', fixed_gate_input=None, protocol='site0_v1'):
    import numpy as np
    import torch
    from run import parse_args
    from models.Chronos2DynamicGate import Model
    from layers.chronos2_adapter import ForecastOnlyBottleneckAdapter
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    args = parse_args(['--model', 'Chronos2Adapter', '--adapter_version', 'b131_linear_v1',
                      '--protocol', protocol, '--seed', '42', '--num_workers', '0',
                      '--is_training', '1', '--gpu', '0' if device == 'cpu' else device.split(':')[-1],
                      '--use_gpu', '0' if device == 'cpu' else '1'])
    args.gate_mode = gate_mode
    args.fixed_gate_input = fixed_gate_input
    model = Model(args).to(device)
    # Match A01's explicit B131 initialization seed; no old fitted weights reused.
    torch.manual_seed(42)
    model.adapter = ForecastOnlyBottleneckAdapter().to(device)
    return args, model


def raw_batch(batch, device):
    x, y, _, _, cov = batch
    return (x[..., 0].to(device), cov['historical_nwp'].to(device),
            cov['future_nwp'].to(device), y[:, -24:].to(device))


def check(device, gate_mode='dynamic', protocol='site0_v1'):
    import torch
    from torch.utils.data import default_collate, DataLoader
    from data_provider.data_factory import data_provider
    from models.Chronos2Adapter import Model as Native
    args, model = setup(device, gate_mode, protocol=protocol)
    dataset, _ = data_provider(args, 'train')
    fixed_input = None
    if gate_mode == 'fixed':
        total = torch.zeros(64, dtype=torch.float64)
        count = 0
        for batch in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(42)):
            p, h, f, _ = raw_batch(batch, 'cpu')
            inputs = model.gate_inputs(p, h, f).double()
            total += inputs.sum((0, 1))
            count += inputs.shape[0] * inputs.shape[1]
        fixed_input = (total / count).float().tolist()
        model.fixed_gate_input.copy_(torch.tensor(fixed_input, device=device))
    indices = [32, 56] if protocol == 'site0_v1' else [32, len(dataset) - 1]
    batch = default_collate([dataset[i] for i in indices])
    power, history, future, target = raw_batch(batch, device)
    model.eval()
    with torch.no_grad():
        native = Native.encode_inputs(model, power, history, future)
        opened = model.encode_inputs(power, history, future, torch.ones_like(future))
        closed = model.encode_inputs(power, history, future, torch.zeros_like(future))
        error = max((a-b).abs().max().item() for a, b in zip(native, opened))
        changed = (opened[0]-closed[0]).abs().max().item()
    assert error < 1e-6 and changed > 1e-7
    model.stage = 'base'
    model.train()
    prediction = model.forward_features(*model.encode_inputs(power, history, future))
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    grads = [p.grad for p in model.selector.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    assert prediction.shape == target.shape == (2, 24, 1) and torch.isfinite(loss)
    fixed_invariance = None
    if gate_mode == 'fixed':
        # Check after a parameter perturbation: initial constant output alone
        # would not distinguish this control from the dynamic model.
        with torch.no_grad():
            model.selector[-1].weight.add_(.01)
            g = model.gates(power, history, future)
            altered = model.gates(power + 1, history * 2, future - 1)
            fixed_invariance = bool(torch.equal(g, altered) and
                                    torch.equal(g, g[:1, :1].expand_as(g)))
        assert fixed_invariance
    record = dict(status='passed_real_train_batch_forward_backward', device=device,
        protocol=protocol, training_windows=len(dataset),
        training_indices=indices, training_site_ids=batch[4]['site_id'].tolist(),
        all_one_gate_native_max_abs=error,
        zero_gate_feature_max_abs_difference=changed, loss=float(loss.detach()),
        gate_gradient_l1=sum(g.abs().sum().item() for g in grads), backbone_frozen=True,
        gate_shape=list(model.gates(power, history, future).shape),
        selector_parameters=sum(p.numel() for p in model.selector.parameters()),
        gate_mode=gate_mode, fixed_gate_input=fixed_input,
        fixed_gate_invariance=fixed_invariance,
        formal_training_complete=False, test_accessed=False)
    save(REPORT / ('preflight_cpu.json' if device == 'cpu' else 'preflight_gpu.json'), record)
    print(json.dumps(record), flush=True)
    return record


def train(gpu, status, run_id, selector_lr, gate_mode='dynamic', protocol='site0_v1'):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from data_provider.data_factory import data_provider
    from layers.chronos2_adapter import residual_readout
    from utils.tools import EarlyStopping
    from utils.metrics import metric

    device = f'cuda:{gpu}'
    preflight = check(device, gate_mode, protocol)
    args, model = setup(device, gate_mode, preflight['fixed_gate_input'], protocol)
    output = ROOT / 'outputs' / protocol / run_id
    checkpoints = ROOT / 'checkpoints' / protocol / run_id
    output.mkdir(exist_ok=False)
    checkpoints.mkdir(exist_ok=False)
    args.experiment_label = 'A01_dynamic_soft_gate_development'
    config = dict(vars(args), selector_lr=selector_lr, microbatch=8, effective_batch=512,
                  eval_microbatch=8, test_policy='No test access; independent holdout not yet specified',
                  gate_location='Normalized values before native patch embedding; time and masks unchanged',
                  history_gate='Mean of future horizon gates; target power gate fixed at1',
                  gate_initial_value=.95, selector_hidden=32,
                  stages='Train selector+B131; then freeze both and train Linear',
                  selection='validation MAE, matching A01; report RMSE for comparison',
                  limitation='Zero gate is not exact covariate deletion; soft value weighting only')
    save(output / 'config.json', config)
    datasets = {split: data_provider(args, split)[0] for split in ('train', 'val')}
    datasets['train'].save_protocol(output)
    loaders = {split: DataLoader(ds, batch_size=8, shuffle=split == 'train', num_workers=0)
               for split, ds in datasets.items()}
    status.update(state='training_selector_b131', gpu=gpu, output_dir=str(output), checkpoint_dir=str(checkpoints))
    save(REPORT / 'runs.json', status)

    @torch.no_grad()
    def validation(loader, cached=False):
        model.eval()
        total_abs = total_square = count = 0
        for batch in loader:
            if cached:
                emb, loc, scale, target = [x.to(device) for x in batch]
                pred = model.forward_features(emb, loc, scale)
            else:
                power, history, future, target = raw_batch(batch, device)
                pred = model.forward_features(*model.encode_inputs(power, history, future))
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite validation predictions')
            error = (pred-target).double()
            total_abs += error.abs().sum().item()
            total_square += error.square().sum().item()
            count += error.numel()
        return dict(mae=total_abs/count, rmse=(total_square/count)**.5)

    def snapshot(epoch, metrics):
        return dict(epoch=epoch, validation_metrics=metrics, config=config,
                    adapter_state_dict=model.adapter.state_dict(),
                    selector_state_dict=model.selector.state_dict(),
                    readout_state_dict=model.readout.state_dict())

    model.readout.requires_grad_(False)
    optimizer = torch.optim.AdamW([{'params': model.adapter.parameters(), 'lr': .0016},
                                  {'params': model.selector.parameters(), 'lr': selector_lr}], weight_decay=.05)
    stopping = EarlyStopping(10)
    history_rows = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cursor = group_start = 0
        total_loss = 0.
        for batch in loaders['train']:
            power, history, future, target = raw_batch(batch, device)
            pred = model.forward_features(*model.encode_inputs(power, history, future))
            loss = torch.nn.functional.l1_loss(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite gate/B131 training loss')
            group_size = min(512, len(datasets['train']) - group_start)
            (loss * len(target) / group_size).backward()
            total_loss += loss.item()*len(target)
            cursor += len(target)
            if cursor-group_start == group_size:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_start = cursor
        val = validation(loaders['val'])
        improved, stopped = stopping.update(val['mae'])
        history_rows.append(dict(epoch=epoch, train_mae=total_loss/cursor, **val))
        if improved:
            torch.save(snapshot(epoch, val), checkpoints / 'selector_b131.pth')
        save(output / 'selector_b131_history.json', history_rows)
        logging.info('selector+B131 epoch=%s validation=%s', epoch, val)
        status.update(epoch=epoch, validation=val)
        save(REPORT / 'runs.json', status)
        if stopped:
            break
    selected = torch.load(checkpoints / 'selector_b131.pth', map_location=device, weights_only=True)
    status.update(selector_completed_epochs=epoch, best_selector_epoch=selected['epoch'])
    model.adapter.load_state_dict(selected['adapter_state_dict'])
    model.selector.load_state_dict(selected['selector_state_dict'])
    model.adapter.requires_grad_(False)
    model.selector.requires_grad_(False)
    model.eval()
    # Caching is valid only now, after selector and B131 have been frozen.
    cached = {}
    for split, ds in datasets.items():
        values = [[], [], [], []]
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
                power, history, future, target = raw_batch(batch, device)
                for dest, tensor in zip(values, (*model.encode_inputs(power, history, future), target)):
                    dest.append(tensor.cpu())
        cached[split] = TensorDataset(*[torch.cat(x) for x in values])
    cached_loaders = {s: DataLoader(ds, batch_size=512 if s == 'train' else 1024,
                                   shuffle=s == 'train', num_workers=0) for s, ds in cached.items()}
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model.readout = residual_readout('b131_linear_v1').to(device)
    model.stage = 'b131_linear_v1'
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=.000016, weight_decay=.05)
    stopping = EarlyStopping(10)
    history_rows = []
    status.update(state='training_linear', epoch=0)
    save(REPORT / 'runs.json', status)
    for epoch in range(1, 101):
        model.train()
        total_loss = samples = 0
        for batch in cached_loaders['train']:
            emb, loc, scale, target = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model.forward_features(emb, loc, scale), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Linear training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(target)
            samples += len(target)
        val = validation(cached_loaders['val'], cached=True)
        improved, stopped = stopping.update(val['mae'])
        history_rows.append(dict(epoch=epoch, train_mae=total_loss/samples, **val))
        if improved:
            torch.save(snapshot(epoch, val), checkpoints / 'checkpoint.pth')
        save(output / 'linear_history.json', history_rows)
        logging.info('Linear epoch=%s validation=%s', epoch, val)
        status.update(epoch=epoch, validation=val)
        save(REPORT / 'runs.json', status)
        if stopped:
            break
    selected = torch.load(checkpoints / 'checkpoint.pth', map_location=device, weights_only=True)
    model.readout.load_state_dict(selected['readout_state_dict'])
    predictions, labels, gates, sites, times = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loaders['val']:
            power, history, future, target = raw_batch(batch, device)
            w = model.gates(power, history, future)
            predictions.append(model.forward_features(*model.encode_inputs(power, history, future, w)).cpu().numpy())
            labels.append(target.cpu().numpy())
            gates.append(w.cpu().numpy())
            sites.append(batch[4]['site_id'].numpy())
            times.append(batch[4]['forecast_timestamps'].numpy())
    arrays = {key: np.concatenate(value) for key, value in dict(pred=predictions, true=labels,
               gates=gates, site_ids=sites, forecast_timestamps=times).items()}
    valdir = output / 'validation'
    valdir.mkdir()
    for key, value in arrays.items():
        np.save(valdir / f'{key}.npy', value)
    metrics = metric(arrays['pred'], arrays['true'])
    assert abs(metrics['rmse']-selected['validation_metrics']['rmse']) < 1e-6
    save(valdir / 'metrics.json', dict(metrics, split='validation'))
    status.update(state='completed_validation_only', metrics=metrics,
                  linear_completed_epochs=epoch, best_linear_epoch=selected['epoch'], test_accessed=False,
                  next_step='Review validation; independent final evaluation period remains unspecified')
    save(REPORT / 'runs.json', status)


def main():
    global REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='Chronos2DynamicGate_A01_site0_seed42_20261014')
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    parser.add_argument('--selector-lr', type=float, default=.000016)
    parser.add_argument('--gpu-index', type=int, choices=range(8), default=None)
    parser.add_argument('--protocol', choices=['site0_v1', 'in_domain', 'zeroshot_v1'], default='site0_v1')
    parser.add_argument('--gate-mode', choices=['dynamic', 'fixed'], default='dynamic')
    parser.add_argument('--check-cpu', action='store_true')
    parser.add_argument('--resume-waiting', action='store_true',
                        help='Restart only a manually paused queue that never began training')
    options = parser.parse_args()
    REPORT = options.report_dir.resolve()
    REPORT.mkdir(parents=True, exist_ok=True)
    if options.protocol != 'site0_v1' and (REPORT == ROOT / 'reports/A01_site0_dynamic_gate_seed42_20261014' or 'site0' in options.run_id):
        parser.error('Non-Site0 protocols require independent --report-dir and --run-id')
    if options.check_cpu:
        check('cpu', options.gate_mode, options.protocol)
        return
    path = REPORT / 'runs.json'
    previous = None
    if path.exists():
        previous = json.loads(path.read_text())
        if not options.resume_waiting or previous['state'] != 'paused_before_training':
            raise FileExistsError('Inspect existing run; only --resume-waiting of paused_before_training is supported')
        if previous.get('output_dir') or previous.get('checkpoint_dir'):
            raise ValueError('Training already began; cannot restart as an unstarted queue')
    status = dict(state='waiting_for_idle_gpu', pid=os.getpid(), seed=42, protocol=options.protocol,
                  scope='Single dynamic-gate candidate, train/validation only, 100/10 per stage',
                  selector_lr=options.selector_lr, gate_mode=options.gate_mode,
                  run_id=options.run_id, test_accessed=False)
    if previous is not None:
        status['previous_wait'] = previous
    save(path, status)
    logging.basicConfig(filename=REPORT / 'run.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')
    try:
        from utils.resources import require_idle_gpus
        while True:
            gpu = None
            for index in (range(8) if options.gpu_index is None else [options.gpu_index]):
                try:
                    require_idle_gpus(str(index))
                    gpu = index
                    break
                except ValueError:
                    pass
            if gpu is not None:
                break
            time.sleep(60)
        require_idle_gpus(str(gpu))
        train(gpu, status, options.run_id, options.selector_lr, options.gate_mode, options.protocol)
    except Exception as error:
        status.update(state='failed_stopped', error=str(error))
        save(path, status)
        logging.exception('Dynamic gate development failed')
        raise


if __name__ == '__main__':
    main()
