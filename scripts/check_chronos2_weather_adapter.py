"""Real CPU/GPU microbatch checks, without updating any experiment checkpoint."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def state_digest(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output', help='Evidence JSON; defaults to preflight_cpu.json or preflight_gpu.json')
    opts = parser.parse_args()
    allowed = os.sched_getaffinity(0)
    cores = [9, 10] if {9, 10}.issubset(allowed) else sorted(allowed)[:2]
    os.sched_setaffinity(0, cores)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if opts.gpu is not None:
        from utils.resources import require_idle_gpus
        require_idle_gpus(str(opts.gpu))
    import numpy as np
    import torch
    from data_provider.data_factory import data_provider
    from models.Chronos2WeatherAdapter import Model
    from run import parse_args
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu' if opts.gpu is None else f'cuda:{opts.gpu}')
    evidence = {'stage': 'real_microbatch_correctness_check', 'device': str(device),
                'experiment_label': 'A02', 'base_experiment_label': 'A01',
                'cpu_cores': cores, 'threads': 2, 'num_workers': 0,
                'full_training_complete': False, 'formal_test_complete': False,
                'software': {name: importlib.metadata.version(name) for name in
                             ('torch', 'chronos-forecasting', 'transformers', 'numpy', 'pandas')},
                'runs': []}
    for protocol in ('in_domain', 'zeroshot_v1'):
        args = parse_args(['--model', 'Chronos2WeatherAdapter', '--protocol', protocol,
                           '--is_training', '1', '--use_gpu', str(int(opts.gpu is not None)),
                           '--gpu', str(opts.gpu or 0)])
        torch.manual_seed(42)
        model = Model(args).float().to(device)
        assert sum(p.numel() for p in model.weather_adapter.parameters()) == 29249
        assert all(not p.requires_grad for p in model.base.parameters())
        indices = [0, 1, 2545, 25449]
        dataset, _ = data_provider(args, 'test')
        samples = [dataset[index] for index in indices]
        power = torch.stack([s[0] for s in samples]).to(device)
        history = torch.stack([s[4]['historical_nwp'] for s in samples]).to(device)
        future = torch.stack([s[4]['future_nwp'] for s in samples]).to(device)
        targets = torch.stack([s[1][-24:] for s in samples]).to(device)
        cache = ROOT / 'outputs' / protocol / 'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future/test'
        embeds, loc, scale = [torch.from_numpy(np.load(cache / f'{name}.npy', mmap_mode='r')[indices].copy()).to(device)
                              for name in ('embeds', 'loc', 'scale')]
        for name, actual in (('targets', targets),
                             ('site_ids', torch.stack([s[4]['site_id'] for s in samples]).to(device)),
                             ('forecast_timestamps', torch.stack([s[4]['forecast_timestamps'] for s in samples]).to(device))):
            expected = torch.from_numpy(np.load(cache / f'{name}.npy', mmap_mode='r')[indices].copy()).to(device)
            assert torch.equal(actual, expected), name
        model.train()
        assert not any(m.training for m in model.base.modules())
        before = state_digest(model.base)
        with torch.no_grad():
            base = model.base.forward_features(embeds, loc, scale)
            prediction = model.forward_features(embeds, loc, scale, future)
            assert torch.equal(base, prediction)
            direct_features = model.base.encode_inputs(power[..., 0], history, future)
            direct = model.forward_features(*direct_features, future)
            direct_difference = float((direct - prediction).abs().max())
            saved = torch.from_numpy(np.load(ROOT / 'outputs' / protocol /
                'Chronos2Adapter_b131_linear_v1_seed42_sl24_pl24_history_future/pred.npy',
                mmap_mode='r')[indices].copy()).to(device)
            saved_difference = float((saved - prediction).abs().max())
            assert saved_difference <= 2e-6, saved_difference
        optimizer = torch.optim.AdamW(model.weather_adapter.parameters(), lr=args.learning_rate,
                                     weight_decay=0.05, betas=(0.9, 0.999))
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_features(embeds, loc, scale, future)
            loss = torch.nn.functional.l1_loss(prediction, targets)
            assert torch.isfinite(loss)
            loss.backward()
            grads = {name: float(p.grad.abs().sum()) for name, p in model.weather_adapter.named_parameters()
                     if p.grad is not None and torch.isfinite(p.grad).all()}
            assert len(grads) == len(list(model.weather_adapter.parameters()))
            assert all(p.grad is None for p in model.base.parameters())
            if step == 0:
                assert grads['concat_layer.2.weight'] > 0
            else:
                assert grads['hidden_state_out.1.weight'] > 0 and grads['cov_out.weight'] > 0
            steps.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_l1': grads})
            optimizer.step()
        assert state_digest(model.base) == before
        with torch.no_grad():
            adapted = model.base.adapter(embeds)
            constant = future[:, :1].expand(-1, 24, -1).clone()
            residual = model.weather_adapter(adapted, constant)
            assert torch.equal(residual[:, :1].expand(-1, 16, -1), residual[:, :16])
            assert torch.equal(residual[:, 16:17].expand(-1, 8, -1), residual[:, 16:])
            changed = future.clone()
            changed[:, 5, 0] += 10.
            delta = model.weather_adapter(adapted, changed) - model.weather_adapter(adapted, future)
            assert delta[:, 5].abs().max() > 0
            other_hours = delta.clone()
            other_hours[:, 5] = 0
            assert torch.count_nonzero(other_hours) == 0
            missing = future.clone()
            missing[:, 3, 0] = float('nan')
            assert torch.isfinite(model.weather_adapter(adapted, missing)).all()
            clone = type(model.weather_adapter)().to(device)
            clone.load_state_dict(model.weather_adapter.state_dict(), strict=True)
            assert torch.equal(clone(adapted, future), model.weather_adapter(adapted, future))
        evidence['runs'].append({
            'protocol': protocol, 'sample_indices': indices, 'batch_size': len(indices),
            'output_shape': list(prediction.shape), 'trainable_parameters': 29249,
            'zero_residual_initial_prediction_exact': True,
            'cached_vs_direct_prediction_max_abs_difference': direct_difference,
            'cached_vs_direct_prediction_matches': direct_difference <= 2e-6,
            'accepted_saved_prediction_max_abs_difference': saved_difference,
            'frozen_base_state_sha256_before_and_after': before,
            'base_gradients_absent_and_weights_unchanged': True,
            'all_weather_gradients_finite': True, 'upstream_gradients_nonzero_on_second_step': True,
            'forecast_patch_hour_counts': [16, 8], 'hourly_nwp_alignment_passed': True,
            'missing_value_mask_passed': True, 'weather_state_reload_exact': True,
            'transient_optimization_steps': steps, 'base_provenance': model.base_provenance,
        })
        del model, optimizer
    aligned = all(row['cached_vs_direct_prediction_matches'] for row in evidence['runs'])
    evidence['status'] = 'passed' if aligned else 'blocked_native_encoder_alignment'
    evidence['cached_path_microbatch_checks_passed'] = True
    evidence['full_experiment_ready'] = aligned and opts.gpu is not None
    output = Path(opts.output) if opts.output else (
        ROOT / 'reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012' /
        ('preflight_cpu.json' if opts.gpu is None else 'preflight_gpu.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'status': evidence['status'], 'device': str(device), 'evidence': str(output)}))
    return 0 if aligned else 1


if __name__ == '__main__':
    raise SystemExit(main())
