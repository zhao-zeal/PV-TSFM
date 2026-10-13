"""Check the approved readout repair on real train windows; save no experiment weights."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/FusionSF_inputs_two_protocols_seed42_20261011'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--variants', nargs='+', default=['power_future_nwp'],
                        choices=['power_future_nwp', 'power_satellite_history_future_nwp', 'power_history_future_nwp'])
    options = parser.parse_args()
    os.sched_setaffinity(0, [0, 7])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    import torch
    from run import configure_resources, parse_args
    from data_provider.data_factory import data_provider
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.losses import forecasting_loss

    argv = ['--model', 'FusionSF', '--fusion_input', options.variants[0], '--is_training', '1',
            '--seed', '42', '--seq_len', '24', '--pred_len', '24', '--batch_size', '256',
            '--eval_batch_size', '64', '--use_gpu', '1', '--gpu', str(options.gpu), '--num_workers', '0']
    resources = parse_args(argv + ['--protocol', 'in_domain'])
    configure_resources(resources)
    evidence = dict(stage='readout_repair_correctness_only', created_at=datetime.now().astimezone().isoformat(),
        gpu=options.gpu, cpu_cores=[0, 7], threads=2, num_workers=0, precision='fp32',
        initialization='positive_v1', initial_output_bound=[0.05, 0.15],
        readout_lr_factor=0.01, prior_initialization_only_probe_failed_variant='power_future_nwp',
        full_training_complete=False, formal_test_complete=False, runs=[])
    remaining = options.variants != ['power_future_nwp']
    path = REPORT / ('positive_readout_remaining_preflight_20261012.json' if remaining
                     else 'positive_readout_preflight_20261012.json')
    record = json.loads((REPORT / 'runs.json').read_text())
    failed = [Path(row['checkpoint_dir']) / 'checkpoint.pth' for row in record.get('failed_attempts', [])]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in failed if p.exists()}
    jobs = [(variant, protocol) for variant in options.variants for protocol in ('in_domain', 'zeroshot_v1')]
    for variant, protocol in jobs:
        batch_size = record['throughput_settings_20261012'][variant]['batch_size']
        argv[argv.index('--fusion_input') + 1] = variant
        argv[argv.index('--batch_size') + 1] = str(batch_size)
        torch.manual_seed(42)
        reference = Exp_Long_Term_Forecast(parse_args(argv + ['--protocol', protocol]))
        official = {key: value.detach().cpu().clone() for key, value in reference.model.state_dict().items()}
        original_modules = [(name, type(module).__name__) for name, module in reference.model.named_modules()]
        del reference
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        args = parse_args(argv + ['--protocol', protocol, '--fusion_readout_init', 'positive_v1'])
        experiment = Exp_Long_Term_Forecast(args)
        assert original_modules == [(name, type(module).__name__) for name, module in experiment.model.named_modules()]
        current = experiment.model.state_dict()
        changed = [key for key in official if not torch.equal(official[key], current[key].cpu())]
        assert set(changed) == {'backbone.mlp_heads.0.1.weight', 'backbone.mlp_heads.0.1.bias'}
        dataset, loader = data_provider(args, 'train')
        iterator = iter(loader)
        batch = next(iterator)
        optimizer = experiment._select_optimizer()
        scheduler = experiment._select_scheduler(optimizer)
        experiment.model.train()
        steps = []
        for step in range(24 if remaining else 12):
            if remaining and step == 12:
                # Bounded correctness check at the peak rate too, not a formal epoch trajectory.
                for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                    group['lr'] = base_lr
            optimizer.zero_grad(set_to_none=True)
            result, targets, _ = experiment._forward_batch(batch)
            prediction = result['prediction']
            if step == 0:
                assert torch.all((prediction >= 0.05 - 1e-6) & (prediction <= 0.15 + 1e-6))
            assert torch.count_nonzero(prediction) > 0, 'Readout returned to the all-zero state'
            loss = forecasting_loss(result, targets, 'FusionSF')
            assert torch.isfinite(loss)
            supervised_loss = torch.nn.functional.l1_loss(prediction, targets)
            supervised_loss.backward(retain_graph=True)
            gradients = {name: float(parameter.grad.abs().sum()) for name, parameter in experiment.model.named_parameters()
                         if parameter.grad is not None}
            assert all(torch.isfinite(parameter.grad).all() for parameter in experiment.model.parameters()
                       if parameter.grad is not None)
            assert gradients['backbone.mlp_heads.0.1.weight'] > 0
            decoder_gradient = sum(value for name, value in gradients.items() if name.startswith('backbone.temporal_transformer.'))
            weather_gradient = sum(value for name, value in gradients.items() if name.startswith('backbone.guide_encoder.'))
            assert decoder_gradient > 0 and weather_gradient > 0
            context_gradient = sum(value for name, value in gradients.items() if name.startswith('backbone.ctx_encoder.'))
            if experiment.model.use_satellite:
                assert context_gradient > 0
            steps.append(dict(step=step + 1, lr=optimizer.param_groups[0]['lr'], loss=float(loss.detach()),
                supervised_l1_loss=float(supervised_loss.detach()),
                nonzero_predictions=int(torch.count_nonzero(prediction)),
                prediction_min=float(prediction.detach().min()), prediction_max=float(prediction.detach().max()),
                readout_lr=optimizer.param_groups[1]['lr'],
                readout_gradient_l1=gradients['backbone.mlp_heads.0.1.weight'],
                decoder_gradient_l1=decoder_gradient, weather_encoder_gradient_l1=weather_gradient,
                satellite_encoder_supervised_gradient_l1=context_gradient if experiment.model.use_satellite else None))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in experiment.model.parameters()
                       if parameter.grad is not None)
            optimizer.step()
            if step == 0:
                # Only probe the next warmup LR; this does not simulate a complete epoch.
                scheduler.step()
            batch = next(iterator)
        experiment.model.eval()
        with torch.no_grad():
            prediction = experiment._forward_batch(batch)[0]['prediction']
        assert torch.isfinite(prediction).all() and torch.count_nonzero(prediction) > 0
        state = {key: value.detach().clone() for key, value in experiment.model.state_dict().items()}
        experiment.model.load_state_dict(state, strict=True)
        with torch.no_grad():
            replay = experiment._forward_batch(batch)[0]['prediction']
        assert torch.equal(prediction, replay)
        evidence['runs'].append(dict(variant=variant, protocol=protocol, batch_size=batch_size, train_windows=len(dataset),
            changed_initial_state_keys=changed, unchanged_architecture=True,
            preserved_relu_vq_loss_and_backbone_initialization=True,
            transient_steps_at_initial_warmup_and_peak_lr=steps,
            supervised_readout_decoder_weather_gradients_nonzero=True,
            final_eval_nonzero_predictions=int(torch.count_nonzero(prediction)), state_reload_exact=True,
            config=vars(args)))
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
        del experiment, optimizer, scheduler, loader, iterator, dataset, batch, state, official
        torch.cuda.empty_cache()
        print(json.dumps(dict(variant=variant, protocol=protocol, batch_size=batch_size,
                             status='passed', final_nonzero=evidence['runs'][-1]['final_eval_nonzero_predictions'])), flush=True)
    assert all(value == hashlib.sha256(Path(p).read_bytes()).hexdigest() for p, value in source_hashes.items())
    evidence.update(status='passed_real_batch_positive_readout_and_supervised_gradients',
                    failed_checkpoint_sha256_unchanged=source_hashes)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=evidence['status'], evidence=str(path))))


if __name__ == '__main__':
    main()
