"""Evaluate frozen Chronos-2 or TimesFM 3.0 with an explicit MMSP protocol."""

import argparse
import json
import os
import sys
from pathlib import Path
from pvtsfm.paths import MMSP_DIR, PRETRAINED_DIR

# Apply the shared CPU budget before importing numerical libraries.
CPU_ONLY = '--device=cpu' in sys.argv or ('--device' in sys.argv and sys.argv[sys.argv.index('--device') + 1] == 'cpu')
if CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '4'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import torch
from easytorch.device import set_device_type
from easytorch.utils import set_visible_devices

from basicts.configs import BasicTSForecastingConfig
from pvtsfm.data.mmsp import MMSPForecastingDataset, canonical_protocol
from pvtsfm.models.chronos2 import ChronosC0, ChronosC0Config
from pvtsfm.runner import PVRunner

from .callbacks import MMSPResults, protocol_output_dir
from pvtsfm.resources import require_idle_gpus


ROOT = Path(__file__).resolve().parents[3]


class C0Runner(PVRunner):
    def on_eval_start(self, ckpt_path):
        # The frozen backbone is loaded from its local Hugging Face checkpoint.
        # BasicTS's standard evaluation loop, metrics and result export follow.
        self.logger.info('C0: local pretrained weights loaded; no training or optimizer updates.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['chronos2', 'timesfm3'], default='chronos2')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--gpu', default='6', help='One physical idle GPU; check nvidia-smi before launch')
    parser.add_argument('--model-path')
    parser.add_argument('--data-dir', default=str(MMSP_DIR))
    parser.add_argument('--protocol', required=True, choices=['in_domain', 'zeroshot_v1', 'paper_main_v1', 'fixed_v1', 'paper_main'])
    parser.add_argument('--output-dir')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nwp-mode', choices=['none', 'history', 'future', 'history_future'], default='none')
    args = parser.parse_args()
    args.protocol = canonical_protocol(args.protocol)
    if args.model != 'chronos2' and args.nwp_mode != 'none':
        parser.error('NWP combinations are available for Chronos-2 only')
    if ',' in args.gpu:
        parser.error('C0 uses one idle GPU')
    if args.device == 'gpu':
        require_idle_gpus(args.gpu)
        set_device_type('gpu')
        set_visible_devices(args.gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    set_device_type(args.device)
    is_timesfm = args.model == 'timesfm3'
    if is_timesfm:
        from pvtsfm.models.timesfm3 import TimesFMC0, TimesFMC0Config
    model = TimesFMC0 if is_timesfm else ChronosC0
    config_type = TimesFMC0Config if is_timesfm else ChronosC0Config
    model_path = args.model_path or str(PRETRAINED_DIR / ('timesfm-3.0-pytorch' if is_timesfm else 'chronos-2'))
    run_name = 'TimesFM3_official' if is_timesfm else 'C0'
    experiment_name = run_name if args.nwp_mode == 'none' else f'Chronos2_power_{args.nwp_mode}_nwp'
    model_config = config_type(model_path=str(Path(model_path).resolve()), batch_size=args.batch_size)
    if not is_timesfm:
        model_config.nwp_mode = args.nwp_mode
    cfg = BasicTSForecastingConfig(
        model=model,
        model_config=model_config,
        dataset_name='MMSP', dataset_type=MMSPForecastingDataset,
        dataset_params={
            'data_dir': str(Path(args.data_dir).resolve()), 'protocol': args.protocol,
            'input_len': 24, 'output_len': 24, 'num_sites': 20 if args.protocol == 'zeroshot_v1' else 10,
            'num_ignored_sites': 10 if args.protocol == 'zeroshot_v1' else 0,
            'modality_mode': 'power' if args.nwp_mode == 'none' else 'power_nwp', 'train_ratio': 0.6, 'valid_ratio': 0.2, 'test_ratio': 0.2,
        },
        input_len=24, output_len=24, gpus=args.gpu if args.device == 'gpu' else None, seed=args.seed,
        batch_size=args.batch_size, scaler=None, rescale=False, null_val=float('nan'),
        metrics=['MAE', 'RMSE'], target_metric='MAE', best_metric='min',
        callbacks=[MMSPResults()], test_data_num_workers=0,
        eval_after_train=False, save_results=True, data_protocol=args.protocol,
        experiment_model_name=experiment_name,
        ckpt_save_dir=protocol_output_dir(args.output_dir, args.protocol, ROOT / 'outputs' / args.protocol / experiment_name / f'seed_{args.seed}'),
    )
    cfg.save()
    runner = C0Runner(cfg)
    runner.init_logger(logger_name='BasicTS-C0', log_file_name='evaluation_log')
    runner.eval()
    metadata = {
        'data_protocol': args.protocol, 'experiment': run_name,
        'model': 'google/timesfm-3.0-pytorch' if is_timesfm else 'amazon/chronos-2',
        'model_path': cfg.model_config.model_path, 'device': args.device,
        'seed': args.seed, 'batch_size': args.batch_size, 'input_len': 24, 'output_len': 24,
        'frozen': True, 'power_context_only': args.nwp_mode == 'none', 'point_quantile': 0.5,
        'quantile_levels': [i / 10 for i in range(1, 10)] if is_timesfm else [0.1, 0.5, 0.9],
        'prediction_clipping': cfg.model_config.make_positive if is_timesfm else False, 'finetuned': False,
        'sort_quantiles': True if is_timesfm else None,
        'satellite_used': False, 'nwp_used': args.nwp_mode != 'none',
        'nwp_mode': args.nwp_mode,
        'past_nwp_values_provided': args.nwp_mode in ('history', 'history_future'),
        'future_nwp_values_provided': args.nwp_mode in ('future', 'history_future'),
        'future_power_provided': False,
        'nwp_scaling': 'protocol_training_sites_and_time_standardization_then_chronos_internal_scaling' if args.nwp_mode != 'none' else None,
        'nwp_features': runner.test_data_loader.dataset.storage.scaler_state.get('nwp', {}).get('feature_names', []),
        'future_nwp_availability': 'assumed_available_at_forecast_origin; CSV lacks issue timestamps' if args.nwp_mode in ('future', 'history_future') else None,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if args.device == 'cpu' else None,
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'data_loader_workers': 0,
    }
    if is_timesfm:
        metadata.update(
            evaluation_api='TimesFM3Evaluator', official_benchmark_switches=True,
            **{name: getattr(cfg.model_config, name) for name in (
                'use_symmetric_averaging', 'make_positive', 'sort_quantiles',
                'use_znorm', 'padding_mode', 'return_quantiles')},
            nonnegative_policy='Clamp only when all original context values are nonnegative; retain raw small negative power readings.',
        )
    else:
        metadata.update(
            evaluation_api='Chronos2Pipeline.predict_quantiles',
            cross_learning=cfg.model_config.cross_learning,
            context_length_limit=runner.model.pipeline.model_context_length,
            actual_context_length=cfg.model_config.input_len,
            limit_prediction_length=False, model_native_scaling=True,
            quantile_sorting=False,
        )
    (Path(runner.ckpt_save_dir) / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{run_name}_RESULT_DIR={runner.ckpt_save_dir}')


if __name__ == '__main__':
    main()
