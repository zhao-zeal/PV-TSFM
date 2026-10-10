"""Run a requested model with FusionSF script settings and an explicitly selected MMSP protocol."""

import argparse
from pathlib import Path
from pvtsfm.paths import MMSP_DIR

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pvtsfm.launcher import PVLauncher as BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from pvtsfm.data.mmsp import MMSPForecastingDataset, canonical_protocol
from pvtsfm.models.fusion_baselines import (
    DLinearConfig, FusionSFDLinear, FusionSFPatchTST, PatchTSTConfig,
)
from basicts.runners.callback import EarlyStopping

from .callbacks import MMSPResults, protocol_output_dir
from .fusionsf_config import ROOT, build_config
from pvtsfm.resources import require_idle_gpus


def reproduction_config(model, gpus, output_dir=None, data_dir=None, protocol='in_domain'):
    protocol = canonical_protocol(protocol)
    output_dir = protocol_output_dir(output_dir, protocol, ROOT / 'checkpoints' / protocol / model / 'script' / 'seed_42')
    if model == 'FusionSF':
        cfg = build_config('fusionSF_3modal', 'script', gpus, data_dir,
                           output_dir, 42, 100, 16, 4, protocol)
        cfg.test_batch_size = 64
        cfg.train_data_pin_memory = cfg.val_data_pin_memory = cfg.test_data_pin_memory = True
        return cfg
    is_patch = model == 'PatchTST'
    return BasicTSForecastingConfig(
        model=FusionSFPatchTST if is_patch else FusionSFDLinear,
        model_config=PatchTSTConfig() if is_patch else DLinearConfig(),
        dataset_name='MMSP', dataset_type=MMSPForecastingDataset,
        dataset_params={
            'data_dir': str(Path(data_dir or MMSP_DIR).resolve()),
            'protocol': protocol, 'modality_mode': 'power',
            'input_len': 24, 'output_len': 24, 'num_sites': 20 if protocol == 'zeroshot_v1' else 10,
            'num_ignored_sites': 10 if protocol == 'zeroshot_v1' else 0,
            'train_ratio': 0.6, 'valid_ratio': 0.2, 'test_ratio': 0.2,
        },
        input_len=24, output_len=24, gpus=gpus, seed=42,
        scaler=None, rescale=False, null_val=float('nan'), loss='MSE',
        optimizer=AdamW,
        optimizer_params={'lr': 0.0002 if is_patch else 0.0016,
                          'weight_decay': 0.05, 'betas': [0.9, 0.95]},
        lr_scheduler=ReduceLROnPlateau,
        lr_scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 10},
        lr_scheduler_monitor='RMSE', num_epochs=50, batch_size=16 if is_patch else 64,
        metrics=['MAE', 'RMSE'], target_metric='MAE', best_metric='min',
        callbacks=[EarlyStopping(patience=20 if is_patch else 10), MMSPResults()],
        train_data_num_workers=4, val_data_num_workers=4, test_data_num_workers=4,
        test_interval=51, eval_after_train=True, save_results=True,
        ckpt_save_dir=str(output_dir), data_protocol=protocol,
        fusionsf_preset='TSbaselines.sh',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['FusionSF', 'DLinear', 'PatchTST'])
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--protocol', required=True, choices=['in_domain', 'zeroshot_v1', 'paper_main_v1', 'fixed_v1', 'paper_main'])
    parser.add_argument('--output-dir')
    parser.add_argument('--data-dir')
    parser.add_argument('--checkpoint', help='Evaluate an existing baseline checkpoint without training')
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args()
    cfg = reproduction_config(args.model, args.gpus, args.output_dir, args.data_dir, args.protocol)
    if args.print_config:
        print(cfg)
        return
    require_idle_gpus(args.gpus)
    if args.checkpoint:
        BasicTSLauncher.launch_evaluation(cfg, args.checkpoint, gpus=args.gpus)
    else:
        BasicTSLauncher.launch_training(cfg)


if __name__ == '__main__':
    main()
