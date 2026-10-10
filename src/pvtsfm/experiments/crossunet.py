"""Cross-Unet's effective official model/optimizer settings on MMSP 24→24."""

import argparse
from pathlib import Path
from pvtsfm.paths import MMSP_DIR

from torch.optim import Adam

from pvtsfm.launcher import PVLauncher as BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from pvtsfm.data.mmsp import canonical_protocol
from pvtsfm.data.mmsp.crossunet_dataset import MMSPCrossUnetDataset
from pvtsfm.models.crossunet import CrossUnet, CrossUnetConfig
from pvtsfm.models.crossunet.scheduler import CrossUnetType3LR
from basicts.runners.callback import EarlyStopping

from pvtsfm.experiments.callbacks import MMSPResults, protocol_output_dir
from pvtsfm.resources import require_idle_gpus


ROOT = Path(__file__).resolve().parents[3]


def build_config(protocol, gpus=None, data_dir=None, output_dir=None):
    protocol = canonical_protocol(protocol)
    if protocol not in ('in_domain', 'zeroshot_v1'):
        raise ValueError('Cross-Unet MMSP entry supports in_domain and zeroshot_v1')
    return BasicTSForecastingConfig(
        model=CrossUnet,
        model_config=CrossUnetConfig(input_len=24, output_len=24, seg_len=12,
                                     history_features=1, weather_features=15),
        dataset_name='MMSP', dataset_type=MMSPCrossUnetDataset,
        dataset_params={'data_dir': str(Path(data_dir or MMSP_DIR).resolve()),
                        'protocol': protocol, 'input_len': 24, 'output_len': 24},
        input_len=24, output_len=24, gpus=gpus, seed=2021,
        scaler=None, rescale=False, null_val=float('nan'), loss='MSE',
        optimizer=Adam, optimizer_params={'lr': 0.0001, 'betas': [0.95, 0.999],
                                          'weight_decay': 0.0},
        lr_scheduler=CrossUnetType3LR, lr_scheduler_params={},
        num_epochs=100, batch_size=128,
        metrics=['MSE', 'MAE', 'RMSE'], target_metric='MSE', best_metric='min',
        callbacks=[EarlyStopping(patience=10), MMSPResults()],
        train_data_num_workers=0, val_data_num_workers=0, test_data_num_workers=0,
        test_interval=101, eval_after_train=True, save_results=True,
        ckpt_save_dir=protocol_output_dir(output_dir, protocol,
            ROOT / 'checkpoints' / protocol / 'CrossUnet' / 'official_settings' / 'seed_2021'),
        data_protocol=protocol, experiment_model_name='CrossUnet',
        crossunet_preset='official_model_optimizer_on_project_mmsp',
        ddp_find_unused_parameters=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, choices=['in_domain', 'zeroshot_v1'])
    parser.add_argument('--gpus')
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args()
    cfg = build_config(args.protocol, args.gpus, args.data_dir, args.output_dir)
    if args.print_config:
        print(cfg)
        return
    if not args.gpus:
        parser.error('Training requires an explicitly selected idle GPU')
    require_idle_gpus(args.gpus)
    BasicTSLauncher.launch_training(cfg)


if __name__ == '__main__':
    main()
