"""Prepare and inspect TimeXer input routing; this module does not launch training."""

import argparse
from pathlib import Path
from pvtsfm.paths import MMSP_DIR

from basicts.configs import BasicTSForecastingConfig
from pvtsfm.data.mmsp.timexer_dataset import MMSPTimeXerDataset
from pvtsfm.models.timexer import TimeXerPV, TimeXerPVConfig


ROOT = Path(__file__).resolve().parents[3]


def build_config(protocol, data_dir=None, output_dir=None):
    if protocol not in ('in_domain', 'zeroshot_v1'):
        raise ValueError('Choose in_domain or zeroshot_v1')
    return BasicTSForecastingConfig(
        model=TimeXerPV,
        model_config=TimeXerPVConfig(input_len=24, output_len=24, patch_len=12),
        dataset_name='MMSP', dataset_type=MMSPTimeXerDataset,
        dataset_params={'data_dir': str(Path(data_dir or MMSP_DIR).resolve()),
                        'protocol': protocol, 'input_len': 24, 'output_len': 24},
        input_len=24, output_len=24, gpus=None, seed=2021,
        scaler=None, rescale=False, null_val=float('nan'),
        loss='MSE', metrics=['MSE', 'MAE', 'RMSE'], target_metric='MSE',
        train_data_num_workers=0, val_data_num_workers=0, test_data_num_workers=0,
        ckpt_save_dir=str(output_dir or ROOT / 'outputs' / protocol / 'timexer_interface'),
        data_protocol=protocol, interface_only=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, choices=['in_domain', 'zeroshot_v1'])
    args = parser.parse_args()
    print(build_config(args.protocol))


if __name__ == '__main__':
    main()
