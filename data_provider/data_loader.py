"""One MMSP interface for every model and all project protocols."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .protocols import MMSPProtocolDataset


class Dataset_MMSP(MMSPProtocolDataset):
    """TSLib's four sequence tensors plus one fixed covariate dictionary.

    seq_y includes label_len history values followed by pred_len labels. The
    experiment constructs decoder input with zeros in place of future labels.
    Unloaded modalities have empty tensors; every sample has the same keys.
    """

    def __init__(self, root_path, flag='train', size=(24, 12, 24),
                 protocol='in_domain', modalities='power'):
        self.seq_len, self.label_len, self.pred_len = size
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError('label_len must be between 0 and seq_len')
        super().__init__('MMSP', flag, root_path, input_len=self.seq_len,
                         output_len=self.pred_len, protocol=protocol,
                         modality_mode=modalities)
        times = pd.DatetimeIndex(self.storage.data_sp_time_dt)
        self.calendar = torch.from_numpy(np.stack([
            times.hour / 23 - 0.5, times.dayofweek / 6 - 0.5,
            (times.day - 1) / 30 - 0.5, (times.dayofyear - 1) / 365 - 0.5,
        ], axis=-1)).float()
        self.first_input_index = min(
            self.storage.window_records[i].start_index for i in self.window_indices)

    def __getitem__(self, index):
        site_index, window_index = divmod(index, len(self.window_indices))
        record = self.storage.window_records[self.window_indices[window_index]]
        site = self.storage.data_sp[site_index]
        begin, end = record.start_index, record.start_index + self.seq_len
        label_begin, target_end = end - self.label_len, end + self.pred_len
        sample = super().__getitem__(index)
        # Preserve Cross-Unet's published correlation block independently of model selection.
        correlation_begin = begin if begin - self.first_input_index < self.seq_len else begin - self.seq_len
        covariates = {
            'site_coords': site['lats_lons'].float(),
            'historical_nwp': torch.empty(0),
            'future_nwp': torch.empty(0),
            'satellite': sample.get('stl_input', torch.empty(0)).float(),
            'satellite_coords': sample.get('stl_coords', torch.empty(0)).float(),
            'fusion_time': sample.get('ts_time', torch.empty(0)).float(),
            'correlation_history': site['values'][correlation_begin:correlation_begin + self.seq_len].unsqueeze(-1).float(),
            'site_id': sample['site_id'],
            'forecast_timestamps': sample['forecast_timestamps'],
            'input_start_timestamp': sample['input_start_timestamp'],
            'input_end_timestamp': sample['input_end_timestamp'],
            'forecast_start_timestamp': sample['forecast_start_timestamp'],
            'forecast_end_timestamp': sample['forecast_end_timestamp'],
            'split_id': sample['split_id'],
        }
        if hasattr(self.storage, 'data_ec_grouped'):
            lat, lon = (round(float(value), 1) for value in site['lats_lons'])
            offset = int((record.input_start - self.storage.ec_start_time).total_seconds() // 3600)
            values = self.storage.data_ec_grouped.get_group((lat, lon)).values
            covariates['historical_nwp'] = torch.from_numpy(values[offset:offset + self.seq_len]).float()
            covariates['future_nwp'] = torch.from_numpy(values[offset + self.seq_len:offset + self.seq_len + self.pred_len]).float()
        return (
            sample['inputs'],
            site['values'][label_begin:target_end].unsqueeze(-1).float(),
            self.calendar[begin:end],
            self.calendar[label_begin:target_end],
            covariates,
        )

    def save_protocol(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = super().save_protocol(output_dir)
        summary['data_interface'] = 'tslib_four_sequences_plus_covariates_v1'
        summary['label_len'] = self.label_len
        summary['modalities'] = self.storage.modality_mode
        summary['calendar_features'] = ['hour', 'dayofweek', 'day', 'dayofyear']
        summary['future_nwp_availability'] = 'assumed available; issue time absent from CSV'
        (Path(output_dir) / 'data_protocol.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
