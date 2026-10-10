"""Historical power/NWP interface for official TimeXer single-target forecasting."""

import numpy as np
import pandas as pd
import torch

from .dataset import MMSPForecastingDataset


class MMSPTimeXerDataset(MMSPForecastingDataset):
    def __init__(self, dataset_name, mode, data_dir, input_len=24, output_len=24,
                 protocol='in_domain', memmap=False):
        super().__init__(dataset_name, mode, data_dir, input_len=input_len,
                         output_len=output_len, protocol=protocol,
                         modality_mode='power_nwp', memmap=memmap)
        times = pd.DatetimeIndex(self.storage.data_sp_time_dt)
        # Match TimeXer's time_features(freq='h') order and normalization.
        self.calendar_features = torch.from_numpy(np.stack([
            times.hour / 23 - 0.5, times.dayofweek / 6 - 0.5,
            (times.day - 1) / 30 - 0.5, (times.dayofyear - 1) / 365 - 0.5,
        ], axis=-1)).float()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample.pop('future_nwp')
        sample.pop('ts_time_x')
        sample.pop('ts_time_y')
        window_index = index % len(self.window_indices)
        begin = self.storage.window_records[self.window_indices[window_index]].start_index
        sample['inputs_timestamps'] = self.calendar_features[begin:begin + self.storage.seq_len]
        return sample
