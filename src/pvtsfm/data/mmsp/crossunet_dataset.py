"""Cross-Unet covariates on the existing MMSP protocols, without satellite data."""

import json
from pathlib import Path

from .dataset import MMSPForecastingDataset


class MMSPCrossUnetDataset(MMSPForecastingDataset):
    def __init__(self, dataset_name, mode, data_dir, input_len=24, output_len=24,
                 protocol='in_domain', memmap=False):
        # MMSP's future NWP block has output_len steps. Cross-Unet concatenates
        # it with input_len history steps, so this entry keeps the existing 24→24 task.
        if input_len != output_len:
            raise ValueError('MMSP Cross-Unet requires input_len == output_len')
        super().__init__(dataset_name, mode, data_dir, input_len=input_len,
                         output_len=output_len, protocol=protocol,
                         modality_mode='power_nwp', memmap=memmap)
        self.first_input_index = min(
            self.storage.window_records[i].start_index for i in self.window_indices)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        site_index, window_index = divmod(index, len(self.window_indices))
        record = self.storage.window_records[self.window_indices[window_index]]
        begin, length = record.start_index, self.storage.seq_len
        # Official Dataset_Custom uses the current block for the first seq_len
        # windows of each split, then the immediately preceding history block.
        history_begin = begin if begin - self.first_input_index < length else begin - length
        sample['correlation_history'] = self.storage.data_sp[site_index]['values'][
            history_begin:history_begin + length].unsqueeze(-1).float()
        sample['future_weather'] = sample.pop('future_nwp')
        sample['history_weather'] = sample.pop('historical_nwp')
        return sample

    def save_protocol(self, output_dir):
        summary = super().save_protocol(output_dir)
        summary['crossunet_inputs'] = {
            'history_features': ['power'],
            'weather_features': self.training_storage.scaler_state['nwp']['feature_names'],
            'future_weather_length': self.storage.seq_len,
            'correlation_history': 'previous input block; current block for first input_len windows per split',
            'power_scaling': 'existing MMSP power values; no additional standardization',
            'future_weather_availability': 'assumed available; CSV does not record forecast issue time',
        }
        (Path(output_dir) / 'data_protocol.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
