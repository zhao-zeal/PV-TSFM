"""Expose Site #1, in-domain, cross-site and historical paper MMSP protocols."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset

from .mmsp_storage import MMSPDataset


def canonical_protocol(protocol):
    """Resolve the historical names of the project's same-site protocol."""
    protocol = 'in_domain' if protocol in ('fixed_v1', 'paper_main') else protocol
    if protocol not in ('site1_v1', 'in_domain', 'zeroshot_v1', 'paper_main_v1'):
        raise ValueError(f'Unknown MMSP protocol: {protocol}')
    return protocol


@lru_cache(maxsize=1)
def load_mmsp(data_dir, input_len, output_len, num_sites, num_ignored_sites,
              modality_mode, train_ratio, valid_ratio, test_ratio, protocol):
    protocol = canonical_protocol(protocol)
    if protocol == 'site1_v1':
        num_sites, num_ignored_sites = 2, 1
    elif protocol == 'zeroshot_v1':
        num_sites, num_ignored_sites = 20, 10
    elif protocol == 'paper_main_v1':
        num_sites, num_ignored_sites = 10, 0
    training = MMSPDataset(
        data_dir=data_dir, seq_len=input_len, pred_len=output_len,
        num_sites=num_sites, num_ignored_sites=num_ignored_sites,
        modality_mode=modality_mode, train_ratio=train_ratio,
        valid_ratio=valid_ratio, test_ratio=test_ratio,
        data_pipeline={'version': 'paper_main_v1' if protocol == 'paper_main_v1' else 'fixed_v1'},
    )
    if protocol != 'zeroshot_v1':
        return training, training
    testing = MMSPDataset(
        data_dir=data_dir, seq_len=input_len, pred_len=output_len,
        num_sites=10, num_ignored_sites=0, modality_mode=modality_mode,
        train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio,
        precomputed_scaler_state=training.scaler_state,
    )
    return training, testing


class MMSPProtocolDataset(Dataset):
    def __init__(self, dataset_name, mode, data_dir, input_len=24, output_len=24,
                 num_sites=10, num_ignored_sites=0, modality_mode='all',
                 train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,
                 memmap=False, protocol='in_domain'):
        super().__init__()
        self.protocol = canonical_protocol(protocol)
        self.training_storage, self.testing_storage = load_mmsp(
            str(Path(data_dir).resolve()), input_len, output_len, num_sites,
            num_ignored_sites, modality_mode, train_ratio, valid_ratio, test_ratio, self.protocol,
        )
        split = {'train': 'train', 'val': 'validation', 'test': 'test', 'eval': 'test'}[str(mode)]
        self.storage = self.testing_storage if split == 'test' else self.training_storage
        self.window_indices = [i for i, r in enumerate(self.storage.window_records) if r.split == split]

    def __len__(self):
        return len(self.storage.data_sp) * len(self.window_indices)

    def __getitem__(self, index):
        site_index, window_index = divmod(index, len(self.window_indices))
        raw_index = site_index * len(self.storage.window_records) + self.window_indices[window_index]
        sample = self.storage[raw_index]
        sample['inputs'] = sample.pop('ts_input').float()
        sample['targets'] = sample.pop('ts_target').unsqueeze(-1).float()
        return sample

    @property
    def data(self):
        return np.stack([site['values'].numpy() for site in self.storage.data_sp], axis=1)

    def save_protocol(self, output_dir):
        output_dir = Path(output_dir)
        storage = self.storage
        states = {k: v for k, v in self.training_storage.scaler_state.items() if isinstance(v, dict)}
        arrays = {f'{name}_{key}': state[key] for name, state in states.items() for key in ('mean', 'scale')}
        if arrays:
            np.savez(output_dir / 'scalers.npz', **arrays)
        summary = {
            'version': self.protocol,
            'protocol_origin': 'fusionsf_original_paper' if self.protocol == 'paper_main_v1' else 'project_fixed_v1',
            'previous_name': 'fixed_v1' if self.protocol == 'in_domain' else None,
            'window_pipeline': storage.pipeline_version,
            'split_version': 'split_contained_windows_60_20_20' if self.protocol == 'paper_main_v1' else 'chronological_target_split_v1',
            'scaler_version': 'full_dataset_official_code' if self.protocol == 'paper_main_v1' else 'train_sites_and_time_only_fit_v1',
            'nwp_channels': getattr(storage, 'nwp_channels', None),
            'nwp_interpolation': 'whole_frame_legacy' if self.protocol == 'paper_main_v1' else 'per_grid_bidirectional',
            'sites': [int(site['site']) for site in storage.data_sp],
            'train_sites': [int(site['site']) for site in self.training_storage.data_sp],
            'validation_sites': [int(site['site']) for site in self.training_storage.data_sp],
            'test_sites': [int(site['site']) for site in self.testing_storage.data_sp],
            'input_len': storage.seq_len, 'output_len': storage.pred_len,
            'target_time_ranges': {name: [str(x) for x in bounds] for name, bounds in storage.boundaries.items()},
            'windows': {name: len(data.data_sp) * sum(r.split == name for r in data.window_records)
                        for name, data in [('train', self.training_storage),
                                           ('validation', self.training_storage),
                                           ('test', self.testing_storage)]},
            'scaler_fit': {name: {k: v for k, v in state.items() if k not in ('mean', 'scale')}
                           for name, state in states.items()},
        }
        (output_dir / 'data_protocol.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
