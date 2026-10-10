"""MMSP inputs under FusionSF's original paper and target-time protocols."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .split_utils import build_split_contained_windows, build_target_time_windows, chronological_boundaries


def get_data_spower(data_dir, num_sites=10, num_ignored_sites=0, strict=True):
    frame = pd.read_csv(Path(data_dir) / 'solar_power/solar_power.csv', parse_dates=['datetime'])
    if strict and frame[['datetime', 'power', 'site']].isna().any().any():
        raise ValueError('fixed_v1 does not silently replace missing power/timestamp/site values')
    if not strict:
        frame = frame.fillna(0)
    sites = []
    reference_times = None
    for index, (site, group) in enumerate(frame.groupby('site')):
        if index >= num_sites:
            break
        if index < num_ignored_sites:
            continue
        times = pd.DatetimeIndex(group['datetime'])
        if reference_times is None:
            reference_times = times
        elif not times.equals(reference_times):
            raise ValueError(f'site {site} timestamps are not aligned with the first selected site')
        sites.append({
            'site': site,
            'values': torch.from_numpy(group['power'].values),
            'lats_lons': torch.tensor([group['lat'].iloc[0], group['lon'].iloc[0]]),
        })
    time_features = torch.from_numpy(np.stack([
        reference_times.month, reference_times.day, reference_times.hour,
    ]))
    return sites, time_features, pd.Series(reference_times), len(reference_times)


def get_data_satellite(data_dir, satellite_dir, fit_end_time, external_scaler_state=None):
    folder = Path(data_dir) / satellite_dir
    values = np.load(folder / 'satellite.npy')
    times = np.load(folder / 'satellite_times.npy')
    coords = torch.from_numpy(np.load(folder / 'satellite_coords.npy'))
    flattened = values.reshape(len(values), -1)
    if external_scaler_state is None:
        train_times = pd.DatetimeIndex(times)
        fit_mask = np.ones(len(train_times), dtype=bool) if fit_end_time is None else train_times < pd.Timestamp(fit_end_time)
        scaler = StandardScaler().fit(flattened[fit_mask])
        state = {
            'mean': scaler.mean_, 'scale': scaler.scale_,
            'fit_range': 'full_dataset_legacy' if fit_end_time is None else f'{train_times[fit_mask].min()}..{train_times[fit_mask].max()}',
        }
        values = scaler.transform(flattened).reshape(values.shape)
    else:
        state = {**external_scaler_state, 'source': 'external_training_dataset'}
        values = ((flattened - state['mean']) / state['scale']).reshape(values.shape)
    # Preserve FusionSF's single satellite channel and per-pixel normalization.
    return torch.from_numpy(values[..., :1]), times, coords, state


def get_data_nwp(data_dir, fit_end_time, fit_coordinates, external_scaler_state=None):
    frame = pd.read_csv(Path(data_dir) / 'nwp/nwp.csv', parse_dates=['fcst_date'])
    frame['lat'] = frame['lat'].round(1)
    frame['lon'] = frame['lon'].round(1)
    columns = frame.columns.drop(['fcst_date', 'lat', 'lon'])
    fixed_processing = fit_end_time is not None or external_scaler_state is not None
    if fixed_processing:
        frame = frame.sort_values(['lat', 'lon', 'fcst_date'])
        frame[columns] = frame.groupby(['lat', 'lon'], sort=False)[columns].transform(
            lambda group: group.interpolate(limit_direction='both')
        )
    else:
        frame = frame.interpolate()
    start_time = frame['fcst_date'].iloc[0]
    times = frame['fcst_date'].copy()
    if external_scaler_state is None:
        allowed = {(round(float(lat), 1), round(float(lon), 1)) for lat, lon in fit_coordinates}
        time_mask = np.ones(len(frame), dtype=bool) if fit_end_time is None else times < pd.Timestamp(fit_end_time)
        fit_mask = time_mask & np.array([
            (lat, lon) in allowed for lat, lon in zip(frame['lat'], frame['lon'])
        ])
        scaler = StandardScaler().fit(frame.loc[fit_mask, columns])
        frame.loc[:, columns] = scaler.transform(frame[columns])
        state = {
            'mean': scaler.mean_, 'scale': scaler.scale_,
            'feature_names': columns.tolist(),
            'fit_range': 'full_dataset_legacy' if fit_end_time is None else f'{times[fit_mask].min()}..{times[fit_mask].max()}',
            'fit_coordinates': sorted(allowed),
        }
    else:
        state = {**external_scaler_state, 'source': 'external_training_dataset'}
        if state['feature_names'] != columns.tolist():
            raise ValueError('external NWP scaler feature order does not match test NWP data')
        frame.loc[:, columns] = (frame[columns].to_numpy() - state['mean']) / state['scale']
    frame = frame.drop(columns=['fcst_date'])
    # Paper representation retains the two coordinates as guide channels.
    grouped = (frame.set_index(['lat', 'lon']).groupby(level=['lat', 'lon'])
               if fixed_processing else frame.groupby(['lat', 'lon']))
    return grouped, start_time, state


class MMSPDataset(Dataset):
    def __init__(
        self, data_dir, seq_len=24, label_len=12, pred_len=24,
        num_sites=10, num_ignored_sites=0, modality_mode='power',
        satellite_dir='satellite', data_pipeline=None,
        train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,
        stride=1, precomputed_scaler_state=None,
    ):
        self.data_pipeline = dict(data_pipeline or {'version': 'fixed_v1'})
        if self.data_pipeline['version'] not in ('fixed_v1', 'paper_main_v1'):
            raise ValueError(f'Unknown window pipeline: {self.data_pipeline}')
        if modality_mode not in {'power', 'power_nwp', 'satellite', 'all'}:
            raise ValueError(f'Unknown input mode: {modality_mode}')
        self.pipeline_version = self.data_pipeline['version']
        self.modality_mode = modality_mode
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.num_sites, self.num_ignored_sites = num_sites, num_ignored_sites
        self.data_sp, self.data_sp_time, self.data_sp_time_dt, self.data_sp_length = get_data_spower(
            data_dir, num_sites, num_ignored_sites, strict=self.pipeline_version == 'fixed_v1',
        )
        window_builder = build_split_contained_windows if self.pipeline_version == 'paper_main_v1' else build_target_time_windows
        self.window_records = window_builder(
            self.data_sp_time_dt, seq_len, pred_len, train_ratio, valid_ratio, test_ratio, stride,
        )
        self.boundaries = chronological_boundaries(
            self.data_sp_time_dt, train_ratio, valid_ratio, test_ratio,
        )
        fit_end_time = self.boundaries['train'][1] if self.pipeline_version == 'fixed_v1' else None
        self.scaler_state = {'fit_end_exclusive': str(fit_end_time)} if fit_end_time is not None else {}
        external = precomputed_scaler_state or {}
        if precomputed_scaler_state is not None:
            required = {'satellite', 'nwp'} if modality_mode == 'all' else ({'nwp'} if modality_mode == 'power_nwp' else ({'satellite'} if modality_mode == 'satellite' else set()))
            if not required.issubset(external):
                raise ValueError('Cross-site evaluation must reuse the training dataset scalers')
        if modality_mode in {'satellite', 'all'}:
            self.data_stl, self.data_stl_times, self.data_stl_coords, state = get_data_satellite(
                data_dir, satellite_dir, fit_end_time, external.get('satellite'),
            )
            self.scaler_state['satellite'] = state
        if modality_mode in {'all', 'power_nwp'}:
            coordinates = [site['lats_lons'].tolist() for site in self.data_sp]
            self.data_ec_grouped, self.ec_start_time, state = get_data_nwp(
                data_dir, fit_end_time, coordinates, external.get('nwp'),
            )
            self.scaler_state['nwp'] = state
            self.nwp_channels = len(state['feature_names']) + (2 if self.pipeline_version == 'paper_main_v1' else 0)

    def __len__(self):
        return len(self.data_sp) * len(self.window_records)

    def __getitem__(self, index):
        site_index, window_index = divmod(index, len(self.window_records))
        site = self.data_sp[site_index]
        record = self.window_records[window_index]
        x_begin = record.start_index
        x_end = x_begin + self.seq_len
        y_end = x_end + self.pred_len
        sample = {
            'ts_input': site['values'][x_begin:x_end].unsqueeze(-1),
            'ts_target': site['values'][x_end:y_end],
            'ts_coords': site['lats_lons'],
            'site_id': torch.tensor(int(site['site']), dtype=torch.long),
            'input_start_timestamp': torch.tensor(record.input_start.value),
            'input_end_timestamp': torch.tensor(record.input_end.value),
            'forecast_start_timestamp': torch.tensor(record.forecast_start.value),
            'forecast_end_timestamp': torch.tensor(record.forecast_end.value),
            'forecast_timestamps': torch.tensor(
                pd.DatetimeIndex(self.data_sp_time_dt.iloc[x_end:y_end]).asi8.copy(),
            ),
            'split_id': torch.tensor(
                {'train': 0, 'validation': 1, 'test': 2}[record.split], dtype=torch.int8,
            ),
        }
        if self.modality_mode in {'power', 'power_nwp'}:
            sample['ts_time_x'] = self.data_sp_time[:, x_begin:x_end].T
            sample['ts_time_y'] = self.data_sp_time[:, x_end:y_end].T
            if self.modality_mode == 'power_nwp':
                lat, lon = (round(float(value), 1) for value in site['lats_lons'])
                ec_begin = int((record.input_start - self.ec_start_time).total_seconds() // 3600)
                values = self.data_ec_grouped.get_group((lat, lon)).values
                sample['historical_nwp'] = torch.from_numpy(values[ec_begin:ec_begin + self.seq_len]).float()
                sample['future_nwp'] = torch.from_numpy(values[ec_begin + self.seq_len:ec_begin + self.seq_len + self.pred_len]).float()
            return sample

        stl_begin = int((record.input_start - pd.Timestamp(self.data_stl_times[0])).total_seconds() // 3600)
        images = rearrange(self.data_stl[stl_begin:stl_begin + self.seq_len], 't h w c -> t c h w')
        height, width = images.shape[-2:]
        sample['ts_coords'] = sample['ts_coords'][:, None, None]
        sample['ts_time'] = repeat(self.data_sp_time[:, x_begin:x_end], 'c t -> t c h w', h=height, w=width)
        sample['stl_input'] = images
        sample['stl_coords'] = rearrange(self.data_stl_coords, 'h w c -> c h w')
        if self.modality_mode == 'all':
            lat, lon = (round(float(value), 1) for value in site['lats_lons'])
            ec_begin = int((record.input_start - self.ec_start_time).total_seconds() // 3600) + self.seq_len
            values = self.data_ec_grouped.get_group((lat, lon)).values
            sample['ec_input'] = torch.from_numpy(values[ec_begin:ec_begin + self.pred_len])
        return sample
