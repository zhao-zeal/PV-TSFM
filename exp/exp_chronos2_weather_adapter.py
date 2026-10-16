"""Train only an hourly weather residual on accepted, frozen E11 features."""

import csv
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping


class Exp_Chronos2_Weather_Adapter(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super().__init__(args)
        self.cached_loaders = {}
        self.feature_dir = Path(args.output_dir) / args.protocol / (
            'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future')
        self.weather_dir = Path(args.output_dir) / args.protocol / (
            'Chronos2WeatherAdapter_features_v2_seed42_sl24_pl24_history_future')

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, _ = data_provider(self.args, flag)
        source = self.feature_dir / flag
        manifest = json.loads((source / 'complete.json').read_text())
        if (manifest['windows'] != len(dataset) or manifest['protocol'] != self.args.protocol
                or manifest['feature_shape'] != [2, 768]
                or Path(manifest['model_path']).resolve() != Path(self.args.model_path).resolve()):
            raise ValueError('Existing frozen feature cache does not match this experiment')
        destination = self.weather_dir / flag
        destination.mkdir(parents=True, exist_ok=True)
        current_protocol = dataset.save_protocol(destination)
        old_protocol = json.loads((source / 'data_protocol.json').read_text())
        if json.loads(json.dumps(current_protocol)) != old_protocol:
            raise ValueError('Weather NWP and frozen features use different data protocols')
        with np.load(source / 'scalers.npz') as old, np.load(destination / 'scalers.npz') as current:
            if set(old.files) != set(current.files) or any(
                    not np.array_equal(old[key], current[key]) for key in old.files):
                raise ValueError('NWP scaler differs from the accepted frozen features')
        tensors = [torch.from_numpy(np.load(source / f'{name}.npy')) for name in
                   ('embeds', 'loc', 'scale', 'targets', 'site_ids', 'forecast_timestamps')]
        if any(len(value) != len(dataset) for value in tensors):
            raise ValueError('Incomplete frozen feature cache')
        identity = {
            'windows': len(dataset), 'protocol': self.args.protocol,
            'source': str(source.resolve()),
            'source_manifest_sha256': hashlib.sha256((source / 'complete.json').read_bytes()).hexdigest(),
            'scaler_sha256': hashlib.sha256((source / 'scalers.npz').read_bytes()).hexdigest(),
            'future_nwp_shape': [len(dataset), 24, 15],
            'input': 'Dataset_MMSP train-standardized future NWP; no extra normalization',
        }
        complete = destination / 'complete.json'
        if complete.exists():
            if json.loads(complete.read_text()) != identity:
                raise ValueError('Weather feature cache identity mismatch; do not silently reuse it')
        else:
            arrays = np.lib.format.open_memmap(destination / 'future_nwp.npy', mode='w+',
                                              dtype=np.float32, shape=(len(dataset), 24, 15))
            loader = DataLoader(dataset, batch_size=self.args.cache_batch_size,
                                shuffle=False, num_workers=0)
            cursor = 0
            for batch in loader:
                _, targets, _, _, covariates = batch
                stop = cursor + len(targets)
                for actual, expected in ((targets[:, -24:], tensors[3][cursor:stop]),
                                         (covariates['site_id'], tensors[4][cursor:stop]),
                                         (covariates['forecast_timestamps'], tensors[5][cursor:stop])):
                    if not torch.equal(actual, expected):
                        raise ValueError(f'Frozen features/NWP window alignment differs at {cursor}')
                weather = covariates['future_nwp']
                if weather.shape[1:] != (24, 15) or not torch.isfinite(weather).all():
                    raise ValueError('Dataset_MMSP future NWP is not finite 24x15 data')
                arrays[cursor:stop] = weather.numpy()
                cursor = stop
            if cursor != len(dataset):
                raise ValueError('Incomplete future NWP cache')
            arrays.flush()
            del arrays
            complete.write_text(json.dumps(identity, indent=2) + '\n')
        future = torch.from_numpy(np.load(destination / 'future_nwp.npy'))
        if list(future.shape) != identity['future_nwp_shape'] or not torch.isfinite(future).all():
            raise ValueError('Invalid future NWP cache')
        loader = DataLoader(TensorDataset(*tensors, future),
                            batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
                            shuffle=flag == 'train', num_workers=0)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        embeds, loc, scale, targets, sites, times, weather = [value.to(self.device) for value in batch]
        prediction = self.model.forward_features(embeds, loc, scale, weather)
        return prediction, targets, {'site_id': sites, 'forecast_timestamps': times}

    def _checkpoint(self, epoch, validation, optimizer):
        return {
            'epoch': epoch, 'stage': 'b131_linear_weather_v2',
            'experiment_label': 'A02', 'base_experiment_label': 'A01',
            'validation_metrics': validation, 'config': vars(self.args),
            'weather_state_dict': self.model.weather_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'base_provenance': self.model.base_provenance,
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
        }

    def _load_weather(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint['base_provenance'] != self.model.base_provenance:
            raise ValueError('Weather checkpoint was trained on a different base model')
        for key in ('protocol', 'seed', 'seq_len', 'pred_len', 'model', 'adapter_version'):
            if checkpoint['config'][key] != getattr(self.args, key):
                raise ValueError(f'Weather checkpoint configuration differs: {key}')
        self.model.weather_adapter.load_state_dict(checkpoint['weather_state_dict'], strict=True)
        return checkpoint

    def train(self, setting):
        self._prepare_run(setting, 'train')
        if self.args.precision != 'fp32' or self.args.num_workers != 0:
            raise ValueError('Weather Adapter training currently supports FP32 and num_workers=0 only')
        if self.args.train_epochs < 1 or self.args.patience < 1:
            raise ValueError('Weather Adapter training budget and patience must be positive')
        # A failed cache/native identity check must not become a formal run.
        report = Path(__file__).resolve().parents[1] / 'reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012'
        preflight_path = report / 'preflight_gpu.json'
        if not preflight_path.exists():
            raise ValueError('Weather Adapter GPU preflight is required before full training')
        preflight = json.loads(preflight_path.read_text())
        if preflight.get('status') != 'passed' or not preflight.get('full_experiment_ready'):
            raise ValueError('Weather Adapter preflight has unresolved correctness checks')
        checked = next((row for row in preflight['runs'] if row['protocol'] == self.args.protocol), None)
        if checked is None or checked['base_provenance'] != self.model.base_provenance:
            raise ValueError('GPU preflight does not identify this same-protocol frozen base')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        if (directory / 'checkpoint.pth').exists():
            raise FileExistsError('Weather training already has a checkpoint; no implicit restart')
        train_data, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        self._save_config(directory, train_data)
        (directory / 'base_provenance.json').write_text(json.dumps(self.model.base_provenance, indent=2) + '\n')
        optimizer = torch.optim.AdamW(self.model.weather_adapter.parameters(),
                                     lr=self.args.learning_rate, weight_decay=0.05, betas=(0.9, 0.999))
        stopping = EarlyStopping(self.args.patience)
        baseline = self.vali(val_loader)
        stopping.best = baseline['mae']
        # Epoch 0 competes with all trained checkpoints using validation MAE.
        best_path = directory / 'checkpoint.pth'
        torch.save(self._checkpoint(0, baseline, optimizer), best_path)
        rows = [dict(epoch=0, train_mae='', **baseline)]
        stopped = False
        for epoch in range(1, self.args.train_epochs + 1):
            start = time.perf_counter()
            self.model.train()
            total_loss = samples = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, targets, _ = self._forward_batch(batch)
                loss = torch.nn.functional.l1_loss(prediction, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Weather Adapter nonfinite loss at epoch {epoch}')
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in self.model.weather_adapter.parameters()):
                    raise FloatingPointError('Weather Adapter gradient is missing or nonfinite')
                optimizer.step()
                total_loss += loss.item() * len(targets)
                samples += len(targets)
            validation = self.vali(val_loader)
            if not all(np.isfinite(value) for value in validation.values()):
                raise FloatingPointError('Nonfinite Weather Adapter validation metrics')
            rows.append(dict(epoch=epoch, train_mae=total_loss / samples, **validation))
            logging.info('A02 Weather Adapter epoch %d train_mae=%s validation=%s seconds=%.3f',
                         epoch, total_loss / samples, validation, time.perf_counter() - start)
            improved, stopped = stopping.update(validation['mae'])
            checkpoint = self._checkpoint(epoch, validation, optimizer)
            if improved:
                torch.save(checkpoint, best_path)
            with (directory / 'training_history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        best = self._load_weather(best_path)
        (directory / 'training_summary.json').write_text(json.dumps({
            'completed_epochs': len(rows) - 1, 'best_epoch': best['epoch'],
            'baseline_validation_metrics': baseline, 'best_validation_metrics': best['validation_metrics'],
            'stop_reason': 'patience' if stopped else 'max_epochs',
            'selection': 'minimum validation MAE, including epoch0 zero residual',
            'optimizer': 'AdamW', 'learning_rate': self.args.learning_rate,
            'betas': [0.9, 0.999], 'weight_decay': 0.05, 'loss': 'L1',
            'scheduler': None, 'precision': 'FP32',
            'trainable_parameters': sum(p.numel() for p in self.model.weather_adapter.parameters()),
        }, indent=2) + '\n')
        return self.model

    def test(self, setting, test=0):
        if test:
            self._load_weather(self.args.checkpoint)
        return super().test(setting, test=0)
