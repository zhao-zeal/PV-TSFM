"""E11/E12 staged training; Dataset_MMSP defines every cached input and label."""

import csv
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.chronos2_adapter import ForecastOnlyBottleneckAdapter, residual_readout
from utils.tools import EarlyStopping


class Exp_Chronos2_Adapter(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super().__init__(args)
        self.cached_loaders = {}
        subset = f'_{args.nwp_subset_id}' if getattr(args, 'nwp_subset_id', None) else ''
        self.cache_dir = Path(args.output_dir) / args.protocol / (
            f'Chronos2Adapter_features_v1_seed{args.seed}_sl24_pl24_history_future{subset}')
        self.base_dir = Path(args.checkpoints) / args.protocol / (
            f'Chronos2Adapter_B131_v1_seed{args.seed}_sl24_pl24_history_future{subset}')

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, _ = data_provider(self.args, flag)
        cache = self.cache_dir / flag
        cache.mkdir(parents=True, exist_ok=True)
        manifest = cache / 'complete.json'
        names = ('embeds', 'loc', 'scale', 'targets', 'site_ids', 'forecast_timestamps')
        if not manifest.exists():
            dataset.save_protocol(cache)
            arrays = {}
            shapes = ((2, 768), (1,), (1,), (24, 1), (), (24,))
            for name, shape in zip(names, shapes):
                dtype = np.int64 if name in ('site_ids', 'forecast_timestamps') else np.float32
                arrays[name] = np.lib.format.open_memmap(
                    cache / f'{name}.npy', mode='w+', dtype=dtype, shape=(len(dataset), *shape))
            loader = DataLoader(dataset, batch_size=self.args.cache_batch_size,
                                shuffle=False, num_workers=0)
            self.model.eval()
            cursor = 0
            for index, batch in enumerate(loader):
                power, _, _, _, covariates, targets = self._prepare_batch(batch)
                with torch.no_grad():
                    features = self.model.encode_inputs(
                        power[..., 0], covariates['historical_nwp'], covariates['future_nwp'])
                values = (*features, targets, covariates['site_id'], covariates['forecast_timestamps'])
                stop = cursor + len(power)
                for name, value in zip(names, values):
                    arrays[name][cursor:stop] = value.detach().cpu().numpy()
                cursor = stop
                if index % 100 == 0 or cursor == len(dataset):
                    logging.info('Chronos feature cache %s %d/%d', flag, cursor, len(dataset))
            for value in arrays.values():
                value.flush()
            manifest.write_text(json.dumps({
                'windows': cursor, 'protocol': self.args.protocol, 'model_path': self.args.model_path,
                'input': 'power + selected train-standardized historical/future NWP in Dataset_MMSP order',
                'nwp_indices': getattr(self.args, 'nwp_indices', None),
                'nwp_feature_names': getattr(self.args, 'nwp_feature_names', None),
                'chronos_version': '2.2.2', 'feature_shape': [2, 768],
            }, indent=2) + '\n')
        tensors = [torch.from_numpy(np.load(cache / f'{name}.npy')) for name in names]
        loader = DataLoader(TensorDataset(*tensors),
                            batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
                            shuffle=flag == 'train', num_workers=0)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        embeds, loc, scale, targets, sites, times = [value.to(self.device) for value in batch]
        prediction = self.model.forward_features(embeds, loc, scale)
        return prediction, targets, {'site_id': sites, 'forecast_timestamps': times}

    def _seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def _train_stage(self, module, train_loader, val_loader, directory, learning_rate, stage):
        directory.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate,
                                     weight_decay=0.05, betas=(0.9, 0.999))
        stopping = EarlyStopping(self.args.patience)
        rows = []
        path = directory / 'checkpoint.pth'
        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            total_loss = samples = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, targets, _ = self._forward_batch(batch)
                loss = torch.nn.functional.l1_loss(prediction, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{stage} nonfinite loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
                samples += len(targets)
            validation = self.vali(val_loader)
            row = dict(epoch=epoch, train_mae=total_loss / samples, **validation)
            rows.append(row)
            logging.info('Adapter %s epoch %d train=%s validation=%s', stage, epoch,
                         row['train_mae'], validation)
            improved, stopped = stopping.update(validation['mae'])
            checkpoint = dict(epoch=epoch, stage=stage, validation_metrics=validation,
                              adapter_state_dict=self.model.adapter.state_dict(),
                              optimizer_state_dict=optimizer.state_dict(), config=vars(self.args))
            if stage != 'base':
                checkpoint['readout_state_dict'] = self.model.readout.state_dict()
            if improved:
                torch.save(checkpoint, path)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        with (directory / 'training_history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selected = torch.load(path, map_location=self.device, weights_only=True)
        self.model.adapter.load_state_dict(selected['adapter_state_dict'], strict=True)
        if stage != 'base':
            self.model.readout.load_state_dict(selected['readout_state_dict'], strict=True)
        (directory / 'training_summary.json').write_text(json.dumps({
            'stage': stage, 'completed_epochs': len(rows), 'best_epoch': selected['epoch'],
            'best_validation_metrics': selected['validation_metrics'],
            'stop_reason': 'patience' if stopped else 'max_epochs',
            'learning_rate': learning_rate, 'optimizer': 'AdamW', 'betas': [0.9, 0.999],
            'weight_decay': 0.05, 'loss': 'L1', 'scheduler': None,
            'trainable_parameters': sum(p.numel() for p in module.parameters()),
        }, indent=2) + '\n')

    def train(self, setting):
        self._prepare_run(setting, 'train')
        train_data, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        self.model.stage = 'base'
        self._seed()
        self.model.adapter = ForecastOnlyBottleneckAdapter().to(self.device)
        self.model.readout.requires_grad_(False)
        base_path = self.base_dir / 'checkpoint.pth'
        if base_path.exists():
            if not (self.base_dir / 'training_summary.json').exists():
                raise ValueError('Shared B131 training is incomplete; cannot reuse its interim checkpoint')
            checkpoint = torch.load(base_path, map_location=self.device, weights_only=True)
            for key in ('protocol', 'seed', 'seq_len', 'pred_len', 'modalities', 'nwp_mode',
                        'model_path', 'train_epochs', 'patience', 'batch_size', 'effective_precision'):
                if checkpoint['config'].get(key) != getattr(self.args, key, None):
                    raise ValueError(f'Shared B131 training configuration differs: {key}')
            self.model.adapter.load_state_dict(checkpoint['adapter_state_dict'], strict=True)
            logging.info('Reuse current-protocol B131 checkpoint: %s', base_path)
        else:
            self.base_dir.mkdir(parents=True)  # Never overwrite an incomplete/concurrent base run.
            self._save_config(self.base_dir, train_data)
            self._train_stage(self.model.adapter, train_loader, val_loader,
                              self.base_dir, 0.0016, 'base')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        self._save_config(directory, train_data)
        self.model.stage = self.args.adapter_version
        self.model.adapter.requires_grad_(False)
        self._seed()
        self.model.readout = residual_readout(self.args.adapter_version).to(self.device)
        self._train_stage(self.model.readout, train_loader, val_loader, directory,
                          self.args.learning_rate, self.args.adapter_version)
        return self.model

    def test(self, setting, test=0):
        if test:
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device, weights_only=True)
            self.model.adapter.load_state_dict(checkpoint['adapter_state_dict'], strict=True)
            self.model.readout.load_state_dict(checkpoint['readout_state_dict'], strict=True)
            self.model.stage = self.args.adapter_version
            self.model.adapter.requires_grad_(False)
        return super().test(setting, test=0)
