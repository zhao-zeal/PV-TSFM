"""A03 experiments reuse Dataset_MMSP, frozen features and the shared evaluator."""

import csv
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.Chronos2HistoryWeatherAdapter import Model
from utils.tools import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Exp_Chronos2_History_Weather_Adapter(Exp_Long_Term_Forecast):
    def _build_model(self):
        return Model(self.args).float()

    def __init__(self, args):
        super().__init__(args)
        self.cached_loaders = {}
        self.feature_dir = ROOT / 'outputs' / args.protocol / 'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future'
        self.history_dir = ROOT / 'outputs' / args.protocol / 'Chronos2HistoryWeatherAdapter_inputs_v1_seed42_sl24_pl24'

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, _ = data_provider(self.args, flag)
        source = self.feature_dir / flag
        destination = self.history_dir / flag
        manifest = json.loads((source / 'complete.json').read_text())
        if (manifest['windows'] != len(dataset) or manifest['protocol'] != self.args.protocol
                or manifest['feature_shape'] != [2, self.model.base.backbone.config.d_model]
                or Path(manifest['model_path']).resolve() != Path(self.args.model_path).resolve()):
            raise ValueError('Frozen feature identity differs from A03 inputs')
        # Validate protocol/scalers without rewriting any A01/A02 artifact.
        with tempfile.TemporaryDirectory(prefix='a03_scaler_') as temporary:
            protocol = json.loads(json.dumps(dataset.save_protocol(temporary)))
            if protocol != json.loads((source / 'data_protocol.json').read_text()):
                raise ValueError('A03 data protocol differs from accepted frozen features')
            with np.load(Path(temporary) / 'scalers.npz') as current, np.load(source / 'scalers.npz') as old:
                if set(current.files) != set(old.files) or any(not np.array_equal(current[k], old[k]) for k in old.files):
                    raise ValueError('A03 scaler differs from accepted frozen features')
        tensors = [torch.from_numpy(np.load(source / f'{name}.npy')) for name in
                   ('embeds', 'loc', 'scale', 'targets', 'site_ids', 'forecast_timestamps')]
        if any(len(tensor) != len(dataset) for tensor in tensors):
            raise ValueError('Incomplete frozen cache')
        c = self.model.weather_adapter.weather_dim
        identity = dict(protocol=self.args.protocol, windows=len(dataset),
            source=str(source.resolve()), source_manifest_sha256=file_sha(source / 'complete.json'),
            source_scaler_sha256=file_sha(source / 'scalers.npz'),
            historical_power_shape=[len(dataset), self.args.seq_len, 1],
            historical_nwp_shape=[len(dataset), self.args.seq_len, c],
            future_nwp_shape=[len(dataset), self.args.pred_len, c],
            feature_names=protocol['scaler_fit']['nwp']['feature_names'],
            input='Unchanged Dataset_MMSP values and train-fitted scaler; original interpolation/issuance limitations retained')
        destination.mkdir(parents=True, exist_ok=True)
        complete = destination / 'complete.json'
        names = ('historical_power', 'historical_nwp', 'future_nwp')
        if complete.exists():
            if json.loads(complete.read_text()) != identity:
                raise ValueError('A03 input cache identity differs; no silent reuse')
        else:
            arrays = {name: np.lib.format.open_memmap(destination / f'{name}.npy', mode='w+',
                        dtype=np.float32, shape=tuple(identity[f'{name}_shape'])) for name in names}
            cursor = 0
            for batch in DataLoader(dataset, batch_size=self.args.cache_batch_size, shuffle=False, num_workers=0):
                power, target, _, _, cov = batch
                stop = cursor + len(power)
                for value, expected in ((target[:, -self.args.pred_len:], tensors[3][cursor:stop]),
                    (cov['site_id'], tensors[4][cursor:stop]), (cov['forecast_timestamps'], tensors[5][cursor:stop])):
                    if not torch.equal(value, expected):
                        raise ValueError(f'A03/frozen feature identity differs at window {cursor}')
                if not torch.all(cov['input_end_timestamp'] < cov['forecast_start_timestamp']):
                    raise ValueError('Historical input crosses forecast origin')
                for name, value in zip(names, (power, cov['historical_nwp'], cov['future_nwp'])):
                    if tuple(value.shape[1:]) != tuple(identity[f'{name}_shape'][1:]) or not torch.isfinite(value).all():
                        raise ValueError('Dataset_MMSP input shape or finite check failed')
                    arrays[name][cursor:stop] = value.numpy()
                cursor = stop
            if cursor != len(dataset):
                raise ValueError('Incomplete historical input cache')
            for value in arrays.values():
                value.flush()
            del arrays
            dataset.save_protocol(destination)
            complete.write_text(json.dumps(identity, indent=2) + '\n')
        inputs = [torch.from_numpy(np.load(destination / f'{name}.npy')) for name in names]
        if any(list(v.shape) != identity[f'{name}_shape'] or not torch.isfinite(v).all() for name, v in zip(names, inputs)):
            raise ValueError('A03 cached input is invalid')
        loader = DataLoader(TensorDataset(*tensors, *inputs),
            batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
            shuffle=flag == 'train', num_workers=0)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        embeds, loc, scale, targets, sites, times, power, history, future = [v.to(self.device) for v in batch]
        pred = self.model.forward_features(embeds, loc, scale, future, power, history)
        return pred, targets, dict(site_id=sites, forecast_timestamps=times)

    def _checkpoint(self, epoch, validation, optimizer):
        return dict(epoch=epoch, stage=self.args.adapter_version, config=vars(self.args),
            validation_metrics=validation, weather_state_dict=self.model.weather_adapter.state_dict(),
            relation_state_dict=self.model.history_relation.state_dict(),
            base_provenance=self.model.base_provenance, optimizer_state_dict=optimizer.state_dict(),
            cpu_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(self.device))

    def load_selected(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint['config'] != vars(self.args) or checkpoint['base_provenance'] != self.model.base_provenance:
            raise ValueError('A03 checkpoint/config/frozen base identity differs')
        self.model.weather_adapter.load_state_dict(checkpoint['weather_state_dict'], strict=True)
        self.model.history_relation.load_state_dict(checkpoint['relation_state_dict'], strict=True)
        return checkpoint

    def train(self, setting):
        check = json.loads((REPORT / 'preflight_gpu.json').read_text())
        item = next(row for row in check['variants'] if row['variant'] == self.args.history_variant)
        if check['status'] != 'passed' or item['base_provenance'] != self.model.base_provenance:
            raise ValueError('Same-base A03 GPU correctness checks required before training')
        if check['source_sha256'] != self.args.a03_source_sha256:
            raise ValueError('Code changed after preflight')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        if (directory / 'checkpoint.pth').exists():
            raise FileExistsError('A03 checkpoint exists; do not implicitly restart')
        dataset, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        self._save_config(directory, dataset)
        (directory / 'base_provenance.json').write_text(json.dumps(self.model.base_provenance, indent=2) + '\n')
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay,
                                     betas=tuple(self.args.optimizer_betas))
        stopping = EarlyStopping(self.args.patience)
        baseline = self.vali(val_loader)
        stopping.best = baseline['mae']
        best_path = directory / 'checkpoint.pth'
        torch.save(self._checkpoint(0, baseline, optimizer), best_path)
        rows = [dict(epoch=0, train_mae='', **baseline)]
        stopped = False
        for epoch in range(1, self.args.train_epochs + 1):
            start = time.perf_counter()
            self.model.train()
            loss_sum = samples = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, targets, _ = self._forward_batch(batch)
                loss = torch.nn.functional.l1_loss(prediction, targets)
                if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                    raise FloatingPointError('A03 nonfinite prediction/loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('A03 missing or nonfinite trainable gradient')
                if any(p.grad is not None for p in self.model.base.parameters()):
                    raise RuntimeError('Frozen A01 received gradients')
                optimizer.step()
                loss_sum += loss.item() * len(targets)
                samples += len(targets)
            validation = self.vali(val_loader)
            if not all(np.isfinite(value) for value in validation.values()):
                raise FloatingPointError('A03 nonfinite validation metric')
            rows.append(dict(epoch=epoch, train_mae=loss_sum / samples, **validation))
            improved, stopped = stopping.update(validation['mae'])
            checkpoint = self._checkpoint(epoch, validation, optimizer)
            if improved:
                torch.save(checkpoint, best_path)
            with (directory / 'training_history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            logging.info('A03-%s epoch %d train_mae=%.8f validation=%s seconds=%.3f',
                self.args.history_variant, epoch, loss_sum / samples, validation, time.perf_counter() - start)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        selected = self.load_selected(best_path)
        summary = dict(completed_epochs=len(rows) - 1, best_epoch=selected['epoch'],
            baseline_validation_metrics=baseline, best_validation_metrics=selected['validation_metrics'],
            stop_reason='patience' if stopped else 'max_epochs',
            selection='minimum validation MAE including epoch0 zero residual',
            trainable_parameters=sum(p.numel() for p in params),
            relation_parameters=sum(p.numel() for p in self.model.history_relation.parameters()),
            budget='A02 max100/patience10, batch512/eval1024, FP32, L1, AdamW lr0.000016/wd0.05; no scheduler',
            initialization='A02 Weather Adapter reinitialized seed42; frozen accepted A01 reused; no fitted A02 warm start')
        (directory / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return self.model

    def _record_result(self, directory, metrics):
        logging.info('A03 test artifacts saved at %s; registration awaits artifact acceptance', directory)
