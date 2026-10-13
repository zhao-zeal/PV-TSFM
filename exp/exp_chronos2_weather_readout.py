"""A04 uses the existing common dataset, cache checks and formal evaluator."""

import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter, file_sha
from models.Chronos2WeatherReadout import Model
from utils.tools import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/Chronos2WeatherReadout_two_protocols_seed42_20261012'


class Exp_Chronos2_Weather_Readout(Exp_Chronos2_History_Weather_Adapter):
    def _build_model(self):
        return Model(self.args).float()

    def __init__(self, args):
        super().__init__(args)
        if args.protocol == 'in_domain':
            self.history_dir = ROOT / 'outputs' / args.protocol / 'Chronos2WeatherReadout_inputs_v1_seed42_sl24_pl24'

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, loader = super()._get_data(flag)
        if self.args.memory_variant == 'a':
            return dataset, loader
        source = self.feature_dir / flag
        directory = ROOT / 'outputs' / self.args.protocol / 'Chronos2WeatherReadout_context_v1_seed42_sl24_pl24' / flag
        directory.mkdir(parents=True, exist_ok=True)
        complete = directory / 'complete.json'
        tensors = loader.dataset.tensors
        count = (self.args.seq_len + self.model.base.backbone.chronos_config.input_patch_size - 1) // self.model.base.backbone.chronos_config.input_patch_size
        shape = [len(dataset), count, self.model.base.backbone.config.d_model]
        identity = dict(protocol=self.args.protocol, windows=len(dataset), context_shape=shape,
                        feature_manifest_sha256=file_sha(source / 'complete.json'),
                        input_manifest_sha256=file_sha(self.history_dir / flag / 'complete.json'),
                        model_path=str(Path(self.args.model_path).resolve()),
                        method='Same A01 encode with two forecast patches; power context positions only; REG excluded; contextual states include known future NWP')
        if complete.exists():
            previous = json.loads(complete.read_text())
            if previous['identity'] != identity:
                raise ValueError('A04 context cache identity differs')
        else:
            contexts = np.lib.format.open_memmap(directory / 'context.npy', mode='w+', dtype=np.float32, shape=tuple(shape))
            max_embed_difference = max_base_difference = 0.
            cursor = 0
            self.model.eval()
            with torch.no_grad():
                for batch in DataLoader(dataset, batch_size=self.args.cache_batch_size, shuffle=False, num_workers=0):
                    power, target, _, _, cov = batch
                    stop = cursor + len(power)
                    for value, expected in ((target[:, -self.args.pred_len:], tensors[3][cursor:stop]),
                            (cov['site_id'], tensors[4][cursor:stop]), (cov['forecast_timestamps'], tensors[5][cursor:stop]),
                            (power, tensors[6][cursor:stop]), (cov['historical_nwp'], tensors[7][cursor:stop]),
                            (cov['future_nwp'], tensors[8][cursor:stop])):
                        if not torch.equal(value, expected):
                            raise ValueError(f'A04 context/input/window identity differs at {cursor}')
                    if not torch.all(cov['input_end_timestamp'] < cov['forecast_start_timestamp']):
                        raise ValueError('Historical input crosses forecast origin')
                    values = self.model.encode_memory(power[..., 0].to(self.device),
                        cov['historical_nwp'].to(self.device), cov['future_nwp'].to(self.device))
                    embeds, loc, scale, context = values
                    cached = [v[cursor:stop].to(self.device) for v in tensors[:3]]
                    difference = float((embeds - cached[0]).abs().max())
                    base_difference = float((self.model.base.forward_features(embeds, loc, scale) -
                        self.model.base.forward_features(*cached)).abs().max())
                    if not torch.allclose(embeds, cached[0], atol=3e-5, rtol=1e-5) or base_difference > 2e-6:
                        raise ValueError(f'A04 encode changed accepted A01 states: {difference}, {base_difference}')
                    if context.shape != (len(power), *shape[1:]) or not torch.isfinite(context).all():
                        raise ValueError('Invalid A04 context shape/values')
                    max_embed_difference = max(max_embed_difference, difference)
                    max_base_difference = max(max_base_difference, base_difference)
                    contexts[cursor:stop] = context.cpu().numpy()
                    cursor = stop
                    if cursor % (self.args.cache_batch_size * 100) == 0:
                        print(f'A04 context {self.args.protocol}/{flag}: {cursor}/{len(dataset)}', flush=True)
            assert cursor == len(dataset)
            contexts.flush()
            del contexts
            save = dict(identity=identity, all_windows_inputs_targets_sites_times_and_past_boundary_checked=True,
                        max_forecast_embed_difference=max_embed_difference, max_A01_prediction_difference=max_base_difference)
            complete.write_text(json.dumps(save, indent=2) + '\n')
        context = torch.from_numpy(np.load(directory / 'context.npy'))
        if list(context.shape) != shape or not torch.isfinite(context).all():
            raise ValueError('Invalid A04 contextual memory cache')
        loader = DataLoader(TensorDataset(*tensors, context),
                            batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
                            shuffle=flag == 'train', num_workers=0)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        values = [v.to(self.device) for v in batch]
        embeds, loc, scale, targets, sites, times, power, history, future = values[:9]
        context = values[9] if len(values) == 10 else None
        pred = self.model.forward_features(embeds, loc, scale, future, context)
        return pred, targets, dict(site_id=sites, forecast_timestamps=times)

    def _checkpoint(self, epoch, validation, optimizer):
        return dict(epoch=epoch, stage=self.args.adapter_version, config=vars(self.args),
            validation_metrics=validation, weather_state_dict=self.model.weather_adapter.state_dict(),
            base_provenance=self.model.base_provenance, optimizer_state_dict=optimizer.state_dict(),
            cpu_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(self.device))

    def load_selected(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint['config'] != vars(self.args) or checkpoint['base_provenance'] != self.model.base_provenance:
            raise ValueError('A04 checkpoint/config/frozen base identity differs')
        self.model.weather_adapter.load_state_dict(checkpoint['weather_state_dict'], strict=True)
        return checkpoint

    def train(self, setting):
        check = json.loads((REPORT / 'preflight_gpu.json').read_text())
        item = next(row for row in check['variants'] if row['variant'] == self.args.memory_variant and row['protocol'] == self.args.protocol)
        if check['status'] != 'passed' or item['base_provenance'] != self.model.base_provenance:
            raise ValueError('Same-base A04 GPU correctness checks required before training')
        if check['source_sha256'] != self.args.a04_source_sha256:
            raise ValueError('Code changed after preflight')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        if (directory / 'checkpoint.pth').exists():
            raise FileExistsError('A04 checkpoint exists; do not implicitly restart')
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
                    raise FloatingPointError('A04 nonfinite prediction/loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('A04 missing or nonfinite trainable gradient')
                if any(p.grad is not None for p in self.model.base.parameters()):
                    raise RuntimeError('Frozen A01 received gradients')
                optimizer.step()
                loss_sum += loss.item() * len(targets)
                samples += len(targets)
            validation = self.vali(val_loader)
            if not all(np.isfinite(value) for value in validation.values()):
                raise FloatingPointError('A04 nonfinite validation metric')
            rows.append(dict(epoch=epoch, train_mae=loss_sum / samples, **validation))
            improved, stopped = stopping.update(validation['mae'])
            checkpoint = self._checkpoint(epoch, validation, optimizer)
            if improved:
                torch.save(checkpoint, best_path)
            with (directory / 'training_history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            logging.info('A04-%s epoch %d train_mae=%.8f validation=%s seconds=%.3f',
                self.args.memory_variant, epoch, loss_sum / samples, validation, time.perf_counter() - start)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        selected = self.load_selected(best_path)
        summary = dict(completed_epochs=len(rows) - 1, best_epoch=selected['epoch'],
            baseline_validation_metrics=baseline, best_validation_metrics=selected['validation_metrics'],
            stop_reason='patience' if stopped else 'max_epochs',
            selection='minimum validation MAE including epoch0 zero residual',
            trainable_parameters=sum(p.numel() for p in params),
            budget='A02 max100/patience10, batch512/eval1024, FP32, L1, AdamW lr0.000016/wd0.05; no scheduler',
            initialization='A02 Weather Adapter reinitialized seed42; frozen accepted A01 reused; no fitted A02 warm start')
        (directory / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return self.model

    def _record_result(self, directory, metrics):
        logging.info('A04 test artifacts saved at %s; registration awaits artifact acceptance', directory)
