"""A05 cached-token training with a frozen accepted A01 and RMSE selection."""

import copy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.Chronos2MCDecoder import Model
from utils.acceptance import append_validation
from utils.tools import EarlyStopping


class Exp_Chronos2_MC_Decoder(Exp_Long_Term_Forecast):
    names = ('tokens', 'base', 'metadata', 'pcorr_features', 'targets',
             'site_ids', 'forecast_timestamps')

    def _build_model(self):
        return Model(self.args).float()

    def __init__(self, args):
        super().__init__(args)
        self.cached_loaders = {}
        self.cache_dir = Path(args.output_dir) / args.protocol / args.a05_cache_id

    def _cache_identity(self, dataset):
        return dict(
            protocol=self.args.protocol, context=24, horizon=24, windows=len(dataset),
            root_path=str(Path(self.args.root_path).resolve()),
            token_shape=[16, 4, 768], token_dtype='float32',
            pcorr_feature_shape=[16], model_path=self.args.model_path,
            base_provenance=self.model.base_provenance,
            token_layout='PV,NWP15; H0,H1,F0,F1; REG excluded; contextual Chronos-2 states',
            pcorr_layout='Pearson([NWP15,previous-power,current-power]) first16 vs current; negative clipped',
            source_implementation='Chronos2MCDecoder.encode_all_v1')

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, _ = data_provider(self.args, flag)
        directory = self.cache_dir / flag
        complete = directory / 'complete.json'
        expected = self._cache_identity(dataset)
        if complete.exists():
            if json.loads(complete.read_text())['identity'] != expected:
                raise ValueError('A05 token cache identity differs')
        else:
            directory.mkdir(parents=True, exist_ok=False)
            dataset.save_protocol(directory)
            source = (Path(self.args.output_dir) / self.args.protocol /
                      'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future' / flag)
            old = None
            if source.exists():
                if json.loads((directory / 'data_protocol.json').read_text()) != json.loads(
                        (source / 'data_protocol.json').read_text()):
                    raise ValueError('A05 cache protocol differs from accepted A01 cache')
                with np.load(directory / 'scalers.npz') as current, np.load(source / 'scalers.npz') as accepted:
                    if current.files != accepted.files or any(
                            not np.array_equal(current[key], accepted[key]) for key in accepted.files):
                        raise ValueError('A05 scaler differs from accepted A01 cache')
                old = {name: np.load(source / f'{name}.npy', mmap_mode='c') for name in
                       ('embeds', 'loc', 'scale', 'targets', 'site_ids', 'forecast_timestamps')}
            arrays, cursor, max_embed, max_base = {}, 0, 0., 0.
            self.model.eval()
            for batch in DataLoader(dataset, batch_size=self.args.cache_batch_size,
                                    shuffle=False, num_workers=0):
                power, target, _, _, covariates = batch
                stop = cursor + len(power)
                history = covariates['historical_nwp'].to(self.device)
                future = covariates['future_nwp'].to(self.device)
                previous = covariates['correlation_history'].to(self.device)
                if not all(torch.isfinite(value).all() for value in (power, history.cpu(), future.cpu(), previous.cpu())):
                    raise ValueError('A05 cache source contains nonfinite values')
                tokens, base, metadata, pcorr, positions = self.model.encode_all(
                    power.to(self.device), history, future, previous)
                if old is not None:
                    for value, name in ((target[:, -24:], 'targets'),
                                        (covariates['site_id'], 'site_ids'),
                                        (covariates['forecast_timestamps'], 'forecast_timestamps')):
                        if not np.array_equal(value.numpy(), old[name][cursor:stop]):
                            raise ValueError(f'A05/A01 cache window differs: {name} at {cursor}')
                    accepted = [torch.from_numpy(old[name][cursor:stop]).to(self.device)
                                for name in ('embeds', 'loc', 'scale')]
                    reference = self.model.base.forward_features(*accepted)
                    embed_difference = (tokens[:, 0, -2:] - accepted[0]).abs().max().item()
                    base_difference = (base - reference).abs().max().item()
                    if not torch.allclose(tokens[:, 0, -2:], accepted[0], atol=3e-5, rtol=1e-5):
                        raise ValueError(f'A05 changed accepted A01 forecast states: {embed_difference}')
                    if base_difference > 2e-6:
                        raise ValueError(f'A05 changed accepted A01 prediction: {base_difference}')
                    max_embed, max_base = max(max_embed, embed_difference), max(max_base, base_difference)
                values = (tokens, base, metadata, pcorr, target[:, -24:],
                          covariates['site_id'], covariates['forecast_timestamps'])
                for name, value in zip(self.names, values):
                    value = value.cpu().numpy()
                    if not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite A05 cache array: {name}')
                    if cursor == 0:
                        arrays[name] = np.lib.format.open_memmap(
                            directory / f'{name}.npy', mode='w+', dtype=value.dtype,
                            shape=(len(dataset), *value.shape[1:]))
                    arrays[name][cursor:stop] = value
                cursor = stop
                if cursor % 6400 == 0:
                    logging.info('A05 cache %s/%s %d/%d', self.args.protocol, flag, cursor, len(dataset))
            if cursor != len(dataset):
                raise ValueError('Incomplete A05 cache')
            for array in arrays.values():
                array.flush()
            del arrays
            np.save(directory / 'positions.npy', positions.cpu().numpy())
            complete.write_text(json.dumps(dict(
                identity=expected, all_windows_checked=True,
                accepted_A01_cache_compared=old is not None,
                max_A01_embed_difference=max_embed, max_A01_prediction_difference=max_base,
                future_power_unobserved=True), indent=2) + '\n')
        tensors = [torch.from_numpy(np.load(directory / f'{name}.npy', mmap_mode='c'))
                   for name in self.names]
        if not all(len(value) == len(dataset) for value in tensors):
            raise ValueError('Incomplete A05 cached tensor set')
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
            shuffle=flag == 'train', num_workers=0,
            generator=torch.Generator().manual_seed(self.args.seed))
        self.positions = torch.from_numpy(np.load(directory / 'positions.npy')).to(self.device)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        tokens, base, metadata, pcorr, targets, sites, times = [value.to(self.device) for value in batch]
        prediction = self.model.forward_cached(tokens, base, metadata, pcorr, self.positions)
        return prediction, targets, dict(site_id=sites, forecast_timestamps=times)

    def check_initialization(self, dataset, loader, directory):
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        raw = next(iter(DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=3, num_workers=0)))
        power, targets, _, _, covariates = raw
        self.model.eval()
        with torch.no_grad():
            native = self.model(power.to(self.device), None, None, None,
                                {key: value.to(self.device) for key, value in covariates.items()})
            cached_batch = [value[indices] for value in loader.dataset.tensors]
            cached, truth, _ = self._forward_batch(cached_batch)
        if not torch.allclose(native, cached, atol=2e-6, rtol=0):
            raise ValueError('A05 native and cached paths differ')
        if not torch.equal(cached, cached_batch[1].to(self.device)):
            raise ValueError('Zero-initialized A05 residual does not exactly equal A01')
        if not torch.equal(truth.cpu(), targets[:, -24:]):
            raise ValueError('A05 cached targets differ from raw dataset')
        if not torch.equal(self.positions.cpu(), torch.tensor([-32., -16., 0., 16.])):
            raise ValueError('A05 native patch positions changed')
        initial = copy.deepcopy(self.model.a05_head.state_dict())
        optimizer = torch.optim.AdamW(self.model.a05_head.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)
        gradients = {}
        self.model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction, truth, _ = self._forward_batch(cached_batch)
            torch.nn.functional.mse_loss(prediction, truth).backward()
            for name, parameter in self.model.a05_head.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f'A05 missing/nonfinite gradient: {name}')
                gradients[name] = max(gradients.get(name, 0.), parameter.grad.abs().max().item())
            if any(parameter.grad is not None for parameter in self.model.base.parameters()):
                raise ValueError('Frozen A01 received gradients')
            optimizer.step()
        missing = [name for name, value in gradients.items() if value == 0]
        if missing:
            raise ValueError(f'A05 parameters received only zero gradients: {missing}')
        self.model.a05_head.load_state_dict(initial)
        self.model.a05_head.zero_grad(set_to_none=True)
        receipt = dict(
            status='passed', indices=indices,
            native_cache_max_abs=(native - cached).abs().max().item(),
            zero_residual_equals_A01=True, frozen_A01_gradients_absent=True,
            all_trainable_parameters_receive_nonzero_gradient_within_three_steps=True,
            token_shape=list(cached_batch[0].shape), pcorr_shape=list(cached_batch[3].shape),
            positions=self.positions.cpu().tolist())
        if directory is not None:
            (directory / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt

    def _checkpoint(self, epoch, validation, optimizer):
        return dict(
            epoch=epoch, stage=f'A05-{self.args.a05_variant}', config=vars(self.args),
            validation_metrics=validation, a05_state_dict=self.model.a05_head.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), base_provenance=self.model.base_provenance,
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=(torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None))

    def load_selected(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint['base_provenance'] != self.model.base_provenance:
            raise ValueError('A05 checkpoint uses a different frozen A01')
        for key in ('model', 'protocol', 'seed', 'seq_len', 'pred_len', 'a05_variant',
                    'a05_width', 'a05_heads', 'a05_ff'):
            if checkpoint['config'][key] != getattr(self.args, key):
                raise ValueError(f'A05 checkpoint configuration differs: {key}')
        self.model.a05_head.load_state_dict(checkpoint['a05_state_dict'], strict=True)
        return checkpoint

    def train(self, setting):
        self._prepare_run(setting, 'train')
        if self.args.precision != 'fp32' or self.args.num_workers != 0:
            raise ValueError('A05 supports FP32 and num_workers=0 only')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        if (directory / 'checkpoint.pth').exists():
            raise FileExistsError('A05 checkpoint exists; no implicit restart')
        dataset, train_loader = self._get_data('train')
        _, validation_loader = self._get_data('val')
        self._save_config(directory, dataset)
        preflight = self.check_initialization(dataset, train_loader, directory)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        train_loader.generator.manual_seed(self.args.seed)
        optimizer = torch.optim.AdamW(
            self.model.a05_head.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay, betas=tuple(self.args.optimizer_betas))
        baseline = self.vali(validation_loader)
        stopping = EarlyStopping(self.args.patience)
        stopping.best = baseline['rmse']
        best_path = directory / 'checkpoint.pth'
        torch.save(self._checkpoint(0, baseline, optimizer), best_path)
        append_validation(directory, 0, baseline)
        logging.info('A05-%s epoch 0 validation=%s', self.args.a05_variant, baseline)
        stopped, checkpoint = False, None
        for epoch in range(1, self.args.train_epochs + 1):
            started = time.perf_counter()
            self.model.train()
            total, count = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, truth, _ = self._forward_batch(batch)
                loss = torch.nn.functional.mse_loss(prediction, truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite A05 training loss')
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in self.model.a05_head.parameters()):
                    raise FloatingPointError('Missing/nonfinite A05 gradient')
                optimizer.step()
                total += loss.item() * len(truth)
                count += len(truth)
            validation = self.vali(validation_loader)
            append_validation(directory, epoch, validation, total / count)
            improved, stopped = stopping.update(validation['rmse'])
            checkpoint = self._checkpoint(epoch, validation, optimizer)
            if improved:
                torch.save(checkpoint, best_path)
            logging.info('A05-%s epoch %d train_mse=%.8f validation=%s seconds=%.2f',
                         self.args.a05_variant, epoch, total / count, validation,
                         time.perf_counter() - started)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        selected = self.load_selected(best_path)
        summary = dict(
            best_epoch=selected['epoch'], completed_epochs=epoch,
            baseline_validation_metrics=baseline,
            best_validation_metrics=selected['validation_metrics'],
            stop_reason='patience' if stopped else 'max_epochs',
            trainable_parameters=sum(p.numel() for p in self.model.a05_head.parameters()),
            selection='minimum validation RMSE including zero-residual epoch 0',
            loss='MSE', optimizer='AdamW', learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay, preflight=preflight)
        (directory / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return self.model

    def test(self, setting, test=0):
        if test:
            self.load_selected(self.args.checkpoint)
        return super().test(setting, test=0)

    def _record_result(self, directory, metrics, *, specific_checks=None):
        from utils.metrics import metric

        directory = Path(directory)
        prediction = np.load(directory / 'pred.npy')
        truth = np.load(directory / 'true.npy')
        sites = np.load(directory / 'site_ids.npy')
        times = np.load(directory / 'forecast_timestamps.npy')
        dataset, _ = data_provider(self.args, 'test')
        cursor, max_difference = 0, 0.
        self.model.eval()
        with torch.no_grad():
            for power, target, _, _, covariates in DataLoader(
                    dataset, batch_size=self.args.cache_batch_size, shuffle=False, num_workers=0):
                replay = self.model(
                    power.to(self.device), None, None, None,
                    {key: value.to(self.device) for key, value in covariates.items()}).cpu().numpy()
                stop = cursor + len(power)
                difference = float(np.abs(replay - prediction[cursor:stop]).max())
                if difference > 2e-6:
                    raise ValueError(f'A05 raw replay differs at {cursor}: {difference}')
                if not np.array_equal(target[:, -24:].numpy(), truth[cursor:stop]):
                    raise ValueError(f'A05 raw labels differ at {cursor}')
                if not np.array_equal(covariates['site_id'].numpy(), sites[cursor:stop]):
                    raise ValueError(f'A05 raw sites differ at {cursor}')
                if not np.array_equal(covariates['forecast_timestamps'].numpy(), times[cursor:stop]):
                    raise ValueError(f'A05 raw timestamps differ at {cursor}')
                cursor, max_difference = stop, max(max_difference, difference)
        if cursor != len(prediction):
            raise ValueError('A05 raw replay has incomplete test coverage')
        base_dir = (Path(self.args.output_dir) / self.args.protocol /
                    'Chronos2Adapter_b131_linear_v1_seed42_sl24_pl24_history_future')
        base = np.load(base_dir / 'pred.npy')
        for name, value in (('true', truth), ('site_ids', sites), ('forecast_timestamps', times)):
            if not np.array_equal(value, np.load(base_dir / f'{name}.npy')):
                raise ValueError(f'A05 and accepted A01 differ: {name}')
        current_metrics, base_metrics = metric(prediction, truth), metric(base, truth)
        analysis = dict(
            current=current_metrics, A01=base_metrics,
            change_from_A01={
                key: dict(absolute=current_metrics[key] - base_metrics[key],
                          relative_percent=100 * (current_metrics[key] / base_metrics[key] - 1))
                for key in ('mae', 'rmse')},
            variant=self.args.a05_variant,
            limits=['seed42 only', 'future NWP issue time is absent from MMSP source CSV'])
        (directory / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
        checks = dict(
            full_raw_selected_checkpoint_replay_windows=cursor,
            max_abs_difference=max_difference,
            same_A01_labels_sites_times=True,
            frozen_A01_gradients_absent=all(parameter.grad is None for parameter in self.model.base.parameters()),
            preflight=str((Path(self.args.checkpoints) / self.args.protocol /
                           self.args.model_id / 'preflight.json').resolve()))
        super()._record_result(directory, metrics, specific_checks=checks)
