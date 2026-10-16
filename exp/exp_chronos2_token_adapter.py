"""Stage-1 controlled token experiments using the shared data and acceptance path."""

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
from models.Chronos2TokenAdapter import Model
from utils.acceptance import append_validation
from utils.tools import EarlyStopping


class Exp_Chronos2_Token_Adapter(Exp_Long_Term_Forecast):
    names = ('tokens', 'base', 'future', 'metadata', 'correlation', 'targets', 'site_ids', 'forecast_timestamps')

    def _build_model(self):
        return Model(self.args).float()

    def __init__(self, args):
        super().__init__(args)
        self.cached_loaders = {}
        self.cache_dir = Path(args.output_dir) / args.protocol / args.token_cache_id

    def _get_data(self, flag):
        if flag in self.cached_loaders:
            return self.cached_loaders[flag]
        dataset, _ = data_provider(self.args, flag)
        directory = self.cache_dir / flag
        complete = directory / 'complete.json'
        expected = dict(protocol=self.args.protocol, context=self.args.seq_len, horizon=24,
                        windows=len(dataset), shape=[16, 4, 768], dtype='float32',
                        model_path=self.args.model_path, base_provenance=self.model.base_provenance,
                        layout='PV,NWP15; history2,future2; REG excluded; native contextual states',
                        source_implementation='Chronos2TokenAdapter.encode_all_v1')
        if complete.exists():
            if json.loads(complete.read_text())['identity'] != expected:
                raise ValueError('Full-token cache identity differs')
        else:
            directory.mkdir(parents=True, exist_ok=False)
            dataset.save_protocol(directory)
            source = Path(self.args.output_dir) / self.args.protocol / 'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future' / flag
            if json.loads((directory / 'data_protocol.json').read_text()) != json.loads((source / 'data_protocol.json').read_text()):
                raise ValueError('Token cache protocol differs from accepted A01 cache')
            with np.load(directory / 'scalers.npz') as current, np.load(source / 'scalers.npz') as old:
                assert current.files == old.files
                assert all(np.array_equal(current[k], old[k]) for k in old.files)
            old = {k: np.load(source / f'{k}.npy', mmap_mode='c') for k in
                   ('embeds', 'loc', 'scale', 'targets', 'site_ids', 'forecast_timestamps')}
            arrays, cursor, max_embed, max_base = {}, 0, 0., 0.
            self.model.eval()
            for batch in DataLoader(dataset, batch_size=self.args.cache_batch_size, shuffle=False, num_workers=0):
                power, target, _, _, cov = batch
                stop = cursor + len(power)
                for value, name in ((target[:, -24:], 'targets'), (cov['site_id'], 'site_ids'),
                                    (cov['forecast_timestamps'], 'forecast_timestamps')):
                    assert np.array_equal(value.numpy(), old[name][cursor:stop]), name
                assert (cov['input_end_timestamp'] < cov['forecast_start_timestamp']).all()
                history, future = cov['historical_nwp'].to(self.device), cov['future_nwp'].to(self.device)
                assert torch.isfinite(history).all() and torch.isfinite(future).all()
                tokens, base, metadata, corr, positions = self.model.encode_all(power.to(self.device), history, future)
                original = [torch.from_numpy(old[k][cursor:stop]).to(self.device) for k in ('embeds', 'loc', 'scale')]
                with torch.no_grad():
                    reference = self.model.base.forward_features(*original)
                difference = (tokens[:, 0, -2:] - original[0]).abs().max().item()
                base_difference = (base - reference).abs().max().item()
                assert torch.allclose(tokens[:, 0, -2:], original[0], atol=3e-5, rtol=1e-5), difference
                assert base_difference <= 2e-6, base_difference
                max_embed, max_base = max(max_embed, difference), max(max_base, base_difference)
                values = (tokens, base, future, metadata, corr, target[:, -24:], cov['site_id'], cov['forecast_timestamps'])
                for name, value in zip(self.names, values):
                    value = value.cpu().numpy()
                    assert np.isfinite(value).all(), name
                    if cursor == 0:
                        arrays[name] = np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+',
                            dtype=value.dtype, shape=(len(dataset), *value.shape[1:]))
                    arrays[name][cursor:stop] = value
                cursor = stop
                if cursor % 6400 == 0:
                    logging.info('Native full-token cache %s/%s %d/%d', self.args.protocol, flag, cursor, len(dataset))
            assert cursor == len(dataset)
            for array in arrays.values():
                array.flush()
            del arrays
            np.save(directory / 'positions.npy', positions.cpu().numpy())
            complete.write_text(json.dumps(dict(identity=expected, all_windows_checked=True,
                max_A01_embed_difference=max_embed, max_A01_prediction_difference=max_base,
                future_power_unobserved=True, labels_sites_times_scalers_match_A01=True), indent=2) + '\n')
        tensors = [torch.from_numpy(np.load(directory / f'{name}.npy', mmap_mode='c')) for name in self.names]
        assert all(len(value) == len(dataset) for value in tensors)
        loader = DataLoader(TensorDataset(*tensors), batch_size=self.args.batch_size if flag == 'train' else self.args.eval_batch_size,
                            shuffle=flag == 'train', num_workers=0, generator=torch.Generator().manual_seed(self.args.seed))
        self.positions = torch.from_numpy(np.load(directory / 'positions.npy')).to(self.device)
        self.cached_loaders[flag] = dataset, loader
        return dataset, loader

    def _forward_batch(self, batch):
        if self.args.token_variant == 'T1':
            batch = list(batch)
            batch[0] = batch[0][:, :1, -2:]
            batch[3] = batch[3][:, :1, -2:]
        tokens, base, future, metadata, corr, targets, sites, times = [v.to(self.device) for v in batch]
        pred = self.model.forward_cached(tokens, base, future, metadata, corr, self.positions)
        return pred, targets, dict(site_id=sites, forecast_timestamps=times)

    def check_initialization(self, dataset, loader, directory):
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        raw = next(iter(DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=3, num_workers=0)))
        power, targets, _, _, cov = raw
        self.model.eval()
        with torch.no_grad():
            native = self.model(power.to(self.device), None, None, None,
                {k: v.to(self.device) for k, v in cov.items()})
            batch = [v[indices] for v in loader.dataset.tensors]
            cached, truth, _ = self._forward_batch(batch)
        assert torch.allclose(native, cached, atol=2e-6, rtol=0)
        assert torch.equal(cached, batch[1].to(self.device)), 'Zero residual must exactly equal A01'
        assert torch.equal(truth.cpu(), targets[:, -24:])
        metadata = batch[3]
        assert torch.equal(metadata[0, 0, :, 1], torch.tensor([.5, 1., 1., .5]))
        assert torch.equal(metadata[0, 0, :, 2], torch.tensor([.5, 1., 0., 0.]))
        assert torch.equal(self.positions.cpu(), torch.tensor([-32., -16., 0., 16.]))
        frozen = {k: v.detach().cpu().clone() for k, v in self.model.base.state_dict().items()}
        state = copy.deepcopy(self.model.weather_adapter.state_dict())
        optimizer = torch.optim.AdamW(self.model.weather_adapter.parameters(), lr=self.args.learning_rate, weight_decay=.05)
        gradients = {}
        self.model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            pred, truth, _ = self._forward_batch(batch)
            torch.nn.functional.l1_loss(pred, truth).backward()
            for name, param in self.model.weather_adapter.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
                gradients[name] = max(gradients.get(name, 0.), param.grad.abs().max().item())
            assert all(p.grad is None for p in self.model.base.parameters())
            optimizer.step()
        assert all(value > 0 for value in gradients.values()), gradients
        assert all(torch.equal(value, self.model.base.state_dict()[key].cpu()) for key, value in frozen.items())
        self.model.weather_adapter.load_state_dict(state)
        self.model.weather_adapter.zero_grad(set_to_none=True)
        receipt = dict(status='passed', indices=indices, native_cache_max_abs=(native-cached).abs().max().item(),
                       zero_residual_equals_A01=True, frozen_weights_unchanged=True,
                       all_trainable_parameters_receive_nonzero_gradient_within_three_steps=True,
                       gradient_maxima=gradients, patch_padding_metadata_checked=True)
        if directory is not None:
            (directory / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt

    def _checkpoint(self, epoch, validation, optimizer):
        return dict(epoch=epoch, config=vars(self.args), validation_metrics=validation,
                    weather_state_dict=self.model.weather_adapter.state_dict(), base_provenance=self.model.base_provenance,
                    optimizer_state_dict=optimizer.state_dict(), cpu_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state(self.device))

    def load_selected(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        assert checkpoint['config'] == vars(self.args)
        assert checkpoint['base_provenance'] == self.model.base_provenance
        self.model.weather_adapter.load_state_dict(checkpoint['weather_state_dict'])
        return checkpoint

    def train(self, setting):
        self._prepare_run(setting, 'train')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        data, train = self._get_data('train')
        _, val = self._get_data('val')
        self._save_config(directory, data)
        self.check_initialization(data, train, directory)
        frozen_start = {k: v.detach().cpu().clone() for k, v in self.model.base.state_dict().items()}
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        optimizer = torch.optim.AdamW(self.model.weather_adapter.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay, betas=tuple(self.args.optimizer_betas))
        baseline = self.vali(val)
        stopping = EarlyStopping(self.args.patience)
        stopping.best = baseline['mae']
        best_path = directory / 'checkpoint.pth'
        torch.save(self._checkpoint(0, baseline, optimizer), best_path)
        append_validation(directory, 0, baseline)
        logging.info('%s epoch 0 validation=%s', self.args.token_variant, baseline)
        for epoch in range(1, self.args.train_epochs + 1):
            start = time.perf_counter()
            self.model.train()
            total, count = 0., 0
            for batch in train:
                optimizer.zero_grad(set_to_none=True)
                prediction, truth, _ = self._forward_batch(batch)
                loss = torch.nn.functional.l1_loss(prediction, truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.weather_adapter.parameters()):
                    raise FloatingPointError('Missing/nonfinite residual gradient')
                optimizer.step()
                total += loss.item() * len(truth)
                count += len(truth)
            validation = self.vali(val)
            append_validation(directory, epoch, validation, total / count)
            improved, stopped = stopping.update(validation['mae'])
            checkpoint = self._checkpoint(epoch, validation, optimizer)
            if improved:
                torch.save(checkpoint, best_path)
            logging.info('%s epoch %d train_mae=%.8f validation=%s seconds=%.2f',
                         self.args.token_variant, epoch, total / count, validation, time.perf_counter() - start)
            if stopped:
                break
        torch.save(checkpoint, directory / 'last_checkpoint.pth')
        selected = self.load_selected(best_path)
        assert all(torch.equal(v, self.model.base.state_dict()[k].cpu()) for k, v in frozen_start.items())
        summary = dict(best_epoch=selected['epoch'], completed_epochs=epoch, baseline_validation_metrics=baseline,
                       best_validation_metrics=selected['validation_metrics'], stop_reason='patience' if stopped else 'max_epochs',
                       trainable_parameters=sum(p.numel() for p in self.model.weather_adapter.parameters()),
                       selection='minimum validation MAE including zero residual epoch 0', frozen_weights_unchanged=True)
        (directory / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return self.model
