"""One train/validation/test path for all migrated MMSP forecasting models."""

import csv
from contextlib import nullcontext
import json
import logging
from pathlib import Path

import numpy as np
import torch

from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from utils.crossunet_scheduler import CrossUnetType3LR
from utils.fusionsf_scheduler import CosineWarmupScheduler
from utils.losses import forecasting_loss
from utils.metrics import metric
from utils.tools import EarlyStopping, load_model_checkpoint


class Exp_Long_Term_Forecast(Exp_Basic):
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _autocast(self):
        precision = self.args.effective_precision
        if precision == 'fp32':
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, covariates = batch
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        covariates = {name: value.to(self.device) for name, value in covariates.items()}
        decoder = torch.cat([
            batch_y[:, :self.args.label_len],
            torch.zeros_like(batch_y[:, -self.args.pred_len:]),
        ], dim=1)
        return batch_x, batch_x_mark.to(self.device), decoder, batch_y_mark.to(self.device), covariates, batch_y[:, -self.args.pred_len:]

    def _forward_batch(self, batch):
        batch_x, x_mark, decoder, y_mark, covariates, targets = self._prepare_batch(batch)
        with self._autocast():
            result = self.model(batch_x, x_mark, decoder, y_mark, covariates=covariates)
        return result, targets, covariates

    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer == 'adam':
            # Author OneCycleLR constructors also set beta1 to max_momentum=0.95.
            beta1 = 0.95 if self.args.lradj in ('type3', 'type1_onecycle_init') else 0.9
            return torch.optim.Adam(params, lr=self.args.learning_rate, betas=(beta1, 0.999))
        if self.args.optimizer == 'adamw_chronosx':
            return torch.optim.AdamW(params, lr=self.args.learning_rate, weight_decay=0.,
                                     fused=self.device.type == 'cuda')
        return torch.optim.AdamW(params, lr=self.args.learning_rate,
                                 weight_decay=0.05, betas=(0.9, 0.95))

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'fusion_cosine':
            return CosineWarmupScheduler(optimizer, warmup=5, max_iters=self.args.train_epochs)
        if self.args.lradj == 'type3':
            return CrossUnetType3LR(optimizer)
        if self.args.lradj in ('type1', 'type1_onecycle_init'):
            initial = 1 / 25 if self.args.lradj == 'type1_onecycle_init' else 1.
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: initial if epoch == 0 else 0.5 ** (epoch - 1))
        if self.args.lradj == 'linear':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: max(0., 1. - step / self.args.max_steps))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)

    def _save_config(self, directory, dataset):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(vars(self.args), indent=2) + '\n')
        return dataset.save_protocol(directory)

    @torch.no_grad()
    def vali(self, loader):
        self.model.eval()
        total_abs = total_square = count = 0
        if self.args.model == 'ChronosX':
            total_loss = samples = 0
            for batch in loader:
                batch_x, _, _, _, covariates, targets = self._prepare_batch(batch)
                loss = self.model.token_loss(batch_x, targets, covariates)
                total_loss += loss.item() * len(batch_x)
                samples += len(batch_x)
            return {'token_ce': total_loss / samples}
        for batch in loader:
            result, targets, _ = self._forward_batch(batch)
            prediction = result['prediction'] if isinstance(result, dict) else result
            error = (prediction - targets).double()
            total_abs += error.abs().sum().item()
            total_square += error.square().sum().item()
            count += targets.numel()
        return {'mae': total_abs / count, 'rmse': (total_square / count) ** 0.5,
                'mse': total_square / count}

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        directory = Path(self.args.checkpoints) / self.args.protocol / setting
        self._save_config(directory, train_data)
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        stopping = EarlyStopping(self.args.patience)
        best_path = directory / 'checkpoint.pth'
        if self.args.model == 'ChronosX':
            return self._train_chronosx(train_loader, val_loader, optimizer, scheduler, best_path)
        scaler = torch.amp.GradScaler('cuda', enabled=self.args.effective_precision == 'fp16')
        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                result, targets, _ = self._forward_batch(batch)
                with self._autocast():
                    loss = forecasting_loss(result, targets, self.args.model)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            validation = self.vali(val_loader)
            score = validation[self.args.monitor]
            improved, stopped = stopping.update(score)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation['rmse'])
            else:
                scheduler.step()
            logging.info('Epoch %d validation=%s lr=%s', epoch, validation, optimizer.param_groups[0]['lr'])
            if improved:
                torch.save({
                    'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'validation_metrics': validation, 'config': vars(self.args),
                }, best_path)
            if stopped:
                break
        load_model_checkpoint(self.model, best_path, self.device)
        return self.model

    def _train_chronosx(self, train_loader, val_loader, optimizer, scheduler, best_path):
        """Official step budget, accumulation, linear LR and best token-CE checkpoint."""
        iterator = iter(train_loader)
        best = float('inf')
        for step in range(1, self.args.max_steps + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(self.args.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch_x, _, _, _, covariates, targets = self._prepare_batch(batch)
                loss = self.model.token_loss(batch_x, targets, covariates, augment=True)
                (loss / self.args.gradient_accumulation_steps).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if step % self.args.eval_steps == 0 or step == self.args.max_steps:
                validation = self.vali(val_loader)
                logging.info('Step %d validation=%s lr=%s', step, validation, optimizer.param_groups[0]['lr'])
                if validation['token_ce'] < best:
                    best = validation['token_ce']
                    torch.save({'step': step, 'model_state_dict': self.model.state_dict(),
                                'optim_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'validation_metrics': validation, 'config': vars(self.args)}, best_path)
        load_model_checkpoint(self.model, best_path, self.device)
        return self.model

    @torch.no_grad()
    def test(self, setting, test=0):
        if test and self.args.checkpoint:
            load_model_checkpoint(self.model, self.args.checkpoint, self.device)
        data, loader = self._get_data('test')
        directory = Path(self.args.output_dir) / self.args.protocol / setting
        protocol = self._save_config(directory, data)
        self.model.eval()
        predictions, targets, sites, times = [], [], [], []
        for batch in loader:
            result, truth, covariates = self._forward_batch(batch)
            prediction = result['prediction'] if isinstance(result, dict) else result
            predictions.append(prediction.float().cpu().numpy())
            targets.append(truth.cpu().numpy())
            sites.append(covariates['site_id'].cpu().numpy())
            times.append(covariates['forecast_timestamps'].cpu().numpy())
        prediction, truth = np.concatenate(predictions), np.concatenate(targets)
        for name, value in [('pred', prediction), ('true', truth),
                            ('site_ids', np.concatenate(sites)), ('forecast_timestamps', np.concatenate(times))]:
            np.save(directory / f'{name}.npy', value)
        metrics = metric(prediction, truth)
        metrics.update(model=self.args.model, seed=self.args.seed, data_protocol=self.args.protocol,
                       baseline_config=self.args.baseline_config,
                       llm_model=self.args.llm_model if self.args.model == 'TimeLLM' else '',
                       input_len=self.args.seq_len, output_len=self.args.pred_len,
                       **{key: protocol[key] for key in ('train_sites', 'validation_sites', 'test_sites',
                                                       'split_version', 'scaler_version')})
        (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        self._record_result(directory, metrics)
        logging.info('Complete test metrics: %s', metrics)
        return metrics

    def _record_result(self, directory, metrics):
        table = Path(__file__).resolve().parents[1] / 'reports' / self.args.protocol / 'results.csv'
        table.parent.mkdir(parents=True, exist_ok=True)
        fields = ['data_protocol', 'model', 'baseline_config', 'llm_model', 'seed', 'input_len', 'output_len', 'train_sites',
                  'validation_sites', 'test_sites', 'split_version', 'scaler_version',
                  'test_windows', 'mae', 'rmse', 'r2', 'result_dir']
        rows = list(csv.DictReader(table.open())) if table.exists() else []
        rows = [row for row in rows if row['result_dir'] != str(directory.resolve())]
        rows.append({**metrics, 'result_dir': str(directory.resolve())})
        with table.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
