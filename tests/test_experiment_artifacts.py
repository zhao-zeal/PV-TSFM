"""Integrity checks use tiny synthetic artifacts, never MMSP training or GPUs."""

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from utils.acceptance import accept_and_record, validate_artifacts
from utils.metrics import metric
from utils.run_artifacts import reserve_run


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(model='CrossUnet', model_id='trial', protocol='site1_v1',
                           root_path=str(self.root / 'data'), seq_len=4, label_len=2, pred_len=3,
                           seed=42, modalities='power', baseline_config='official', monitor='mse',
                           is_training=1, checkpoint=None, output_dir=str(self.root / 'outputs'),
                           checkpoints=str(self.root / 'checkpoints'))
        self.out = reserve_run(SimpleNamespace(**self.config), 'trial')
        self.ck = self.root / 'checkpoints/site1_v1/trial'
        self.reports = self.root / 'reports'
        times = pd.date_range('2024-01-01', periods=60, freq='h')
        values = (np.arange(60) % 11 / 10).astype(np.float32)
        raw = pd.DataFrame(dict(datetime=times, site=1, power=values))
        data = self.root / 'data/solar_power'
        data.mkdir(parents=True)
        raw.to_csv(data / 'solar_power.csv', index=False)
        self.protocol = dict(version='site1_v1', train_sites=[1], validation_sites=[1],
                             test_sites=[1], sites=[1], input_len=4, label_len=2, output_len=3,
                             modalities='power', split_version='chronological_target_split_v1',
                             scaler_version='train_sites_and_time_only_fit_v1', scaler_fit={},
                             windows=dict(train=30, validation=10, test=10), target_time_ranges={
                                 'train': [str(times[0]), str(times[36])],
                                 'validation': [str(times[36]), str(times[48])],
                                 'test': [str(times[48]), str(times[-1] + pd.Timedelta(hours=1))]})
        indices = np.arange(48, 58)[:, None] + np.arange(3)
        self.truth = values[indices, None]
        self.pred = self.truth + np.float32(.1)
        for key, value in dict(pred=self.pred, true=self.truth, site_ids=np.ones(10, dtype=np.int64),
                               forecast_timestamps=times.asi8[indices]).items():
            np.save(self.out / f'{key}.npy', value)
        self.checkpoint = dict(config=self.config.copy(), epoch=2, validation_metrics=dict(mse=.25),
                               model_state_dict={'weight': torch.tensor([1.])})
        self.save_metadata()
        (self.ck / 'training_history.csv').write_text('epoch,train_loss,mse\n1,.8,.8\n2,.3,.25\n3,.2,.3\n')

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def save_metadata(self):
        for directory in (self.out, self.ck):
            self.write(directory / 'config.json', self.config)
            self.write(directory / 'data_protocol.json', self.protocol)
        self.checkpoint['config'] = self.config.copy()
        torch.save(self.checkpoint, self.ck / 'checkpoint.pth')
        metrics = dict(metric(self.pred, self.truth), model=self.config['model'], seed=42,
                       data_protocol='site1_v1', baseline_config='official', input_len=4, output_len=3, llm_model='')
        metrics.update({key: self.protocol[key] for key in ('train_sites', 'validation_sites',
                       'test_sites', 'split_version', 'scaler_version')})
        self.write(self.out / 'metrics.json', metrics)

    def rows(self):
        with (self.reports / 'site1_v1/results.csv').open() as handle:
            return list(csv.DictReader(handle))

    def accept(self, **kwargs):
        return accept_and_record(self.out, reports_root=self.reports, **kwargs)

    def test_valid_and_repeat_acceptance_has_one_row(self):
        self.assertEqual(self.accept()['status'], 'passed')
        self.assertEqual(self.accept()['selection']['best_epoch'], 2)
        self.assertEqual(len(self.rows()), 1)

    def test_metrics_reject_broadcast_empty_nan(self):
        for pred, truth in ((np.ones((2, 1, 1)), np.ones((2, 3, 1))),
                            (np.empty((0, 3, 1)), np.empty((0, 3, 1))),
                            (np.array([np.nan]), np.array([0.]))):
            with self.subTest(shape=pred.shape), self.assertRaises(ValueError):
                metric(pred, truth)

    def test_bad_arrays_cannot_register(self):
        for filename, replacement in (
                ('pred', np.zeros_like(self.pred)), ('pred', np.full_like(self.pred, np.nan)),
                ('pred', self.pred[:, :1]), ('true', self.truth + 1),
                ('site_ids', np.zeros(10, dtype=np.int64)),
                ('forecast_timestamps', np.zeros((10, 3), dtype=np.int64))):
            path = self.out / f'{filename}.npy'
            original = np.load(path)
            with self.subTest(filename=filename):
                np.save(path, replacement)
                with self.assertRaises(ValueError):
                    self.accept()
                self.assertEqual(self.rows(), [])
                self.assertEqual(json.loads((self.out / 'acceptance.json').read_text())['status'], 'failed')
            np.save(path, original)

    def test_duplicate_window_with_correct_shape_rejected(self):
        for name in ('pred', 'true', 'forecast_timestamps'):
            path = self.out / f'{name}.npy'
            array = np.load(path)
            array[-1] = array[-2]
            np.save(path, array)
        with self.assertRaisesRegex(ValueError, 'forecast windows'):
            self.accept()

    def test_changed_metric_revokes_previous_registration(self):
        self.accept()
        saved = json.loads((self.out / 'metrics.json').read_text())
        saved['mae'] = .001
        self.write(self.out / 'metrics.json', saved)
        with self.assertRaisesRegex(ValueError, 'Recomputed metric'):
            self.accept()
        self.assertEqual(self.rows(), [])

    def test_wrong_checkpoint_selection_and_weights(self):
        for update in (dict(epoch=3), dict(validation_metrics=dict(mse=.2)),
                       dict(model_state_dict={'weight': torch.tensor([float('nan')])})):
            with self.subTest(update=list(update)):
                torch.save(dict(self.checkpoint, **update), self.ck / 'checkpoint.pth')
                with self.assertRaises(ValueError):
                    self.accept()

    def test_configuration_and_scaler_mismatch(self):
        config = dict(self.config, seed=7)
        self.write(self.ck / 'config.json', config)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.accept()
        self.save_metadata()
        self.protocol['scaler_fit'] = {'nwp': {}}
        self.save_metadata()
        np.savez(self.ck / 'scalers.npz', nwp_mean=[0.], nwp_scale=[1.])
        np.savez(self.out / 'scalers.npz', nwp_mean=[2.], nwp_scale=[1.])
        with self.assertRaisesRegex(ValueError, 'scaler differs'):
            self.accept()

    def test_specific_check_gate(self):
        self.config['model'] = 'Chronos2WeatherReadout'
        self.save_metadata()
        self.assertEqual(self.accept()['status'], 'pending_specific_checks')
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.accept(specific_checks={'source': 'unit_test'})['status'], 'passed')
        self.assertEqual(len(self.rows()), 1)
        # Starting a new acceptance does not inherit an earlier specific pass.
        self.assertEqual(self.accept()['status'], 'pending_specific_checks')
        self.assertEqual(self.rows(), [])

    def test_separate_evaluation_uses_source_checkpoint(self):
        self.config.update(model_id='evaluation', is_training=0, checkpoint=str(self.ck / 'checkpoint.pth'))
        evaluation = reserve_run(SimpleNamespace(**self.config), 'evaluation')
        for path in self.out.iterdir():
            if path.is_file():
                (evaluation / path.name).write_bytes(path.read_bytes())
        self.write(evaluation / 'config.json', self.config)
        self.assertEqual(validate_artifacts(evaluation)['selection']['best_epoch'], 2)

    def test_frozen_model_has_no_training_checkpoint_requirement(self):
        self.config.update(model='Chronos2', is_training=0, model_path='local_pretrained')
        self.save_metadata()
        (self.ck / 'checkpoint.pth').unlink()
        self.assertEqual(self.accept()['selection']['selection'], 'frozen_pretrained')

    def test_existing_runs_never_overwrite(self):
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from exp.exp_chronos2_adapter import Exp_Chronos2_Adapter
        from exp.exp_chronos2_weather_adapter import Exp_Chronos2_Weather_Adapter
        from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter
        from exp.exp_chronos2_weather_readout import Exp_Chronos2_Weather_Readout
        before = (self.out / 'config.json').read_bytes()
        for cls in (Exp_Long_Term_Forecast, Exp_Chronos2_Adapter, Exp_Chronos2_Weather_Adapter,
                    Exp_Chronos2_History_Weather_Adapter, Exp_Chronos2_Weather_Readout):
            experiment = cls.__new__(cls)
            experiment.args = SimpleNamespace(**self.config)
            with self.subTest(model=cls.__name__), self.assertRaises(FileExistsError):
                experiment.train('trial')
            with self.subTest(model=cls.__name__), self.assertRaises(FileExistsError):
                experiment.test('trial')
        self.assertEqual((self.out / 'config.json').read_bytes(), before)
        with self.assertRaises(ValueError):
            reserve_run(experiment.args, '..')

    def test_new_run_and_resume_destination_are_independent(self):
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        args = SimpleNamespace(**dict(self.config, model_id='continued', resume=str(self.ck / 'checkpoint.pth')))
        experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        experiment.args = args
        experiment._prepare_run('continued', 'train')
        experiment._prepare_run('continued', 'test')
        with self.assertRaises(FileExistsError):
            experiment._prepare_run('continued', 'test')
        for handler in list(logging.getLogger().handlers):
            if getattr(handler, '_pvtsfm_run_log', False):
                logging.getLogger().removeHandler(handler)
                handler.close()

    def test_default_ids_differ_and_inspection_creates_no_run(self):
        from run import parse_args
        argv = ['--model', 'CrossUnet', '--protocol', 'in_domain', '--print_config',
                '--output_dir', str(self.root / 'inspect'), '--checkpoints', str(self.root / 'inspect_ck')]
        self.assertNotEqual(parse_args(argv).model_id, parse_args(argv).model_id)
        self.assertFalse((self.root / 'inspect').exists())

    def test_concurrent_registrations_keep_both_results(self):
        outputs = []
        for name in ('evaluation_a', 'evaluation_b'):
            config = dict(self.config, model_id=name, is_training=0, checkpoint=str(self.ck / 'checkpoint.pth'))
            out = reserve_run(SimpleNamespace(**config), name)
            for path in self.out.iterdir():
                (out / path.name).write_bytes(path.read_bytes())
            self.write(out / 'config.json', config)
            outputs.append(out)
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda path: accept_and_record(path, reports_root=self.reports), outputs))
        self.assertTrue(all(item['status'] == 'passed' for item in receipts))
        self.assertEqual({row['result_dir'] for row in self.rows()}, {str(path) for path in outputs})

    def test_adapter_replay_allows_device_change_only(self):
        from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter
        from exp.exp_chronos2_weather_readout import Exp_Chronos2_Weather_Readout
        for cls in (Exp_Chronos2_History_Weather_Adapter, Exp_Chronos2_Weather_Readout):
            with self.subTest(model=cls.__name__):
                experiment = cls.__new__(cls)
                experiment.args = SimpleNamespace(**dict(self.config, gpu=7))
                experiment.device = torch.device('cpu')
                experiment.model = SimpleNamespace(weather_adapter=torch.nn.Linear(1, 1),
                    history_relation=torch.nn.Linear(1, 1), base_provenance={'base': 'fixture'})
                checkpoint = dict(config=dict(self.config, gpu=0), base_provenance={'base': 'fixture'},
                    weather_state_dict=experiment.model.weather_adapter.state_dict(),
                    relation_state_dict=experiment.model.history_relation.state_dict())
                path = self.root / 'replay.pth'
                torch.save(checkpoint, path)
                experiment.load_selected(path)
                experiment.args.seed = 7
                with self.assertRaises(ValueError):
                    experiment.load_selected(path)


class PredictionSelectionTests(unittest.TestCase):
    def test_four_baselines_use_100_10_and_prediction_rmse(self):
        from run import parse_args
        for model in ('CrossUnet', 'TimeVLM', 'ChronosX', 'TimeXer'):
            args = parse_args(['--model', model, '--protocol', 'in_domain', '--print_config'])
            self.assertEqual((args.train_epochs, args.patience, args.monitor), (100, 10, 'rmse'))

    def test_chronosx_forecast_validation_repeats_without_consuming_rng(self):
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        class Forecast(torch.nn.Module):
            def token_loss(self, x, target, cov):
                return torch.tensor(9.)

            def forward(self, x, *args, **kwargs):
                return torch.rand_like(x)

        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(model='ChronosX', monitor='rmse', seed=42, effective_precision='fp32')
        exp.device = torch.device('cpu')
        exp.model = Forecast()
        exp._prepare_batch = lambda x: (x, None, None, None, {}, torch.zeros_like(x))
        batches = [torch.zeros(3, 2, 1), torch.zeros(1, 2, 1)]
        before = torch.get_rng_state()
        first = exp.vali(batches)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(7)
        second = exp.vali(batches)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['rmse'] ** 2, first['mse'])
        self.assertEqual(first['token_ce'], 9.)
        self.assertLess(first['rmse'], 1.)

    def test_chronosx_selects_forecast_rmse_when_ce_disagrees(self):
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        class Trainable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def token_loss(self, *args, **kwargs):
                return self.weight.square()

        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(model='ChronosX', monitor='rmse', effective_precision='fp32',
            train_epochs=4, patience=1, gradient_accumulation_steps=1, max_grad_norm=1.)
        exp.device = torch.device('cpu')
        exp.model = Trainable()
        exp._prepare_batch = lambda x: (x, None, None, None, {}, x)
        values = iter([dict(rmse=.3, token_ce=1.), dict(rmse=.2, token_ce=2.),
                       dict(rmse=.4, token_ce=.5)])
        exp.vali = lambda loader: next(values)
        optimizer = torch.optim.Adam(exp.model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pth'
            exp._train_chronosx([torch.zeros(2, 1, 1)], [], optimizer, scheduler, checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['validation_metrics']['rmse'], .2)


class Site0CovariateAblationTests(unittest.TestCase):
    def test_site0_protocol_selects_only_site_zero(self):
        from data_provider.protocols import canonical_protocol
        self.assertEqual(canonical_protocol('site0_v1'), 'site0_v1')
        args = __import__('run').parse_args([
            '--model', 'Chronos2Adapter', '--protocol', 'site0_v1', '--print_config'])
        self.assertEqual(args.protocol, 'site0_v1')

    def test_adapter_slices_history_and_future_identically(self):
        from models.Chronos2Adapter import Model
        model = object.__new__(Model)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(nwp_indices=[0, 2], pred_len=3)
        captured = {}
        class Backbone:
            def encode(self, **kwargs):
                captured.update(kwargs)
                batch = kwargs['group_ids'].max().item() + 1
                groups = len(kwargs['group_ids'])
                return ([torch.zeros(groups, 2, 4)],
                        (torch.zeros(groups), torch.ones(groups)), None, None)
        model.backbone = Backbone()
        power = torch.zeros(2, 4)
        history = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
        future = torch.arange(2 * 3 * 3).reshape(2, 3, 3).float()
        embeds, loc, scale = model.encode_inputs(power, history, future)
        self.assertEqual(captured['context'].shape, (6, 4))
        self.assertEqual(captured['future_covariates'].shape, (6, 3))
        self.assertEqual(embeds.shape, (2, 2, 4))
        self.assertEqual(loc.shape, scale.shape)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
