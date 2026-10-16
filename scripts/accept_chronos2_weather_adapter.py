"""Accept A02 predictions and checkpoints against the accepted same-protocol A01."""

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def accept(report, protocol, gpu):
    import numpy as np
    import torch
    from run import configure_resources, parse_args
    from exp.exp_chronos2_weather_adapter import Exp_Chronos2_Weather_Adapter
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.metrics import metric

    record = json.loads(report.read_text())
    run = next(row for row in record['runs'] if row['protocol'] == protocol)
    out, checkpoint_dir = Path(run['output_dir']), Path(run['checkpoint_dir'])
    from utils.acceptance import accept_and_record
    accept_and_record(out)
    config = json.loads((out / 'config.json').read_text())
    assert config == json.loads((checkpoint_dir / 'config.json').read_text())
    expected = parse_args(run['argv'])
    for key in ('model', 'model_id', 'adapter_version', 'protocol', 'seed', 'seq_len', 'pred_len',
                'batch_size', 'eval_batch_size', 'precision', 'train_epochs', 'patience',
                'learning_rate', 'nwp_mode', 'modalities', 'num_workers', 'experiment_label'):
        assert config[key] == getattr(expected, key), key
    assert config['experiment_label'] == 'A02' and config['base_experiment_label'] == 'A01'
    assert (config['train_epochs'], config['patience'], config['precision']) == (100, 10, 'fp32')
    original = json.loads((ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json').read_text())
    base = next(row for row in original['acceptance']['runs']
                if row['protocol'] == protocol and row['adapter_version'] == 'b131_linear_v1')
    reference = Path(base['output_dir'])
    data_protocol = json.loads((out / 'data_protocol.json').read_text())
    training_protocol = json.loads((checkpoint_dir / 'data_protocol.json').read_text())
    for key in ('version', 'train_sites', 'validation_sites', 'test_sites', 'windows',
                'target_time_ranges', 'scaler_fit', 'data_interface', 'modalities'):
        assert data_protocol[key] == training_protocol[key], key
    assert data_protocol['sites'] == data_protocol['test_sites']
    assert training_protocol['sites'] == training_protocol['train_sites']
    assert data_protocol == json.loads((reference / 'data_protocol.json').read_text())
    assert data_protocol['windows'] == dict(train=76570, validation=25450, test=25450)
    for directory in (out, checkpoint_dir):
        with np.load(directory / 'scalers.npz') as current, np.load(reference / 'scalers.npz') as old:
            assert set(current.files) == set(old.files)
            assert all(np.array_equal(current[key], old[key]) for key in old.files)
    arrays = {name: np.load(out / f'{name}.npy') for name in
              ('pred', 'true', 'site_ids', 'forecast_timestamps')}
    prediction, truth = arrays['pred'], arrays['true']
    assert prediction.shape == truth.shape == (25450, 24, 1)
    assert np.isfinite(prediction).all() and np.isfinite(truth).all()
    for name in ('true', 'site_ids', 'forecast_timestamps'):
        assert np.array_equal(arrays[name], np.load(reference / f'{name}.npy')), name
    for site in data_protocol['test_sites']:
        values = prediction[arrays['site_ids'] == site]
        assert len(values) == 2545 and np.any(values != 0)
    saved = json.loads((out / 'metrics.json').read_text())
    assert saved['model'] == config['model'] and saved['baseline_config'] == 'weather_adapter_variant'
    for key, value in metric(prediction, truth).items():
        assert abs(saved[key] - value) < 1e-14, key
    with (ROOT / 'reports' / protocol / 'results.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['result_dir'] == str(out.resolve())]
    assert len(rows) <= 1
    if rows:
        for key in ('mae', 'rmse', 'r2'):
            assert float(rows[0][key]) == saved[key]
    with (checkpoint_dir / 'training_history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    summary = json.loads((checkpoint_dir / 'training_summary.json').read_text())
    assert [int(row['epoch']) for row in history] == list(range(len(history)))
    assert all(np.isfinite(float(row[key])) for row in history for key in ('mae', 'rmse', 'mse'))
    assert all(np.isfinite(float(row['train_mae'])) for row in history[1:])
    best = min(history, key=lambda row: float(row['mae']))
    checkpoint = torch.load(checkpoint_dir / 'checkpoint.pth', map_location='cpu', weights_only=True)
    assert checkpoint['config'] == config and checkpoint['epoch'] == int(best['epoch'])
    assert checkpoint['validation_metrics']['mae'] == float(best['mae'])
    assert all(torch.isfinite(value).all() for value in checkpoint['weather_state_dict'].values())
    assert summary['best_epoch'] == checkpoint['epoch']
    assert summary['completed_epochs'] == len(history) - 1
    assert len(history) - 1 == 100 or len(history) - 1 - checkpoint['epoch'] == 10
    assert summary['trainable_parameters'] == 29249
    lines = Path(run['log']).read_text().splitlines()
    metrics_in_log = [ast.literal_eval(line.split('Complete test metrics: ', 1)[1])
                      for line in lines if 'Complete test metrics: ' in line]
    assert metrics_in_log == [saved]
    logged_epochs = [int(re.search(r'A02 Weather Adapter epoch (\d+)', line)[1])
                     for line in lines if re.search(r'A02 Weather Adapter epoch (\d+)', line)]
    assert logged_epochs == list(range(1, len(history)))
    manifests = []
    for flag in ('train', 'val', 'test'):
        cache = ROOT / 'outputs' / protocol / 'Chronos2WeatherAdapter_features_v2_seed42_sl24_pl24_history_future' / flag
        manifest = json.loads((cache / 'complete.json').read_text())
        source = Path(manifest['source'])
        assert manifest['windows'] == data_protocol['windows'][flag.replace('val', 'validation')]
        assert manifest['source_manifest_sha256'] == hashlib.sha256((source / 'complete.json').read_bytes()).hexdigest()
        weather = np.load(cache / 'future_nwp.npy', mmap_mode='r')
        assert list(weather.shape) == manifest['future_nwp_shape'] and np.isfinite(weather).all()
        manifests.append(manifest)
    args = parse_args(run['argv'] + ['--gpu', str(gpu)])
    configure_resources(args)
    experiment = Exp_Chronos2_Weather_Adapter(args)
    model = experiment.model.eval()
    assert checkpoint['base_provenance'] == model.base_provenance
    model.weather_adapter.load_state_dict(checkpoint['weather_state_dict'], strict=True)
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    cache = ROOT / 'outputs' / protocol / 'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future/test'
    indices = [0, 1, 2545, 25449]
    features = [torch.from_numpy(np.load(cache / f'{name}.npy', mmap_mode='r')[indices].copy()).to(f'cuda:{gpu}')
                for name in ('embeds', 'loc', 'scale')]
    weather_cache = ROOT / 'outputs' / protocol / 'Chronos2WeatherAdapter_features_v2_seed42_sl24_pl24_history_future/test/future_nwp.npy'
    weather = torch.from_numpy(np.load(weather_cache, mmap_mode='r')[indices].copy()).to(f'cuda:{gpu}')
    with torch.no_grad():
        replay = model.forward_features(*features, weather).cpu().numpy()
    difference = float(np.max(np.abs(replay - prediction[indices])))
    assert difference <= 2e-6, difference
    # The model saves artifacts first; only a fully checked run enters the formal index.
    Exp_Long_Term_Forecast._record_result(experiment, out, saved, specific_checks={'source': __file__})
    with (ROOT / 'reports' / protocol / 'results.csv').open() as handle:
        registered = [row for row in csv.DictReader(handle) if row['result_dir'] == str(out.resolve())]
    assert len(registered) == 1
    assert all(float(registered[0][key]) == saved[key] for key in ('mae', 'rmse', 'r2'))
    base_metrics = json.loads((reference / 'metrics.json').read_text())
    item = dict(protocol=protocol, experiment_label='A02', base_experiment_label='A01',
                status='passed_full_training_test_artifact_acceptance', metrics=saved,
                best_epoch=checkpoint['epoch'], completed_epochs=summary['completed_epochs'],
                base_provenance=model.base_provenance, trainable_parameters=29249,
                selected_checkpoint_replay_max_abs_difference=difference,
                weather_cache_manifests=manifests,
                comparison_to_A01={key: dict(absolute_change=saved[key] - base_metrics[key],
                    relative_change_percent=(saved[key] / base_metrics[key] - 1) * 100)
                    for key in ('mae', 'rmse', 'r2')},
                checks=['same-protocol A01 checkpoint identity; base fully frozen',
                        'full labels/sites/timestamps match accepted current-protocol A01',
                        'train-only scaler and native input protocol unchanged',
                        'finite full predictions; each site nonzero',
                        'minimum validation MAE including epoch0; max100/patience10',
                        'metrics recomputed; log and unique protocol index agree',
                        'selected weather checkpoint small-batch GPU replay'],
                comparison_limit='seed42 only; joint residual does not isolate weather/representation contribution',
                output_dir=str(out), checkpoint_path=str(checkpoint_dir / 'checkpoint.pth'),
                accepted_at=datetime.now().astimezone().isoformat())
    (report.parent / f'acceptance_{protocol}.json').write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
    run.update(state='completed_artifact_acceptance_passed', metrics=saved, acceptance=item)
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(protocol=protocol, status=item['status'], metrics=saved), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--protocol', choices=['in_domain', 'zeroshot_v1'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    options = parser.parse_args()
    accept(options.report.resolve(), options.protocol, options.gpu)
