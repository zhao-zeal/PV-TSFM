"""Common saved-artifact checks and the sole writer of the formal result index."""

import ast
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from utils.metrics import metric


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['data_protocol', 'model', 'baseline_config', 'llm_model', 'seed', 'input_len',
          'output_len', 'train_sites', 'validation_sites', 'test_sites', 'split_version',
          'scaler_version', 'test_windows', 'mae', 'rmse', 'r2', 'result_dir']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def needs_specific_checks(config):
    return (config['model'] in ('TimeVLM', 'ChronosX', 'Chronos2WeatherAdapter',
                                'Chronos2HistoryWeatherAdapter', 'Chronos2WeatherReadout', 'Chronos2TokenAdapter',
                                'Chronos2MCDecoder')
            or (config['model'] == 'FusionSF' and config['baseline_config'] == 'fusion_input_variant'))


def append_validation(directory, epoch, validation, train_loss=None):
    """Record the actual selection candidates, including a resumed best epoch."""
    values = dict(epoch=epoch, train_loss='' if train_loss is None else train_loss, **validation)
    require(all(np.isfinite(value) for value in values.values() if value != ''),
            'Nonfinite training/validation history')
    path = Path(directory) / 'training_history.csv'
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        if not exists:
            writer.writeheader()
        writer.writerow(values)


def _check_samples(config, protocol, pred, truth, sites, times):
    import pandas as pd

    name = config['protocol']
    require(name in ('site0_v1', 'site1_v1', 'in_domain', 'zeroshot_v1'), 'Unsupported formal protocol')
    test_sites = [0] if name == 'site0_v1' else ([1] if name == 'site1_v1' else list(range(10)))
    train_sites = list(range(10, 20)) if name == 'zeroshot_v1' else test_sites
    expected = dict(version=name, train_sites=train_sites, validation_sites=train_sites,
                    test_sites=test_sites, sites=test_sites, input_len=config['seq_len'],
                    output_len=config['pred_len'], label_len=config['label_len'],
                    modalities=config['modalities'], split_version='chronological_target_split_v1',
                    scaler_version='train_sites_and_time_only_fit_v1')
    for key, value in expected.items():
        require(protocol.get(key) == value, f'Protocol mismatch: {key}')
    shape = (protocol['windows']['test'], config['pred_len'], 1)
    require(pred.shape == truth.shape == shape, f'Expected prediction/target shape {shape}')
    require(sites.shape == (shape[0],) and times.shape == shape[:2], 'Invalid site/time shapes')
    require(sites.dtype.kind in 'iu' and times.dtype.kind in 'iu', 'Site/time indices must be integers')
    raw = pd.read_csv(Path(config['root_path']) / 'solar_power/solar_power.csv',
                      usecols=['datetime', 'site', 'power'], parse_dates=['datetime'])
    timeline = None
    cursor = 0
    for site in sorted(set(train_sites + test_sites)):
        group = raw.loc[raw.site == site].sort_values('datetime')
        ts = group.datetime.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        require(len(ts) >= 3 and np.all(np.diff(ts) > 0), f'Invalid raw timeline at site {site}')
        if timeline is None:
            timeline = ts
        require(np.array_equal(ts, timeline), 'Sites must share the protocol timeline')
        n = len(ts)
        train_end, val_end = int(n * .6), int(n * .8)
        seq, horizon = config['seq_len'], config['pred_len']
        if site not in test_sites:
            continue
        starts = np.arange(max(seq, val_end), n - horizon + 1)
        indices = starts[:, None] + np.arange(horizon)
        stop = cursor + len(starts)
        require(np.all(sites[cursor:stop] == site) and stop <= len(sites), 'Site coverage/order differs')
        require(np.array_equal(times[cursor:stop], ts[indices]), 'Missing, duplicate or incorrect forecast windows')
        require(np.array_equal(truth[cursor:stop, :, 0], group.power.to_numpy(dtype=np.float32)[indices]),
                'Labels differ from raw solar power CSV')
        require(np.ptp(pred[cursor:stop]) > 0, f'Constant/zero predictions at site {site}')
        cursor = stop
    require(cursor == len(sites), 'Unexpected extra test samples')
    counts = dict(train=max(0, train_end - horizon - seq + 1) * len(train_sites),
                  validation=max(0, val_end - horizon - max(seq, train_end) + 1) * len(train_sites),
                  test=max(0, n - horizon - max(seq, val_end) + 1) * len(test_sites))
    require(protocol['windows'] == counts, 'Split window counts differ from raw timeline')
    boundaries = {'train': [timeline[0], timeline[train_end]],
                  'validation': [timeline[train_end], timeline[val_end]],
                  'test': [timeline[val_end], timeline[-1] + timeline[-1] - timeline[-2]]}
    for split, bounds in boundaries.items():
        actual = pd.DatetimeIndex(protocol['target_time_ranges'][split]).asi8
        require(np.array_equal(actual, bounds), f'Incorrect {split} target boundaries')


def _check_scalers(directory, protocol):
    expected = {f'{name}_{key}' for name in protocol['scaler_fit'] for key in ('mean', 'scale')}
    if not expected:
        require(not (directory / 'scalers.npz').exists(), 'Unexpected scaler for these inputs')
        return {}
    with np.load(directory / 'scalers.npz', allow_pickle=False) as saved:
        require(set(saved.files) == expected, 'Scaler keys differ from protocol')
        values = {key: saved[key] for key in saved.files}
    for key, value in values.items():
        require(np.isfinite(value).all(), f'Nonfinite scaler: {key}')
        if key.endswith('_scale'):
            require(np.all(value > 0), f'Nonpositive scaler: {key}')
    return values


def _check_checkpoint(directory, config, protocol, scalers):
    if config['model'] in ('Chronos2', 'TimesFM3'):
        return dict(selection='frozen_pretrained', model_path=config['model_path'])
    import torch

    path = (Path(config['checkpoint']) if not config['is_training'] else
            Path(config['checkpoints']) / config['protocol'] / config['model_id'] / 'checkpoint.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    source = checkpoint['config']
    require(read_json(path.parent / 'config.json') == source, 'Checkpoint and training configuration differ')
    ignored = {'model_id', 'is_training', 'checkpoint', 'resume', 'gpu', 'use_gpu',
               'num_workers', 'output_dir', 'checkpoints', 'print_config', 'root_path'}
    if config['is_training']:
        require(source == config, 'Checkpoint and test configuration differ')
    else:
        for key in source.keys() | config.keys():
            if key not in ignored:
                require(source.get(key) == config.get(key), f'Evaluation configuration differs: {key}')
    training = read_json(path.parent / 'data_protocol.json')
    for key in protocol.keys() | training.keys():
        if key != 'sites':  # Training and test stations differ in zero-shot evaluation.
            require(protocol.get(key) == training.get(key), f'Training/test protocol differs: {key}')
    training_scalers = _check_scalers(path.parent, training)
    require(scalers.keys() == training_scalers.keys(), 'Training/test scaler keys differ')
    for key in scalers:
        require(np.array_equal(scalers[key], training_scalers[key]), f'Training/test scaler differs: {key}')
    states = [value for key, value in checkpoint.items()
              if key.endswith('state_dict') and key not in ('optim_state_dict', 'optimizer_state_dict',
                                                            'scheduler_state_dict', 'scaler_state_dict')]
    require(bool(states), 'Checkpoint has no model weights')
    for state in states:
        require(all(torch.isfinite(value).all().item() for value in state.values()), 'Nonfinite checkpoint weights')
    history = path.parent / 'training_history.csv'
    if history.exists():
        with history.open() as handle:
            rows = list(csv.DictReader(handle))
    else:
        # Existing runs predate the shared structured history; do not rewrite them.
        log = Path(source['output_dir']) / source['protocol'] / source['model_id'] / 'run.log'
        rows = [dict(epoch=int(epoch), train_loss=float(loss), **ast.literal_eval(values))
                for epoch, loss, values in re.findall(r'Epoch (\d+) train_loss=(\S+) validation=(\{.*?\}) lr=', log.read_text())]
        if source.get('resume'):
            resumed = torch.load(source['resume'], map_location='cpu', weights_only=True)
            rows.insert(0, dict(epoch=resumed['epoch'], **resumed['validation_metrics']))
    monitor = source['monitor']
    require(bool(rows), 'Missing validation selection history')
    epochs = [int(row['epoch']) for row in rows]
    require(epochs == sorted(set(epochs)), 'Duplicate or unordered validation epochs')
    for row in rows:
        for key in (monitor, 'train_loss', 'train_mae'):
            if key in row and row[key] != '':
                require(np.isfinite(float(row[key])), f'Nonfinite history: {key}')
    best = min(rows, key=lambda row: float(row[monitor]))
    require(checkpoint['epoch'] == int(best['epoch']), 'Checkpoint is not the validation-best epoch')
    require(np.isclose(checkpoint['validation_metrics'][monitor], float(best[monitor]), rtol=1e-12, atol=1e-14),
            'Checkpoint validation score differs from history')
    return dict(checkpoint=str(path.resolve()), best_epoch=checkpoint['epoch'], monitor=monitor,
                best_validation=float(best[monitor]), selection='minimum_validation')


def validate_artifacts(directory):
    """Read-only validation; always re-read evidence rather than trusting a prior pass."""
    directory = Path(directory)
    config = read_json(directory / 'config.json')
    require(directory.name == config['model_id'] and directory.parent.name == config['protocol'],
            'Output directory does not match the run identity')
    protocol = read_json(directory / 'data_protocol.json')
    arrays = [np.load(directory / f'{name}.npy', allow_pickle=False)
              for name in ('pred', 'true', 'site_ids', 'forecast_timestamps')]
    computed = metric(arrays[0], arrays[1])
    _check_samples(config, protocol, *arrays)
    saved = read_json(directory / 'metrics.json')
    for key, value in computed.items():
        require(saved.get(key) == value if value is None else
                isinstance(saved.get(key), (int, float)) and np.isclose(saved[key], value, rtol=1e-12, atol=1e-14),
                f'Recomputed metric differs: {key}')
    expected = dict(model=config['model'], seed=config['seed'], data_protocol=config['protocol'],
                    baseline_config=config['baseline_config'], input_len=config['seq_len'], output_len=config['pred_len'])
    expected.update({key: protocol[key] for key in ('train_sites', 'validation_sites', 'test_sites',
                                                   'split_version', 'scaler_version')})
    for key, value in expected.items():
        require(saved.get(key) == value, f'Metric metadata differs: {key}')
    selection = _check_checkpoint(directory, config, protocol, _check_scalers(directory, protocol))
    return dict(status='passed_common_checks', checked_at=datetime.now(timezone.utc).isoformat(),
                model_id=config['model_id'], metrics=saved, selection=selection,
                specific_checks_required=needs_specific_checks(config),
                checks=['full_test_coverage', 'raw_labels_and_timestamps', 'finite_nondegenerate_predictions',
                        'metrics_recomputed_float64', 'config_protocol_scalers', 'validation_best_checkpoint'],
                limitation='Common checks do not replay predictions or prove NWP issue-time availability')


def _update_index(directory, metrics, reports_root):
    table = Path(reports_root) / directory.parent.name / 'results.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if table.exists():
            with table.open() as handle:
                rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row['result_dir'] != str(directory.resolve())]
        if metrics is not None:
            rows.append({**metrics, 'result_dir': str(directory.resolve())})
        with tempfile.NamedTemporaryFile(mode='w', dir=table.parent, newline='', delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(table)


def accept_and_record(directory, *, specific_checks=None, reports_root=ROOT / 'reports'):
    """Register only a freshly validated run, after any required model-specific checks."""
    directory = Path(directory)
    try:
        receipt = validate_artifacts(directory)
    except Exception as error:
        receipt = dict(status='failed', checked_at=datetime.now(timezone.utc).isoformat(), error=str(error))
        (directory / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
        _update_index(directory, None, reports_root)
        raise
    passed = not receipt['specific_checks_required'] or specific_checks is not None
    receipt.update(status='passed' if passed else 'pending_specific_checks', specific_checks=specific_checks)
    (directory / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    _update_index(directory, receipt['metrics'] if passed else None, reports_root)
    return receipt
