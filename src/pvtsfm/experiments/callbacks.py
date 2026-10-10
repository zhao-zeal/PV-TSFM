"""Save MMSP protocol metadata and compute R² over the complete evaluation set."""

import csv
import fcntl
import json
from pathlib import Path

import numpy as np

from basicts.runners.callback import BasicTSCallback
from pvtsfm.data.mmsp import canonical_protocol
from basicts.utils import RunnerStatus


def protocol_output_dir(output_dir, protocol, default_dir):
    """Keep custom run directories inside their declared protocol namespace."""
    path = Path(output_dir or default_dir)
    protocol = canonical_protocol(protocol)
    path = Path(*('in_domain' if part in ('fixed_v1', 'paper_main') else part for part in path.parts))
    labels = set(path.parts) & {'in_domain', 'paper_main_v1', 'zeroshot_v1'}
    if labels and labels != {protocol}:
        raise ValueError(f'Output directory {path} conflicts with protocol {protocol}')
    return str(path if protocol in path.parts else path / protocol)


def record_result(root, model, seed):
    """Index completed runs by their saved data protocol, including older runs."""
    model = {'ChronosC0': 'C0', 'TimesFMC0': 'TimesFM3',
             'FusionSFDLinear': 'DLinear', 'FusionSFPatchTST': 'PatchTST',
             'FusionSF3Modal': 'FusionSF'}.get(model, model)
    root = Path(root).resolve()
    protocol = json.loads((root / 'data_protocol.json').read_text())
    if protocol['version'] in ('fixed_v1', 'paper_main'):
        protocol.update(version='in_domain', previous_name='fixed_v1', protocol_origin='project_fixed_v1')
        (root / 'data_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    metrics = json.loads((root / 'metrics.json').read_text())
    metrics.update({
        'data_protocol': protocol['version'], 'model': model, 'seed': seed,
        **{key: protocol[key] for key in (
            'input_len', 'output_len', 'train_sites', 'validation_sites', 'test_sites',
            'split_version', 'scaler_version',
        )},
    })
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    table = Path(__file__).resolve().parents[3] / 'reports' / protocol['version'] / 'results.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    columns = ['data_protocol', 'model', 'seed', 'input_len', 'output_len', 'train_sites',
               'validation_sites', 'test_sites', 'split_version', 'scaler_version',
               'test_windows', 'mae', 'rmse', 'r2', 'result_dir']
    row = {key: metrics[key] for key in columns if key != 'result_dir'}
    row['result_dir'] = str(root)
    with table.open('a+', newline='') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        rows = [old for old in csv.DictReader(handle) if old['result_dir'] != str(root)]
        rows.append(row)
        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return metrics


class MMSPResults(BasicTSCallback):
    def on_train_start(self, runner):
        runner.train_data_loader.dataset.save_protocol(runner.ckpt_save_dir)

    def on_test_end(self, runner):
        if runner.status != RunnerStatus.EVALUATING:
            return
        dataset = runner.test_data_loader.dataset
        root = Path(runner.ckpt_save_dir)
        dataset.save_protocol(root)
        results = root / 'test_results'
        prediction = np.load(results / 'prediction.npy', mmap_mode='r').astype(np.float64)
        targets = np.load(results / 'targets.npy', mmap_mode='r').astype(np.float64)
        error = prediction - targets
        variance = np.square(targets - targets.mean()).sum()
        metrics = {
            'test_windows': len(targets),
            'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.square(error).mean())),
            'r2': float(1 - np.square(error).sum() / variance) if variance > 0 else None,
        }
        (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        metrics = record_result(root, runner.cfg.get('experiment_model_name', runner.model_name), runner.cfg.seed)
        # Dataset order is site-major, then chronological windows; no satellite reload.
        records = [dataset.storage.window_records[i] for i in dataset.window_indices]
        sites = np.repeat([int(s['site']) for s in dataset.storage.data_sp], len(records))
        times = np.array([
            dataset.storage.data_sp_time_dt.iloc[r.start_index + dataset.storage.seq_len:
                                                r.start_index + dataset.storage.seq_len + dataset.storage.pred_len]
            .to_numpy(dtype='datetime64[ns]').astype(np.int64)
            for r in records
        ])
        np.save(results / 'site_ids.npy', sites)
        np.save(results / 'forecast_timestamps.npy', np.tile(times, (len(dataset.storage.data_sp), 1)))
        runner.logger.info(f'Complete {dataset.protocol} evaluation metrics: {metrics}')
