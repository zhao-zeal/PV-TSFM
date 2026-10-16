"""Paired circular time-block bootstrap of the already frozen A01 comparison."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/A01_site0_covariate_ablation_seed42_20261014'


def main():
    names = ['full', 'joint_drop_06_07_08_09_10_13_14']
    losses, references, sources = [], [], []
    for name in names:
        receipt = json.loads((REPORT / f'{name}_test_result.json').read_text())
        path = Path(receipt['output_dir'])
        assert receipt['acceptance']['status'] == 'passed'
        pred, true = [np.load(path / f'{key}.npy').astype(np.float64) for key in ('pred', 'true')]
        times, sites = [np.load(path / f'{key}.npy') for key in ('forecast_timestamps', 'site_ids')]
        assert pred.shape == true.shape == (2545, 24, 1)
        assert np.isfinite(pred).all() and np.isfinite(true).all()
        assert set(sites.tolist()) == {0}
        assert np.all(np.diff(times[:, 0]) == 3600 * 10**9)
        references.append((true, times, sites))
        error = pred - true
        losses.append(np.stack([(error**2).mean(axis=(1, 2)), abs(error).mean(axis=(1, 2))], axis=-1))
        sources.append(str(path))
    assert all(np.array_equal(a, b) for a, b in zip(*references))
    # Keep both models and all 24 horizons paired at each hourly forecast origin.
    values = np.stack(losses, axis=1)
    n = len(values)
    prefix = np.concatenate([np.zeros((1, 2, 2)), np.cumsum(np.tile(values, (2, 1, 1)), axis=0)])
    observed = values.mean(axis=0)
    observed[:, 0] = np.sqrt(observed[:, 0])
    results = []
    for length in (168, 24, 72):
        rng = np.random.default_rng(20261014)
        lengths = np.array([length] * (n // length) + ([n % length] if n % length else []))
        starts = rng.integers(0, n, size=(10000, len(lengths)))
        sample = (prefix[starts + lengths] - prefix[starts]).sum(axis=1) / n
        sample[:, :, 0] = np.sqrt(sample[:, :, 0])
        delta = sample[:, 1] - sample[:, 0]
        result = dict(block_hours=length, primary=length == 168, metrics={})
        for index, metric in enumerate(('rmse', 'mae')):
            result['metrics'][metric] = dict(
                full=float(observed[0, index]), selected=float(observed[1, index]),
                delta=float(observed[1, index] - observed[0, index]),
                relative_percent=float(100 * (observed[1, index] / observed[0, index] - 1)),
                delta_ci95=np.quantile(delta[:, index], [.025, .975]).tolist(),
                bootstrap_improvement_fraction=float((delta[:, index] < 0).mean()))
        results.append(result)
    confirmed = all(r['metrics']['rmse']['delta_ci95'][1] < 0 for r in results)
    report = dict(checked_at=datetime.now().astimezone().isoformat(),
        model='A01 Chronos-2 + B131 + Linear', protocol='site0_v1', training_seed=42,
        input_len=24, output_len=24, windows=n, sources=sources,
        method='paired circular moving-block bootstrap; percentile 95% CI; all horizons retained',
        replicates=10000, bootstrap_seed=20261014, primary_block_hours=168,
        criterion='Primary and sensitivity RMSE delta CI upper endpoints must all be below zero',
        selection='Previously frozen seven-variable deletion; no new selection or training',
        results=results, reliability_confirmed=confirmed,
        next_stage='design_only' if confirmed else 'dynamic_design_not_started',
        limitations=['Single trained seed; does not assess training randomness',
                    'Conditional on this test period and frozen candidate; block bootstrap assumes approximate stationarity',
                    'Overlapping forecasts retained within blocks; block lengths checked for sensitivity',
                    'Improvement fraction is not a p-value or posterior probability',
                    'Original NWP issue-time and interpolation limitations remain'])
    (REPORT / 'static_reliability.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
