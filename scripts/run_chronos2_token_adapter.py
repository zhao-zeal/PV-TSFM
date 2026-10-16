"""Approved T1/T2/T3, two protocols, seed42; serial execution and fresh acceptance."""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/Chronos2TokenAdapter_stage1_seed42_20261013'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def configuration(variant, protocol, gpu, stamp):
    from run import parse_args
    from layers.chronos2_token_adapter import matched_mlp_width
    args = parse_args(['--model', 'Chronos2WeatherAdapter', '--protocol', protocol,
                       '--is_training', '1', '--seed', '42', '--use_gpu', '1', '--gpu', str(gpu), '--num_workers', '0'])
    args.model = 'Chronos2TokenAdapter'
    args.token_variant = variant
    args.mlp_width = matched_mlp_width()
    args.adapter_version = f'token_{variant}_v1'
    args.experiment_label = variant
    args.baseline_config = 'native_token_adapter_controlled_v1'
    args.model_id = f'Chronos2TokenAdapter_{variant}_seed42_sl24_pl24_{stamp}'
    args.token_cache_id = f'Chronos2TokenAdapter_native_cache_v1_seed42_sl24_pl24_{stamp}'
    args.token_layout = 'PV then NWP15; native history2,future2; REG excluded'
    args.adapter_width, args.adapter_heads, args.adapter_ff, args.adapter_scales = 64, 4, 128, 2
    args.protocol_limit = 'NWP bidirectional interpolation and assumed issue-time availability retained'
    return args


def accept(exp, output, checkpoint):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from utils.acceptance import accept_and_record
    from utils.metrics import metric
    exp.load_selected(checkpoint / 'checkpoint.pth')
    exp.model.eval()
    pred = np.load(output / 'pred.npy')
    truth = np.load(output / 'true.npy')
    sites = np.load(output / 'site_ids.npy')
    times = np.load(output / 'forecast_timestamps.npy')
    reference = ROOT / 'outputs' / exp.args.protocol / 'Chronos2WeatherAdapter_A02_seed42_sl24_pl24_history_future'
    for name, array in (('true', truth), ('site_ids', sites), ('forecast_timestamps', times)):
        assert np.array_equal(array, np.load(reference / f'{name}.npy')), name
    dataset, _ = exp._get_data('test')
    cursor, max_difference = 0, 0.
    # Full selected-checkpoint replay from raw shared Dataset_MMSP, bypassing token caches.
    with torch.no_grad():
        for power, target, _, _, cov in DataLoader(dataset, batch_size=exp.args.cache_batch_size, num_workers=0):
            replay = exp.model(power.to(exp.device), None, None, None,
                {k: v.to(exp.device) for k, v in cov.items()}).cpu().numpy()
            stop = cursor + len(power)
            difference = float(np.abs(replay - pred[cursor:stop]).max())
            assert difference <= 2e-6, (cursor, difference)
            assert np.array_equal(target[:, -24:].numpy(), truth[cursor:stop])
            assert np.array_equal(cov['site_id'].numpy(), sites[cursor:stop])
            assert np.array_equal(cov['forecast_timestamps'].numpy(), times[cursor:stop])
            max_difference = max(max_difference, difference)
            cursor = stop
    assert cursor == len(pred)
    base = np.load(reference / 'pred.npy')
    base_metrics = metric(base, truth)
    result = metric(pred, truth)
    worst = np.argsort(((base.astype(np.float64) - truth) ** 2).mean(axis=(1, 2)))[-int(np.ceil(len(base) * .1)):]
    analysis = dict(overall=result, A02=base_metrics,
        change_from_A02={k: dict(absolute=result[k]-base_metrics[k], relative_percent=100*(result[k]/base_metrics[k]-1)) for k in ('mae', 'rmse')},
        per_site={str(site): metric(pred[sites == site], truth[sites == site]) for site in np.unique(sites)},
        per_hour=[dict(hour=i+1, **metric(pred[:, i], truth[:, i])) for i in range(24)],
        A02_worst10_common_subset=dict(current=metric(pred[worst], truth[worst]), A02=metric(base[worst], truth[worst])),
        limits=['seed42 exploratory only', 'T3 tests the temporal/PCorr/multiscale bundle', exp.args.protocol_limit])
    save(output / 'analysis.json', analysis)
    checks = dict(full_raw_selected_checkpoint_replay_windows=cursor, max_abs_difference=max_difference,
                  same_A02_labels_sites_times=True, frozen_gradients_absent=all(p.grad is None for p in exp.model.base.parameters()),
                  preflight=str(checkpoint / 'preflight.json'))
    return accept_and_record(output, specific_checks=checks)


def worker(opts):
    from scripts.run_chronos2_history_weather_adapter import configure, reseed
    np, torch = configure(opts.gpu)
    from exp.exp_chronos2_token_adapter import Exp_Chronos2_Token_Adapter
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
    reseed(torch, np)
    args = configuration(opts.variant, opts.protocol, opts.gpu, opts.stamp)
    exp = Exp_Chronos2_Token_Adapter(args)
    exp.train(args.model_id)
    metrics = exp.test(args.model_id)
    out = Path(args.output_dir) / args.protocol / args.model_id
    ck = Path(args.checkpoints) / args.protocol / args.model_id
    receipt = accept(exp, out, ck)
    summary = json.loads((ck / 'training_summary.json').read_text())
    save(REPORT / f'{opts.protocol}_{opts.variant}_result.json', dict(metrics=metrics, acceptance=receipt,
        training=summary, output_dir=str(out), checkpoint_dir=str(ck)))


def preflight(opts):
    from scripts.run_chronos2_history_weather_adapter import configure, reseed
    np, torch = configure(opts.gpu)
    from data_provider.data_factory import data_provider
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from exp.exp_chronos2_token_adapter import Exp_Chronos2_Token_Adapter
    results = []
    for protocol in ('in_domain', 'zeroshot_v1'):
        for variant in ('T1', 'T2', 'T3'):
            reseed(torch, np)
            exp = Exp_Chronos2_Token_Adapter(configuration(variant, protocol, opts.gpu, opts.stamp))
            assert exp.device.type == 'cuda'
            data, _ = data_provider(exp.args, 'train')
            indices = [0, len(data)//2, len(data)-1]
            subset = Subset(data, indices)
            power, target, _, _, cov = next(iter(DataLoader(subset, batch_size=3)))
            tokens, base, meta, corr, pos = exp.model.encode_all(power.to(exp.device),
                cov['historical_nwp'].to(exp.device), cov['future_nwp'].to(exp.device))
            exp.positions = pos
            original = ROOT/'outputs'/protocol/'Chronos2Adapter_features_v1_seed42_sl24_pl24_history_future/train'
            old = torch.from_numpy(np.load(original/'embeds.npy', mmap_mode='c')[indices]).to(exp.device)
            assert torch.allclose(tokens[:,0,-2:], old, atol=3e-5, rtol=1e-5)
            tensors = [v.cpu() for v in (tokens, base, cov['future_nwp'], meta, corr,
                       target[:, -24:], cov['site_id'], cov['forecast_timestamps'])]
            loader = DataLoader(TensorDataset(*tensors), batch_size=3)
            receipt = exp.check_initialization(subset, loader, None)
            results.append(dict(protocol=protocol, variant=variant, device=str(exp.device),
                                historical_A01_cache_max_abs=(tokens[:,0,-2:]-old).abs().max().item(), **receipt))
            print(f'{protocol}/{variant} real-batch preflight passed', flush=True)
            del exp, tokens, base, meta, corr, data, subset, loader
            torch.cuda.empty_cache()
    save(REPORT/'preflight.json', dict(status='passed', checks=results))


def compare_validation(record):
    pairs = [('T2', 'T1'), ('T3', 'T2')]
    results = {}
    for candidate, control in pairs:
        evidence = {}
        for protocol in ('in_domain', 'zeroshot_v1'):
            runs = {r['variant']: r for r in record['runs'] if r['protocol'] == protocol}
            a = runs[candidate]['result']['training']['best_validation_metrics']
            b = runs[control]['result']['training']['best_validation_metrics']
            evidence[protocol] = dict(rmse_reduction_percent=100*(1-a['rmse']/b['rmse']),
                                     mae_degradation_percent=100*(a['mae']/b['mae']-1))
        results[f'{candidate}_vs_{control}'] = dict(protocols=evidence, passes=all(
            v['rmse_reduction_percent'] >= 1 and v['mae_degradation_percent'] <= .5 for v in evidence.values()))
    return dict(comparisons=results, further_stages_eligible=any(v['passes'] for v in results.values()),
                criterion='Both protocols: validation RMSE reduction >=1%, MAE degradation <=0.5%; seed42 exploratory',
                test_used_for_selection=False)


def sync_table(record):
    table = ROOT / 'reports/chronos2_adapter_metrics.csv'
    with table.open(encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
    contents = doc.read_text()
    import re
    for run in record['runs']:
        if run['state'] != 'accepted':
            continue
        label = f'{run["variant"]} (Native Token Adapter)'
        row = next((r for r in rows if r['models'] == label), None)
        if row is None:
            row = {key: '' for key in fields}
            row.update(models=label, input='历史功率+历史/未来NWP（'+('PV未来Token+MLP' if run['variant']=='T1' else '全变量Token+'+('MLP' if run['variant']=='T2' else 'Cross-Unet风格Adapter'))+'）')
            rows.append(row)
        prefix = 'MMSP in_domain' if run['protocol'] == 'in_domain' else 'MMSP zeroshot'
        for key, metric in (('MAE','mae'), ('RMSE','rmse'), ('R^2','r2')):
            row[f'{prefix} {key}'] = f'{run["result"]["metrics"][metric]:.6f}'
        line = '| '+' | '.join(row[key] or '—' for key in fields)+' |'
        pattern = r'^\| '+re.escape(label)+r' \|.*$'
        if re.search(pattern, contents, re.MULTILINE):
            contents = re.sub(pattern, lambda _: line, contents, flags=re.MULTILINE)
        else:
            contents = re.sub(r'(^\| A04-b \(Weather-Conditioned Chronos Readout\) \|.*$)',
                              lambda m: m[1]+'\n'+line, contents, count=1, flags=re.MULTILINE)
    with table.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    doc.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--retry-untrained', action='store_true', help='Retry only a failed attempt with no training epochs/checkpoints')
    parser.add_argument('--variant', choices=['T1', 'T2', 'T3'])
    parser.add_argument('--protocol', choices=['in_domain', 'zeroshot_v1'])
    parser.add_argument('--stamp', default=datetime.now().strftime('%Y%m%dT%H%M%S'))
    opts = parser.parse_args()
    os.sched_setaffinity(0, [9, 10])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if opts.preflight:
        preflight(opts)
        return
    if opts.worker:
        worker(opts)
        return
    failed_attempts = []
    if opts.retry_untrained:
        previous = json.loads((REPORT/'runs.json').read_text())
        assert previous['state']=='failed'
        assert not any(r['state']=='accepted' for r in previous['runs'])
        for path in (ROOT/'checkpoints').glob(f'*/Chronos2TokenAdapter_*_{previous["stamp"]}/*'):
            assert path.suffix!='.pth' and path.name!='training_history.csv', 'Trained runs cannot restart here'
        for path in list(REPORT.glob('*.log')) + [REPORT/'preflight.json']:
            if path.exists(): path.rename(path.with_name(f'failed_{previous["stamp"]}_{path.name}'))
        previous['failure_diagnosis'] = 'Missing use_gpu=1 caused CPU native normalization, inconsistent with GPU A01 cache; no training occurred'
        previous['preflight_device'] = 'cpu; superseded by corrected GPU checks'
        failed_attempts = previous.get('failed_attempts', []) + [previous]
    else:
        REPORT.mkdir(parents=True, exist_ok=False)
    # Parameter match and shared initial values before looking at any validation outcomes.
    import torch
    from layers.chronos2_token_adapter import TokenResidual, matched_mlp_width
    models = {}
    for variant in ('T1', 'T2', 'T3'):
        torch.manual_seed(42)
        models[variant] = TokenResidual(variant, matched_mlp_width())
    counts = {k: sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    assert counts['T1'] == counts['T2'] and abs(counts['T2']/counts['T3']-1) <= .05
    for k, value in models['T1'].state_dict().items():
        assert torch.equal(value, models['T2'].state_dict()[k])
        if not k.startswith('mixer.'):
            assert torch.equal(value, models['T3'].state_dict()[k])
    record = dict(state='running', gpu=opts.gpu, stamp=opts.stamp, pid=os.getpid(),
        cpu_cores=[9,10], cpu_threads=2, num_workers=0, seed=42, input_len=24, output_len=24,
        authorized_scope='Reviewed three-stage proposal; stage1 six runs, validation gates for later stages',
        parameter_counts=counts, mlp_width=matched_mlp_width(), shared_initialization_checked=True,
        failed_attempts=failed_attempts, runs=[])
    for protocol in ('in_domain', 'zeroshot_v1'):
        for variant in ('T1','T2','T3'):
            record['runs'].append(dict(protocol=protocol, variant=variant, state='pending',
                                      log=str(REPORT/f'{protocol}_{variant}.log')))
    save(REPORT/'runs.json', record)
    from utils.resources import require_idle_gpus
    try:
        require_idle_gpus(str(opts.gpu))
        with (REPORT/'preflight.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, '--preflight', '--gpu', str(opts.gpu),
                            '--stamp', opts.stamp], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        time.sleep(3)
        for row in record['runs']:
            require_idle_gpus(str(opts.gpu))
            row.update(state='running', started_at=datetime.now().isoformat())
            save(REPORT/'runs.json', record)
            command = [sys.executable, '-u', __file__, '--worker', '--protocol', row['protocol'],
                       '--variant', row['variant'], '--gpu', str(opts.gpu), '--stamp', opts.stamp]
            with Path(row['log']).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                save(REPORT/'runs.json', record)
                code = process.wait()
            if code:
                raise RuntimeError(f'{row["protocol"]}/{row["variant"]} exited {code}; see {row["log"]}')
            row.update(state='accepted', completed_at=datetime.now().isoformat(), result=json.loads(
                (REPORT/f'{row["protocol"]}_{row["variant"]}_result.json').read_text()))
            save(REPORT/'runs.json', record)
            sync_table(record)
            print(f'{row["protocol"]}/{row["variant"]} accepted', flush=True)
            time.sleep(3)
        record.update(state='stage1_accepted', validation_gate=compare_validation(record))
    except Exception as error:
        record.update(state='failed', error=str(error))
        for row in record['runs']:
            if row['state']=='running': row['state']='failed'
        raise
    finally:
        save(REPORT/'runs.json', record)


if __name__ == '__main__':
    main()
