"""A01 site-0 NWP leave-one-out, validation-only selection, then final test confirmation."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = 'site0_v1'
SEED = 42


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def feature_names():
    with (ROOT / 'datasets/MMSP/data/nwp/nwp.csv').open() as handle:
        names = next(csv.reader(handle))
    return [name for name in names if name not in ('fcst_date', 'lat', 'lon')]


def configuration(indices, subset_id, gpu, model_id, *, checkpoint=None):
    from run import parse_args
    argv = ['--model', 'Chronos2Adapter', '--adapter_version', 'b131_linear_v1',
            '--protocol', PROTOCOL, '--seed', str(SEED), '--seq_len', '24', '--pred_len', '24',
            '--num_workers', '0', '--use_gpu', '1', '--gpu', str(gpu), '--model_id', model_id,
            '--is_training', '0' if checkpoint else '1']
    if checkpoint:
        argv += ['--checkpoint', str(checkpoint)]
    args = parse_args(argv)
    names = feature_names()
    args.nwp_indices = list(indices)
    args.nwp_feature_names = [names[index] for index in indices]
    args.nwp_subset_id = subset_id
    args.baseline_config = 'covariate_ablation'
    args.experiment_label = 'A01_site0_covariate_ablation'
    args.variable_selection_metric = 'validation_rmse'
    args.test_selection_policy = 'test untouched until full and joint candidates fixed by validation'
    return args


def configure(gpu):
    from run import configure_resources
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    args = type('ResourceArgs', (), dict(model='Chronos2Adapter', use_gpu=1, gpu=gpu, num_workers=0))()
    configure_resources(args)
    return np, torch


def validation_predictions(experiment, model_id):
    np, torch = __import__('numpy'), __import__('torch')
    dataset, loader = experiment._get_data('val')
    destination = Path(experiment.args.output_dir) / PROTOCOL / model_id / 'validation'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.json').write_text(json.dumps(vars(experiment.args), indent=2) + '\n')
    protocol = dataset.save_protocol(destination)
    predictions, targets, sites, times = [], [], [], []
    experiment.model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction, target, covariates = experiment._forward_batch(batch)
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            sites.append(covariates['site_id'].cpu().numpy())
            times.append(covariates['forecast_timestamps'].cpu().numpy())
    arrays = dict(pred=np.concatenate(predictions), true=np.concatenate(targets),
                  site_ids=np.concatenate(sites), forecast_timestamps=np.concatenate(times))
    for name, value in arrays.items():
        np.save(destination / f'{name}.npy', value)
    from utils.metrics import metric
    metrics = metric(arrays['pred'], arrays['true'])
    metrics.update(windows=len(arrays['pred']), split='validation', protocol=PROTOCOL,
                   selection_use='candidate comparison only; no test metrics exist')
    save(destination / 'metrics.json', metrics)
    assert protocol['windows'] == dict(train=7657, validation=2545, test=2545)
    assert arrays['pred'].shape == arrays['true'].shape == (2545, 24, 1)
    assert set(arrays['site_ids'].tolist()) == {0}
    return destination, metrics


def train_worker(options):
    np, torch = configure(options.gpu)
    from exp.exp_chronos2_adapter import Exp_Chronos2_Adapter
    from utils.run_artifacts import reserve_run
    indices = [int(value) for value in options.indices.split(',')]
    args = configuration(indices, options.subset_id, options.gpu, options.model_id)
    reserve_run(args, args.model_id)
    experiment = Exp_Chronos2_Adapter(args)
    experiment._run_prepared = args.model_id
    experiment.train(args.model_id)
    validation_dir, metrics = validation_predictions(experiment, args.model_id)
    checkpoint = Path(args.checkpoints) / PROTOCOL / args.model_id / 'checkpoint.pth'
    summary = json.loads((checkpoint.parent / 'training_summary.json').read_text())
    result = dict(state='validation_complete', subset_id=options.subset_id, indices=indices,
                  feature_names=args.nwp_feature_names, validation_metrics=metrics,
                  validation_dir=str(validation_dir), checkpoint=str(checkpoint),
                  checkpoint_summary=summary, completed_at=now())
    save(options.result, result)


def test_worker(options):
    configure(options.gpu)
    from exp.exp_chronos2_adapter import Exp_Chronos2_Adapter
    indices = [int(value) for value in options.indices.split(',')]
    args = configuration(indices, options.subset_id, options.gpu, options.model_id,
                         checkpoint=options.checkpoint)
    experiment = Exp_Chronos2_Adapter(args)
    metrics = experiment.test(args.model_id, test=1)
    output = Path(args.output_dir) / PROTOCOL / args.model_id
    receipt = json.loads((output / 'acceptance.json').read_text())
    assert receipt['status'] == 'passed'
    save(options.result, dict(state='test_accepted', subset_id=options.subset_id,
         indices=indices, feature_names=args.nwp_feature_names, metrics=metrics,
         output_dir=str(output), checkpoint=str(options.checkpoint), acceptance=receipt,
         completed_at=now()))


def wait_for_baselines(path, status):
    while True:
        record = json.loads(path.read_text())
        state = record['state']
        status.update(state='waiting_for_baseline_acceptance', baseline_state=state,
                      last_baseline_check=now())
        yield state
        if state == 'completed_artifact_acceptance_passed':
            return
        if state in ('failed_stopped', 'cancelled', 'blocked'):
            raise RuntimeError(f'Baseline prerequisite ended with {state}')
        time.sleep(60)


def idle_gpu(exclude=()):
    from utils.resources import require_idle_gpus
    for gpu in range(8):
        if gpu in exclude:
            continue
        try:
            require_idle_gpus(str(gpu))
            return gpu
        except ValueError:
            pass
    raise RuntimeError('No completely idle GPU')


def wait_idle_gpu(exclude, report, status):
    while True:
        try:
            return idle_gpu(exclude)
        except RuntimeError:
            status.update(state='waiting_for_idle_gpu', last_gpu_check=now())
            save(report / 'runs.json', status)
            time.sleep(60)


def environment():
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    return env


def run_group(report, tasks, status):
    pending = list(tasks)
    while pending:
        active = []
        for lane, task in enumerate(pending[:2]):
            gpu = wait_idle_gpu([item[1] for item in active], report, status)
            cores = [9, 10] if lane == 0 else [0, 7]
            log = report / f'{task["subset_id"]}_{task["kind"]}.log'
            result = report / f'{task["subset_id"]}_{task["kind"]}_result.json'
            command = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, '-u', __file__,
                       '--worker', task['kind'], '--gpu', str(gpu), '--subset-id', task['subset_id'],
                       '--indices', ','.join(map(str, task['indices'])), '--model-id', task['model_id'],
                       '--result', str(result), '--seed', str(SEED)]
            if task['kind'] == 'test':
                command += ['--checkpoint', task['checkpoint']]
            with log.open('x') as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=environment(),
                                           stdout=handle, stderr=subprocess.STDOUT)
            task.update(state='running', gpu=gpu, cpu_cores=cores, pid=process.pid,
                        log=str(log), result=str(result), started_at=now())
            active.append((task, gpu, process))
        status['active'] = [item[0] for item in active]
        save(report / 'runs.json', status)
        for task, _, process in active:
            task.update(returncode=process.wait(), completed_at=now())
            task['state'] = 'complete' if task['returncode'] == 0 else 'failed'
            save(report / 'runs.json', status)
        if any(task['returncode'] for task, _, _ in active):
            raise RuntimeError('A01 covariate worker failed; see task log')
        pending = pending[len(active):]
    status['active'] = []


def task(subset_id, indices, stamp, kind='train', checkpoint=None):
    suffix = 'validation_train' if kind == 'train' else 'final_test'
    return dict(subset_id=subset_id, indices=list(indices), kind=kind, state='pending',
                checkpoint=str(checkpoint) if checkpoint else None,
                model_id=f'Chronos2Adapter_A01_site0_{subset_id}_{suffix}_seed{SEED}_sl24_pl24_{stamp}')


def load_result(report, item):
    return json.loads((report / f'{item["subset_id"]}_{item["kind"]}_result.json').read_text())


def collect_weather():
    from data_provider.data_factory import data_provider
    from run import parse_args
    import numpy as np
    args = parse_args(['--model', 'Chronos2Adapter', '--protocol', PROTOCOL, '--print_config'])
    values = {}
    for flag in ('train', 'val'):
        dataset, _ = data_provider(args, flag)
        rows = []
        for index in range(len(dataset)):
            rows.append(dataset[index][4]['future_nwp'].numpy().mean(axis=0))
        values[flag] = np.stack(rows)
    return values


def conditional_analysis(report, candidates, joint):
    import numpy as np
    from utils.metrics import metric
    weather = collect_weather()
    full = load_result(report, candidates[0])
    full_dir = Path(full['validation_dir'])
    base_pred = np.load(full_dir / 'pred.npy')
    truth = np.load(full_dir / 'true.npy')
    sites = np.load(full_dir / 'site_ids.npy')
    times = np.load(full_dir / 'forecast_timestamps.npy')
    analyses = []
    comparisons = candidates[1:] + ([joint] if joint else [])
    for candidate in comparisons:
        result = load_result(report, candidate)
        directory = Path(result['validation_dir'])
        pred = np.load(directory / 'pred.npy')
        assert np.array_equal(truth, np.load(directory / 'true.npy'))
        assert np.array_equal(sites, np.load(directory / 'site_ids.npy')) and set(sites.tolist()) == {0}
        assert np.array_equal(times, np.load(directory / 'forecast_timestamps.npy'))
        removed = sorted(set(range(15)) - set(candidate['indices']))
        conditions = {}
        for index in removed:
            low, high = np.quantile(weather['train'][:, index], [1 / 3, 2 / 3])
            state = np.where(weather['val'][:, index] <= low, 'low',
                             np.where(weather['val'][:, index] <= high, 'medium', 'high'))
            groups = {}
            for label in ('low', 'medium', 'high'):
                mask = state == label
                if not mask.any():
                    continue
                current, baseline = metric(pred[mask], truth[mask]), metric(base_pred[mask], truth[mask])
                groups[label] = dict(windows=int(mask.sum()), current=current, full=baseline,
                    delta={key: current[key] - baseline[key] for key in ('mae', 'rmse', 'r2')})
            conditions[feature_names()[index]] = dict(train_thresholds=[float(low), float(high)], groups=groups)
        overall, baseline = metric(pred, truth), metric(base_pred, truth)
        per_horizon = []
        for h in range(24):
            current, base = metric(pred[:, h], truth[:, h]), metric(base_pred[:, h], truth[:, h])
            per_horizon.append(dict(horizon=h + 1, current=current, full=base,
                                    delta={key: current[key] - base[key] for key in ('mae', 'rmse', 'r2')}))
        analyses.append(dict(subset_id=candidate['subset_id'], removed_indices=removed,
            removed_features=[feature_names()[index] for index in removed],
            site0=dict(current=overall, full=baseline,
                       delta={key: overall[key] - baseline[key] for key in ('mae', 'rmse', 'r2')}),
            per_horizon=per_horizon,
            weather_states=conditions))
    analysis = dict(split='validation', sites=[0], site_analysis_note='Single-site task: site grouping has one group only',
                    weather_state_definition='For each removed variable, low/medium/high uses train-window tertiles of mean future standardized NWP; thresholds never use validation/test labels',
                    prediction_horizons=24, comparisons=analyses,
                    test_used_for_condition_analysis=False)
    save(report / 'conditional_analysis.json', analysis)
    return analysis


def execute(options):
    report = options.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    if (report / 'runs.json').exists():
        raise FileExistsError('This covariate task already has a coordinator record')
    names = feature_names()
    assert len(names) == 15
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    status = dict(state='waiting_for_baseline_acceptance', created_at=now(), seed=SEED,
                  protocol=PROTOCOL, model='A01 Chronos-2 + B131 + Linear', seq_len=24, pred_len=24,
                  fixed_training='A01 max100/patience10, L1, AdamW; B131 lr0.0016, Linear lr0.000016; checkpoint selected by validation MAE',
                  variable_selection='validation RMSE only; strict improvement over Full identifies negative contribution',
                  feature_names=names, stages={}, baseline_report=str(options.baseline.resolve()))
    save(report / 'runs.json', status)
    for state in wait_for_baselines(options.baseline.resolve(), status):
        save(report / 'runs.json', status)
    status.update(state='stage1_running', baseline_accepted_at=now())
    full = task('full', range(15), stamp)
    loo = [task(f'loo_{index:02d}', [i for i in range(15) if i != index], stamp) for index in range(15)]
    stage1 = [full, *loo]
    status['stages']['stage1'] = stage1
    run_group(report, stage1, status)
    results = {item['subset_id']: load_result(report, item) for item in stage1}
    full_rmse = results['full']['validation_metrics']['rmse']
    negative = [index for index in range(15)
                if results[f'loo_{index:02d}']['validation_metrics']['rmse'] < full_rmse]
    ranking = sorted([dict(index=index, feature=names[index],
                           validation_rmse=results[f'loo_{index:02d}']['validation_metrics']['rmse'],
                           delta_rmse=results[f'loo_{index:02d}']['validation_metrics']['rmse'] - full_rmse,
                           validation_mae=results[f'loo_{index:02d}']['validation_metrics']['mae'])
                      for index in range(15)], key=lambda x: x['delta_rmse'])
    selection = dict(frozen_at=now(), full_validation=results['full']['validation_metrics'],
                     negative_indices=negative, negative_features=[names[index] for index in negative],
                     leave_one_out_ranking=ranking, test_metrics_seen=False)
    save(report / 'stage1_selection.json', selection)
    status.update(state='stage2_running')
    joint_indices = [index for index in range(15) if index not in negative]
    joint = None
    if negative and joint_indices:
        joint = task('joint_drop_' + '_'.join(f'{index:02d}' for index in negative), joint_indices, stamp)
        status['stages']['stage2'] = [joint]
        run_group(report, [joint], status)
        joint_result = load_result(report, joint)
        joint_better = joint_result['validation_metrics']['rmse'] < full_rmse
    else:
        joint_better = False
        status['stages']['stage2'] = []
    selected = joint if joint_better else full
    decision = dict(frozen_at=now(), selected_subset=selected['subset_id'], selected_indices=selected['indices'],
                    selected_features=[names[index] for index in selected['indices']],
                    joint_subset=None if joint is None else joint['subset_id'],
                    full_validation_rmse=full_rmse,
                    joint_validation_metrics=None if joint is None else load_result(report, joint)['validation_metrics'],
                    rule='joint selected only if its validation RMSE is strictly lower than Full',
                    test_metrics_seen=False)
    save(report / 'stage2_decision.json', decision)
    status.update(state='stage3_validation_analysis')
    conditional_analysis(report, stage1, joint)
    # Test is first touched only after the selection decision is written and immutable as evidence.
    final_candidates = [full] + ([joint] if joint is not None else [])
    tests = [task(item['subset_id'], item['indices'], stamp, kind='test',
                  checkpoint=load_result(report, item)['checkpoint']) for item in final_candidates]
    status['stages']['final_test'] = tests
    status.update(state='final_test_running', selection_frozen_before_test=True)
    run_group(report, tests, status)
    test_results = {item['subset_id']: load_result(report, item) for item in tests}
    final = dict(state='completed_artifact_acceptance_passed', completed_at=now(),
                 stage1_selection=selection, stage2_decision=decision,
                 final_test={key: value['metrics'] for key, value in test_results.items()},
                 conditional_analysis=str(report / 'conditional_analysis.json'),
                 limitations=['seed42 only', 'site0 only; no cross-site condition inference',
                              'NWP issue time absent and bidirectional interpolation retained',
                              'strict validation improvement has no uncertainty threshold'])
    save(report / 'summary.json', final)
    status.update(state=final['state'], completed_at=final['completed_at'], active=[])
    save(report / 'runs.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--worker', choices=['train', 'test'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--subset-id')
    parser.add_argument('--indices')
    parser.add_argument('--model-id')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    options = parser.parse_args()
    SEED = options.seed
    if options.worker == 'train':
        train_worker(options)
    elif options.worker == 'test':
        test_worker(options)
    else:
        if not options.report or not options.baseline:
            parser.error('--report and --baseline are required for the coordinator')
        execute(options)
