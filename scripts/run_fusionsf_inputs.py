"""Run the three authorized FusionSF inputs after the recorded prerequisite passes."""

import argparse
import ast
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/FusionSF_inputs_two_protocols_seed42_20261011'
STATUS = REPORT / 'runs.json'
TIMEVLM = ROOT / 'reports/TimeVLM_two_protocols_seed42_20261011/runs.json'
VARIANTS = {
    'power_future_nwp': '历史功率+未来NWP',
    'power_satellite_history_future_nwp': '历史功率+历史卫星+历史/未来NWP',
    'power_history_future_nwp': '历史功率+历史/未来NWP',
}
CORES = [0, 7]


def now():
    return datetime.now().astimezone().isoformat()


def save(record):
    STATUS.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    runs = []
    for variant, label in VARIANTS.items():
        for protocol in ('in_domain', 'zeroshot_v1'):
            model_id = f'FusionSF_{variant}_seed42_sl24_pl24'
            argv = ['--model', 'FusionSF', '--fusion_input', variant,
                    '--protocol', protocol, '--is_training', '1', '--seed', '42',
                    '--seq_len', '24', '--pred_len', '24', '--num_workers', '0',
                    '--use_gpu', '1', '--model_id', model_id]
            runs.append(dict(variant=variant, input=label, protocol=protocol,
                             model_id=model_id, argv=argv, state='pending',
                             checkpoint_dir=str(ROOT / 'checkpoints' / protocol / model_id),
                             output_dir=str(ROOT / 'outputs' / protocol / model_id),
                             log=str(REPORT / f'{variant}_{protocol}.log')))
    record = dict(state='prepared_waiting_timevlm', created_at=now(),
                  authorization='User requested these three inputs after TimeVLM; sequential execution only',
                  prerequisite=str(TIMEVLM), seed=42, seq_len=24, pred_len=24,
                  cpu_cores=CORES, threads=2, num_workers=0, max_simultaneous_gpus=1,
                  history_nwp_integration='Concatenate 15 train-standardized historical NWP channels with historical power at matching times; keep future NWP guide',
                  architecture='Original script backbone/ReLU/VQ in TS; remove satellite branch and context VQ for inputs without satellite',
                  training='batch/eval16; FP32; AdamW lr0.0016 wd0.05 betas0.9/0.95; L1+VQ; warmup5 cosine100; max100 epochs; patience20; minimum validation MAE',
                  evaluation='All25450x24; global MAE/RMSE/R2; no clipping/night mask; artifacts accepted before continuing',
                  runs=runs)
    save(record)
    sync_table(record)
    return record


def active(run):
    try:
        command = Path(f'/proc/{run["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    model_id = run.get('model_id') or run['argv'][run['argv'].index('--model_id') + 1]
    return model_id in command and 'run.py' in command


def idle_gpu(preferred=None):
    from utils.resources import require_idle_gpus
    while True:
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index',
                                          '--format=csv,noheader,nounits'], text=True)
        indices = [int(x) for x in devices.splitlines()]
        if preferred is not None:
            indices = [preferred]
        for index in indices:
            try:
                require_idle_gpus(str(index))
                return index
            except ValueError:
                pass
        time.sleep(30)


def watch_training(process, gpu, path):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'gpu', 'memory_mib', 'utilization_percent'])
        while process.poll() is None:
            try:
                values = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                    '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                writer.writerow([now(), *[value.strip() for value in values.strip().split(',')]])
                handle.flush()
            except subprocess.CalledProcessError:
                writer.writerow([now(), gpu, 'unavailable', 'unavailable'])
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    return process.returncode


def preflight(run, gpu):
    import torch
    from run import parse_args, configure_resources
    from data_provider.data_factory import data_provider
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.losses import forecasting_loss
    from torch.utils.data._utils.collate import default_collate
    argv = run['argv'] + (['--gpu', str(gpu)] if gpu is not None else ['--use_gpu', '0'])
    args = parse_args(argv)
    configure_resources(args)
    torch.manual_seed(42)
    dataset, _ = data_provider(args, 'train')
    assert len(dataset) == 76570
    batch = default_collate([dataset[i] for i in (12, 13)])
    assert batch[4]['historical_nwp'].shape == (2, 24, 15)
    assert batch[4]['future_nwp'].shape == (2, 24, 15)
    assert torch.isfinite(batch[4]['historical_nwp']).all()
    assert torch.isfinite(batch[4]['future_nwp']).all()
    experiment = Exp_Long_Term_Forecast(args)
    model = experiment.model
    expected = torch.cat([batch[0], batch[4]['historical_nwp']], -1) if model.use_history_nwp else batch[0]
    captured = []
    hook = model.backbone.ts_embedding.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu()))
    model.train()
    result, targets, _ = experiment._forward_batch(batch)
    hook.remove()
    assert torch.equal(captured[0][..., :expected.size(-1)], expected)
    assert result['prediction'].shape == targets.shape == (2, 24, 1)
    loss = forecasting_loss(result, targets, 'FusionSF')
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not model.use_satellite:
        assert not hasattr(model.backbone, 'ctx_encoder') and not hasattr(model.backbone, 'mixer')
        assert batch[4]['satellite'].numel() == 0
    model.eval()
    with torch.no_grad():
        prediction = experiment._forward_batch(batch)[0]['prediction']
    assert torch.isfinite(prediction).all() and (prediction >= 0).all()
    return dict(status='passed_real_batch_forward_backward', checked_at=now(),
                prediction_shape=list(prediction.shape), loss=float(loss.detach()),
                historical_nwp_consumed=model.use_history_nwp,
                satellite_consumed=model.use_satellite,
                power_and_history_embedding_columns_verified=True,
                performance_reproduction_complete=False, config=vars(args))


def accept(run, gpu):
    import numpy as np
    import pandas as pd
    import torch
    from run import parse_args, configure_resources
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.metrics import metric
    from utils.tools import load_model_checkpoint
    out, ckpt = Path(run['output_dir']), Path(run['checkpoint_dir'])
    from utils.acceptance import accept_and_record
    accept_and_record(out)
    config = json.loads((out / 'config.json').read_text())
    assert config == json.loads((ckpt / 'config.json').read_text())
    assert config['fusion_input'] == run['variant'] and config['seed'] == 42
    assert config['baseline_config'] == 'fusion_input_variant'
    protocol = json.loads((out / 'data_protocol.json').read_text())
    training = json.loads((ckpt / 'data_protocol.json').read_text())
    for key in ('version', 'train_sites', 'validation_sites', 'test_sites', 'windows', 'target_time_ranges', 'scaler_fit'):
        assert protocol[key] == training[key], key
    assert protocol['version'] == run['protocol']
    assert protocol['windows'] == dict(train=76570, validation=25450, test=25450)
    assert protocol['train_sites'] == list(range(10) if run['protocol'] == 'in_domain' else range(10, 20))
    assert protocol['test_sites'] == list(range(10))
    baseline_record = json.loads((ROOT / 'reports/FusionSF_two_protocols_seed42_20261010/runs.json').read_text())
    assert baseline_record['acceptance']['status'] == 'passed'
    baseline_run = next(item for item in baseline_record['acceptance']['runs'] if item['protocol'] == run['protocol'])
    reference = Path(baseline_run['output_dir'])
    with np.load(ckpt / 'scalers.npz') as train_scalers, np.load(out / 'scalers.npz') as test_scalers, np.load(reference / 'scalers.npz') as reference_scalers:
        assert train_scalers.files == test_scalers.files
        for key in train_scalers.files:
            assert np.array_equal(train_scalers[key], test_scalers[key]), key
            assert np.array_equal(train_scalers[key], reference_scalers[key]), key
    arrays = {name: np.load(out / f'{name}.npy') for name in ('pred', 'true', 'site_ids', 'forecast_timestamps')}
    pred, truth, sites, times = [arrays[k] for k in ('pred', 'true', 'site_ids', 'forecast_timestamps')]
    assert pred.shape == truth.shape == (25450, 24, 1)
    assert np.isfinite(pred).all() and np.isfinite(truth).all() and np.all(pred >= 0)
    for name in ('true', 'site_ids', 'forecast_timestamps'):
        assert np.array_equal(arrays[name], np.load(reference / f'{name}.npy')), name
    raw = pd.read_csv(ROOT / 'datasets/MMSP/data/solar_power/solar_power.csv', parse_dates=['datetime'])
    for site in range(10):
        mask = sites == site
        assert mask.sum() == 2545 and np.any(pred[mask] != 0), f'zero station {site}'
        group = raw[raw.site == site]
        raw_times = group.datetime.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        indices = np.searchsorted(raw_times, times[mask])
        assert np.array_equal(raw_times[indices], times[mask])
        assert np.array_equal(group.power.to_numpy(dtype=np.float32)[indices], truth[mask, :, 0])
    saved = json.loads((out / 'metrics.json').read_text())
    for key, value in metric(pred, truth).items():
        assert abs(value - saved[key]) < 1e-14, key
    lines = Path(run['log']).read_text().splitlines()
    epochs = []
    for line in lines:
        match = re.search(r'Epoch (\d+) train_loss=(\S+) validation=(\{.*\}) lr=', line)
        if match:
            validation = ast.literal_eval(match[3])
            assert np.isfinite(float(match[2])) and all(np.isfinite(x) for x in validation.values())
            epochs.append((int(match[1]), validation))
    assert [ast.literal_eval(line.split('Complete test metrics: ', 1)[1]) for line in lines if 'Complete test metrics: ' in line] == [saved]
    checkpoint = torch.load(ckpt / 'checkpoint.pth', map_location='cpu', weights_only=True)
    assert checkpoint['config'] == config
    assert all(torch.isfinite(x).all() for x in checkpoint['model_state_dict'].values())
    resume = run.get('throughput_resume_20261012')
    candidates = list(epochs)
    first_epoch = 1
    if resume:
        source = torch.load(resume['checkpoint'], map_location='cpu', weights_only=True)
        first_epoch = source['epoch'] + 1
        candidates.insert(0, (source['epoch'], source['validation_metrics']))
    assert [epoch for epoch, _ in epochs] == list(range(first_epoch, first_epoch + len(epochs)))
    best = min(candidates, key=lambda item: item[1]['mae'])
    assert checkpoint['epoch'] == best[0] and checkpoint['validation_metrics'] == best[1]
    last_epoch = epochs[-1][0]
    assert last_epoch == config['train_epochs'] or last_epoch - best[0] == config['patience']
    index_path = ROOT / 'reports' / run['protocol'] / 'results.csv'
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        index_fields, index_rows = reader.fieldnames, list(reader)
    matches = [row for row in index_rows if row['result_dir'] == str(out.resolve())]
    assert len(matches) <= 1
    if matches:
        assert matches[0]['baseline_config'] == 'fusion_input_variant'
        for key in ('mae', 'rmse', 'r2'):
            assert float(matches[0][key]) == saved[key]
    if run.get('readout_repair_20261012'):
        assert config['fusion_readout_init'] == 'positive_v1' and config['fusion_readout_lr_factor'] == 0.01
        assert not config['resume']
        assert checkpoint['scheduler_state_dict']['base_lrs'] == [
            config['learning_rate'], config['learning_rate'] * config['fusion_readout_lr_factor']]
    args = parse_args(run['argv'] + ['--gpu', str(gpu)])
    configure_resources(args)
    experiment = Exp_Long_Term_Forecast(args)
    load_model_checkpoint(experiment.model, ckpt / 'checkpoint.pth', experiment.device)
    experiment.model.eval()
    _, loader = experiment._get_data('test')
    with torch.no_grad():
        replay = experiment._forward_batch(next(iter(loader)))[0]['prediction'].cpu().numpy()
    difference = float(np.max(np.abs(replay - pred[:len(replay)])))
    assert difference < 1e-5, difference
    experiment._record_result(out, saved, specific_checks={'source': __file__, 'replay_max_abs': difference})
    baseline = json.loads((reference / 'metrics.json').read_text())
    return dict(status='passed_full_artifact_acceptance', checked_at=now(), metrics=saved,
                completed_epochs=last_epoch, newly_completed_epochs=len(epochs),
                resumed_from_epoch=first_epoch - 1, best_epoch=best[0], selection='minimum validation MAE',
                checks=['all labels match raw CSV/reference', 'protocol/sites/times match',
                        'train-only scalers equal reference and checkpoint', 'finite weights/predictions; no zero station',
                        'minimum validation MAE checkpoint', 'full metrics/log/unique index agree', 'saved checkpoint first batch replay'],
                checkpoint_replay_max_absolute_difference=difference,
                comparison_to_original_fusionsf={key: dict(absolute_change=saved[key]-baseline[key],
                    relative_change_percent=(saved[key]/baseline[key]-1)*100) for key in ('mae', 'rmse', 'r2')},
                readout_repair=run.get('readout_repair_20261012'),
                early_validation_observation=(json.loads((REPORT / f'{run["variant"]}_{run["protocol"]}_validation_observation_20261012.json').read_text())
                    if not run.get('readout_repair_20261012') and (REPORT / f'{run["variant"]}_{run["protocol"]}_validation_observation_20261012.json').exists() else None),
                comparison_limit='Input/branch and measured batch configuration differ; single seed; does not isolate input-only effects')


def sync_table(record):
    with (ROOT / 'reports/experiment_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        fields = csv.DictReader(handle).fieldnames
    rows = []
    for variant, label in VARIANTS.items():
        row = dict.fromkeys(fields, '')
        row.update(models=f'FusionSF ({variant})', input=label + '+坐标/时间编码')
        if any(run['variant'] == variant and run.get('readout_repair_20261012') for run in record['runs']):
            row['models'] = f'FusionSF ({variant}; readout_repair_v1)'
        for run in record['runs']:
            if run['variant'] == variant and run['state'] == 'completed_artifact_acceptance_passed':
                prefix = 'MMSP in_domain' if run['protocol'] == 'in_domain' else 'MMSP zeroshot'
                for label_metric, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')]:
                    row[f'{prefix} {label_metric}'] = f'{run["acceptance"]["metrics"][key]:.6f}'
        rows.append(row)
    with (ROOT / 'reports/fusionsf_input_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
    text = doc.read_text()
    section = '\n## FusionSF三种输入实验\n\n'
    if record.get('latest_authorization_20261012'):
        section += f'2026-10-12最新授权：A02两协议完成并验收后在GPU4继续以下六组，取消99%停止条件，允许通过batch调整优化真实吞吐；每种先in_domain再zeroshot_v1，seed42、24→24。当前队列状态：`{record["state"]}`。固定GPU4、CPU0,7两线程、num_workers0。\n\n'
        if record.get('repair_authorization_20261012'):
            section += '用户随后明确批准“修复重跑”：首组旧尝试epoch23结束，25450×24正式预测全零，验收失败；旧Metric/预测/权重/配置/日志及诊断保留在原目录及runs.json的failed_attempts，未进入正式结果表。功率+未来NWP两协议采用readout_repair_v1独立目录，从seed42/epoch1重新训练，不复用退化权重或优化器。\n\n'
            section += '修复仅改变读出初始化和读出学习率：原Linear权重保留随机方向，行L2范数缩至0.05/sqrt(128)，bias0.1，使初始输出在[0.05,0.15]；主干AdamW lr0.0016，读出组lr0.000016（主干1/100）。结构/ReLU/VQ/L1+VQ、FP32、max100/patience20、warmup5-cosine100和协议保持。首种输入两协议已正式验收；用户随后批准停止含卫星退化组、保留产物，将同一修复扩展到剩余两种输入的两个协议，经真实批次功率监督梯度检查后从epoch1重跑。短检查不是正式训练或性能复现；修复配置不能称官方默认配置或单纯输入消融。\n\n'
            section += '必要检查：`reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_preflight_20261012.json`；仅正初始化而不降低读出学习率的失败诊断日志保留。\n\n'
            if record.get('remaining_readout_repair_authorization_20261012'):
                section += '剩余两种输入检查：`reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_remaining_preflight_20261012.json`；停止前含卫星最佳epoch2权重的六样本ReLU前负值及零监督梯度证据保留，未进行该旧尝试的全量正式测试。\n\n'
        else:
            section += 'FP32、max100/patience20、lr0.0016及原AdamW/warmup5-cosine100保持；三种输入各比较batch16/64/256真实步骤，固定最快可用配置，不按测试成绩选择。增大batch改变优化轨迹及每轮更新次数，不能将新旧成绩或不同输入的差异完全归因于输入。首组从旧第2轮最佳权重继续epoch3，原RNG及后续未保存状态未恢复；原权重/配置/日志和原99%停止证据保留。\n\n'
        for variant, settings in record.get('throughput_settings_20261012', {}).items():
            section += f'- `{variant}`：batch{settings["batch_size"]}/eval{settings["eval_batch_size"]}。\n'
        section += '\n真实吞吐证据：`reports/FusionSF_inputs_two_protocols_seed42_20261011/batch_throughput_20261012.json`；训练期GPU采样含验证阶段，每30秒一次，不将峰值称为持续满载。\n\n'
        observation = REPORT / 'power_future_nwp_in_domain_validation_observation_20261012.json'
        if observation.exists() and not record.get('repair_authorization_20261012'):
            section += '首组早期验证连续不变且匹配已保存的全零参考，记为疑似全零，未证明永久退化；不自动停止、改配置或重跑。正式测试若全零则产物验收失败并阻止下一组。证据：`reports/FusionSF_inputs_two_protocols_seed42_20261011/power_future_nwp_in_domain_validation_observation_20261012.json`。\n\n'
    else:
        section += f'用户授权TimeVLM两协议退出并验收后串行执行以下三种输入，每种先in_domain再zeroshot_v1，共六组；seed42、24→24。当前队列状态：`{record["state"]}`。单GPU、CPU0,7两线程、num_workers=0。\n\n'
    section += '历史NWP与对应时刻功率拼成16通道，未来NWP保留原17通道引导（15天气+经纬度）；无卫星版本去除卫星分支及context VQ。保留当前script配置、ReLU输出、TS VQ、L1+VQ、AdamW lr0.0016、warmup5/cosine100、max100/patience20、验证MAE选权重。新输入属于研究变体，独立填表，不替换原FusionSF主表或复用历史0.040592。训练/测试/验收失败立即停止。\n\n'
    section += '| ' + ' | '.join(fields) + ' |\n| ' + ' | '.join(['---'] * len(fields)) + ' |\n'
    for row in rows:
        section += '| ' + ' | '.join(row[key] or '—' for key in fields) + ' |\n'
    section += '\n[下载三种输入CSV](../reports/fusionsf_input_metrics.csv)。脚本：`scripts/run_fusionsf_inputs.py`；唯一运行/验收记录：`reports/FusionSF_inputs_two_protocols_seed42_20261011/runs.json`。权重和预测按各run的checkpoint_dir/output_dir保存，修复尝试独立命名；正式索引及独立表仅填已验收指标。全部完成后停止，无后续模型或新种子。\n'
    if '\n## FusionSF三种输入实验\n' in text:
        text = re.sub(r'\n## FusionSF三种输入实验\n.*?(?=\n## |\Z)', lambda _: section, text, flags=re.DOTALL)
    else:
        text += section
    doc.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-run', type=int, choices=range(6))
    parser.add_argument('--accept-run', type=int, choices=range(6))
    parser.add_argument('--gpu', type=int, help='Fix the queue to this GPU; check/accept mode uses it too')
    args = parser.parse_args()
    os.sched_setaffinity(0, CORES)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    record = prepare()
    if args.prepare_only:
        return 0
    if args.check_run is not None or args.accept_run is not None:
        index = args.check_run if args.check_run is not None else args.accept_run
        run = record['runs'][index]
        if args.check_run is not None:
            run['preflight'] = preflight(run, args.gpu)
        else:
            run['acceptance'] = accept(run, args.gpu)
            run['state'] = 'completed_artifact_acceptance_passed'
            run['completed_at'] = now()
        save(record)
        sync_table(record)
        return 0
    lock = (REPORT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prerequisite_path = Path(record.get('prerequisite', str(TIMEVLM)))
    record.update(queue_pid=os.getpid(), state='waiting_prerequisite', queue_started_at=now(), fixed_gpu=args.gpu)
    save(record)
    sync_table(record)
    try:
        prerequisite = json.loads(prerequisite_path.read_text())
        while any(active(run) for run in prerequisite['runs']):
            time.sleep(30)
        prerequisite = json.loads(prerequisite_path.read_text())
        if prerequisite_path != TIMEVLM:
            assert prerequisite['acceptance']['status'] == 'passed_full_training_test_artifact_acceptance'
            assert all(run['state'] == 'completed_artifact_acceptance_passed' for run in prerequisite['runs'])
        elif prerequisite.get('acceptance', {}).get('status') != 'passed_full_artifact_acceptance':
            for run in prerequisite['runs']:
                if not (Path(run['output_dir']) / 'metrics.json').exists():
                    raise RuntimeError(f'TimeVLM exited without formal metrics: {run["protocol"]}')
            gpu = idle_gpu()
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/accept_timevlm.py'), '--gpu', str(gpu)], cwd=ROOT, check=True)
        prerequisite = json.loads(prerequisite_path.read_text())
        assert prerequisite['acceptance']['status'] in ('passed_full_artifact_acceptance',
                                                      'passed_full_training_test_artifact_acceptance')
        record.update(state='running', prerequisite_acceptance_confirmed_at=now())
        save(record)
        for index in range(6):
            record = json.loads(STATUS.read_text())
            run = record['runs'][index]
            if run['state'] == 'completed_artifact_acceptance_passed':
                continue
            if run['state'] == 'training':
                # A coordinator can be replaced to pin its GPU without interrupting its training child.
                if args.gpu is not None and run['gpu'] != args.gpu:
                    raise RuntimeError('Existing training is on a different GPU')
                while active(run):
                    time.sleep(30)
                if not (Path(run['output_dir']) / 'metrics.json').exists():
                    raise RuntimeError(f'Adopted training exited without formal metrics: {run["model_id"]}')
                run.update(state='complete_pending_acceptance', coordinator_adopted_existing_training=True)
                save(record)
            if run['state'] == 'complete_pending_acceptance':
                gpu = idle_gpu(run['gpu'])
                subprocess.run([sys.executable, '-u', __file__, '--accept-run', str(index), '--gpu', str(gpu)], cwd=ROOT, check=True)
                continue
            resuming = run['state'] == 'resume_pending'
            if run.get('readout_repair_20261012'):
                proof = json.loads(Path(run['readout_repair_20261012']['preflight']).read_text())
                assert proof['status'] == 'passed_real_batch_positive_readout_and_supervised_gradients'
                assert any(row['protocol'] == run['protocol'] and row.get('variant', 'power_future_nwp') == run['variant']
                           and row['batch_size'] == run['throughput_config_20261012']['batch_size'] for row in proof['runs'])
                assert '--resume' not in run['argv']
            if not resuming and (run['state'] != 'pending' or Path(run['output_dir']).exists() or Path(run['checkpoint_dir']).exists()):
                raise RuntimeError(f'Refusing to repeat an existing/unfinished run: {run["model_id"]} {run["protocol"]}')
            gpu = idle_gpu(args.gpu)
            if run.get('preflight', {}).get('status') != 'passed_real_batch_forward_backward':
                subprocess.run([sys.executable, '-u', __file__, '--check-run', str(index), '--gpu', str(gpu)], cwd=ROOT, check=True)
            idle_gpu(gpu)
            record = json.loads(STATUS.read_text())
            run = record['runs'][index]
            run.update(gpu=gpu, state='training', started_at=now())
            with Path(run['log']).open('a' if resuming else 'w') as log:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run.py'),
                                            *run['argv'], '--gpu', str(gpu)], cwd=ROOT,
                                           stdout=log, stderr=subprocess.STDOUT)
                run['pid'] = process.pid
                save(record)
                suffix = '_readout_repair_v1' if run.get('readout_repair_20261012') else ''
                run['gpu_utilization_log'] = str(REPORT / f'{run["variant"]}_{run["protocol"]}{suffix}_gpu_utilization_20261012.csv')
                save(record)
                returncode = watch_training(process, gpu, Path(run['gpu_utilization_log']))
            run['returncode'] = returncode
            run['state'] = 'complete_pending_acceptance' if returncode == 0 else 'failed'
            save(record)
            if returncode:
                raise RuntimeError(f'Training/test failed: {run["model_id"]} {run["protocol"]}; exit={returncode}')
            idle_gpu(gpu)
            subprocess.run([sys.executable, '-u', __file__, '--accept-run', str(index), '--gpu', str(gpu)], cwd=ROOT, check=True)
        record = json.loads(STATUS.read_text())
        record.update(state='completed_artifact_acceptance_passed', completed_at=now())
        save(record)
        sync_table(record)
    except Exception as error:
        record = json.loads(STATUS.read_text())
        record.update(state='failed_stopped', error=str(error), stopped_at=now())
        save(record)
        sync_table(record)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
