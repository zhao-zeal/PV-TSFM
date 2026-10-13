"""Run only authorized A02, in_domain then zeroshot_v1; A01 is reused without training."""

import argparse
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
REPORT = ROOT / 'reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012'
RECORD = REPORT / 'runs.json'
MODEL_ID = 'Chronos2WeatherAdapter_A02_seed42_sl24_pl24_history_future'


def now():
    return datetime.now().astimezone().isoformat()


def save(record):
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def sync_table(record):
    table = ROOT / 'reports/chronos2_adapter_metrics.csv'
    with table.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    base_label = 'A01 (Chronos-2 + B131 + Linear; E11)'
    new_label = 'A02 (A01 + Weather Adapter)'
    for row in rows:
        if row['models'] == 'Chronos-2 + Adapter (b131_linear_v1)':
            row['models'] = base_label
    new = next((row for row in rows if row['models'] == new_label), None)
    if new is None:
        new = {field: '' for field in fields}
        new.update(models=new_label, input='历史功率+历史/未来NWP')
        rows.append(new)
    for run in record['runs']:
        if run['state'] == 'completed_artifact_acceptance_passed':
            prefix = 'MMSP in_domain' if run['protocol'] == 'in_domain' else 'MMSP zeroshot'
            for label, key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('R^2', 'r2')]:
                new[f'{prefix} {label}'] = f'{run["metrics"][key]:.6f}'
    with table.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    doc = ROOT / 'docs/EXPERIMENT_STATUS.md'
    contents = doc.read_text().replace('| Chronos-2 + Adapter (b131_linear_v1) |', f'| {base_label} |')
    line = '| ' + ' | '.join(new[field] or '—' for field in fields) + ' |'
    if re.search(r'^\| A02 \(A01 \+ Weather Adapter\) \|.*$', contents, flags=re.MULTILINE):
        contents = re.sub(r'^\| A02 \(A01 \+ Weather Adapter\) \|.*$', line, contents, flags=re.MULTILINE)
    else:
        contents = re.sub(r'(^\| Chronos-2 \+ Adapter \(b131_mlp63_v1\) \|.*$)',
                          lambda match: match[1] + '\n' + line, contents, count=1, flags=re.MULTILINE)
    heading = '## A01/A02 Weather Adapter（2026-10-12）'
    block = (f'{heading}\n\n'
             '用户批准命名A01/A02并编写代码、开展实验。A01映射当前已验收的b131_linear_v1/E11，复用本项目同协议最佳权重和正式结果，不重新训练；E12原编号和记录保留。A02冻结Chronos-2、B131及Linear读出，仅训练新Weather Adapter，未指定为Ours。\n\n'
             'A02将2×768适配后表征按16/8小时对齐，与逐小时未来15维NWP及缺失标记各投影32维，拼接→ReLU→MLP(64,32,1)，功率残差加到反归一化后的A01预测；末层零初始化，共29249个新参数。借鉴ChronosX OIB的融合方式，输出为功率残差而非token logits。骨干原有历史/未来NWP输入和归一化不改。\n\n'
             '沿用Adapter读出预算：in_domain→zeroshot_v1串行，seed42、24→24、FP32、batch512/eval1024、AdamW lr0.000016/wd0.05/betas0.9,0.999、L1、max100/patience10、无调度，按验证MAE选权重（包括epoch0零残差）。单空闲GPU、CPU9,10两线程、num_workers0；失败停止，不自动重跑、调参或扩展消融。\n\n'
             '既有冻结特征保留，未来NWP从Dataset_MMSP读取，全部标签/站点/时间索引及训练scaler对齐后缓存。草稿CPU原生编码存在近零方差数值差异，保留诊断；GPU原生/缓存预测一致性、冻结权重、零初始化及梯度检查通过才运行，未改Chronos-2数学结构。\n\n'
             f'当前状态：`{record["state"]}`；各协议：' + '；'.join(f'{run["protocol"]}={run["state"]}' for run in record['runs']) + '。\n\n'
             '唯一入口`scripts/run_chronos2_weather_adapter.py`；唯一运行与验收记录`reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012/runs.json`，GPU检查`preflight_gpu.json`，原CPU诊断保留。正式验收后更新既有Adapter独立表，不覆盖官方Chronos-2主表。A02对A01的比较不能单独证明天气帮助利用表征；仅天气/仅表征/联合的贡献对照未启动，需另行授权。\n')
    if heading in contents:
        start = contents.index(heading)
        end = contents.find('\n## ', start + len(heading))
        contents = contents[:start] + block + (contents[end:] if end >= 0 else '')
    else:
        contents = contents.rstrip() + '\n\n' + block
    doc.write_text(contents)


def wait_idle(gpu):
    sys.path.insert(0, str(ROOT))
    from utils.resources import require_idle_gpus
    for _ in range(12):
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index',
                                           '--format=csv,noheader,nounits'], text=True)
        candidates = [gpu, *[int(index) for index in devices.splitlines() if int(index) != gpu]]
        for candidate in candidates:
            try:
                require_idle_gpus(str(candidate))
                return candidate
            except ValueError:
                continue
        time.sleep(10)
    raise RuntimeError('No idle GPU; stop without starting another experiment')


def execute(command, log, utilization=None, gpu=None, on_start=None):
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if on_start is not None:
            on_start(process.pid)
        if utilization is None:
            return process.wait(), process.pid
        with utilization.open('w', newline='') as samples:
            writer = csv.writer(samples)
            writer.writerow(['timestamp', 'gpu', 'memory_mib', 'utilization_percent'])
            while process.poll() is None:
                values = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                    '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                writer.writerow([now(), *[value.strip() for value in values.strip().split(',')]])
                samples.flush()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        return process.returncode, process.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    options = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        draft = continuation = None
        if RECORD.exists():
            draft = json.loads(RECORD.read_text())
            if (draft.get('state') == 'failed'
                    and draft.get('failure') in ('in_domain artifact acceptance exited 1',
                        'GPU0 is still occupied; stop without starting another experiment')
                    and draft['runs'][0]['state'] == 'completed_artifact_acceptance_passed'
                    and draft['runs'][0]['returncode'] == 0
                    and draft['runs'][1]['state'] == 'pending'):
                continuation, draft = draft, None
            elif (draft.get('status') != 'implemented_cached_cpu_checks_passed_waiting_gpu_preflight'
                  or not all(run.get('state') == 'not_started' for run in draft.get('runs', []))):
                raise FileExistsError('A02 already has a run record; no implicit duplicate or restart')
        for protocol in ('in_domain', 'zeroshot_v1'):
            if continuation and any(run['protocol'] == protocol and
                                    run['state'] == 'completed_artifact_acceptance_passed'
                                    for run in continuation['runs']):
                continue
            if ((ROOT / 'checkpoints' / protocol / MODEL_ID / 'checkpoint.pth').exists()
                    or (ROOT / 'outputs' / protocol / MODEL_ID / 'metrics.json').exists()):
                raise FileExistsError('A02 already has training/test artifacts; do not repeat it')
        os.sched_setaffinity(0, [9, 10])
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '2'
        preflight = json.loads((REPORT / 'preflight_gpu.json').read_text())
        assert preflight['status'] == 'passed' and preflight['full_experiment_ready']
        record = dict(experiment_label='A02', base_experiment_label='A01', historical_base_name='E11',
                      state='running', coordinator_pid=os.getpid(), started_at=now(), gpu=options.gpu,
                      cpu_cores=[9, 10], threads=2, num_workers=0, preflight=preflight,
                      authorization='User: name A01/A02; implement then conduct experiments',
                      training_budget='Inherited Adapter readout max100/patience10, FP32, batch512/eval1024, lr0.000016',
                      baseline_retraining=False, additional_ablations_authorized=False, runs=[])
        if draft is not None:
            record['implementation_checks_before_gpu'] = {
                key: draft[key] for key in ('experiment_input_integration', 'validation', 'cleanup', 'preflight')
                if key in draft}
            record['resolved_draft_blockers'] = dict(previous=draft.get('blockers', []),
                resolution='Host GPU0 available; two-protocol GPU preflight passed without native model changes')
        for protocol in ('in_domain', 'zeroshot_v1'):
            argv = ['--model', 'Chronos2WeatherAdapter', '--protocol', protocol, '--is_training', '1',
                    '--seed', '42', '--seq_len', '24', '--pred_len', '24', '--model_id', MODEL_ID,
                    '--use_gpu', '1', '--gpu', str(options.gpu), '--num_workers', '0']
            record['runs'].append(dict(protocol=protocol, state='pending', argv=argv,
                output_dir=str(ROOT / 'outputs' / protocol / MODEL_ID),
                checkpoint_dir=str(ROOT / 'checkpoints' / protocol / MODEL_ID),
                log=str(REPORT / f'{protocol}_training.log')))
        if continuation:
            record = continuation
            record.setdefault('coordinator_corrections', []).append(dict(
                previous_coordinator_pid=record['coordinator_pid'], failed_at=record['failed_at'],
                failure=record['failure'], correction=(
                    'Acceptance registers the formal index only after checks pass' if 'acceptance' in record['failure']
                    else 'GPU0 occupied before next protocol; select another idle GPU'),
                training_repeated=False, full_inference_repeated=False, continued_at=now()))
            record.update(state='running', coordinator_pid=os.getpid(), gpu=options.gpu)
            record.pop('failure')
            record.pop('failed_at')
        save(record)
        original_path = ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json'
        original = json.loads(original_path.read_text())
        for version in original['version_registry']:
            if version['adapter_version'] == 'b131_linear_v1':
                version['experiment_label'] = 'A01'
        original['reindexing_authorization_20261012'] = 'User chose A01 for E11, A02 for the new Weather Adapter; historical identities/artifacts unchanged'
        original_path.write_text(json.dumps(original, ensure_ascii=False, indent=2) + '\n')
        sync_table(record)
        try:
            for index in range(len(record['runs'])):
                run = record['runs'][index]
                if run['state'] == 'completed_artifact_acceptance_passed':
                    continue
                selected_gpu = wait_idle(options.gpu)
                run['argv'][run['argv'].index('--gpu') + 1] = str(selected_gpu)
                run.update(state='training', started_at=now(), gpu=selected_gpu)
                save(record)
                sync_table(record)
                print(f'{now()} start A02 {run["protocol"]} GPU{selected_gpu}', flush=True)
                def started(pid):
                    run['pid'] = pid
                    save(record)
                code, pid = execute([sys.executable, str(ROOT / 'run.py'), *run['argv']],
                    Path(run['log']), REPORT / f'{run["protocol"]}_gpu_utilization.csv', selected_gpu, started)
                run.update(pid=pid, returncode=code, finished_at=now())
                if code:
                    raise RuntimeError(f'{run["protocol"]} train/test exited {code}')
                run['state'] = 'complete_pending_acceptance'
                save(record)
                acceptance_gpu = wait_idle(selected_gpu)
                code, _ = execute([sys.executable, str(ROOT / 'scripts/accept_chronos2_weather_adapter.py'),
                    '--report', str(RECORD), '--protocol', run['protocol'], '--gpu', str(acceptance_gpu)],
                    REPORT / f'{run["protocol"]}_acceptance.log')
                if code:
                    raise RuntimeError(f'{run["protocol"]} artifact acceptance exited {code}')
                record = json.loads(RECORD.read_text())
                sync_table(record)
                print(f'{now()} accepted A02 {record["runs"][index]["protocol"]}', flush=True)
            record.update(state='completed_artifact_acceptance_passed', completed_at=now(),
                          acceptance=dict(status='passed_full_training_test_artifact_acceptance',
                              runs=[run['acceptance'] for run in record['runs']], full_reinference=False))
        except Exception as error:
            record = json.loads(RECORD.read_text())
            record.update(state='failed', failure=str(error), failed_at=now())
            for run in record['runs']:
                if run['state'] in ('training', 'complete_pending_acceptance'):
                    run['state'] = 'failed'
            save(record)
            sync_table(record)
            raise
        save(record)
        sync_table(record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
