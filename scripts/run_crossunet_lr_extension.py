"""Two additional learning rates; validation selection, then winner test acceptance."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_baseline_limited_tuning as previous
REPORT = ROOT / 'reports/CrossUnet_lr_extension_in_domain_seed42_20261016'
INDEX = REPORT / 'runs.json'
save = previous.save


def prepare():
    from run import parse_args
    old = json.loads((previous.REPORT / 'runs.json').read_text())
    ref = next(t for t in old['trials'] if t['model'] == 'CrossUnet' and t['candidate'] == 'lr2e4')
    trials = []
    for name, rate in [('lr3e4', .0003), ('lr4e4', .0004)]:
        argv = list(ref['argv'])
        argv[argv.index('--learning_rate') + 1] = str(rate)
        argv[argv.index('--model_id') + 1] = f'CrossUnet_lr_extension_{name}_seed42_sl24_pl24_20261016'
        config = vars(parse_args(argv))
        differences = {k: [v, config.get(k)] for k, v in ref['parsed_config'].items() if v != config.get(k)}
        assert set(differences) <= {'learning_rate', 'model_id'}, differences
        trials.append(dict(model='CrossUnet', candidate=name, argv=argv, state='pending', config_differences=differences))
    REPORT.mkdir(exist_ok=False)
    save(INDEX, dict(state='prepared', authorization='User: 继续调参再测试验收',
        scope='CrossUnet only; in_domain seed42 24->24; two added learning rates .0003/.0004; unchanged batch128 FP32 type3 max100 patience10',
        selection='Lowest validation RMSE among prior three configurations and these two; freeze winner before test; reuse accepted prior winner if unchanged',
        limitation='Single seed; benchmark test previously observed; this is continued development, not a fresh independent holdout',
        resources='One idle GPU; CPU9,10 two threads; workers0; serial; failure stops without automatic retry',
        reference=old['selected_by_validation']['CrossUnet'],
        reference_test=old['formal_tests']['runs']['CrossUnet'], trials=trials))


def evaluate(gpu):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from run import configure_resources
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.tools import load_model_checkpoint
    from utils.acceptance import accept_and_record
    status = json.loads(INDEX.read_text())
    winner = status['selected_by_validation']
    checkpoint = Path(winner['checkpoint'])
    config = json.loads((checkpoint.parent / 'config.json').read_text())
    config.update(is_training=0, checkpoint=str(checkpoint), gpu=gpu, model_id=config['model_id'] + '_formal_test')
    args = argparse.Namespace(**config)
    configure_resources(args)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    experiment = Exp_Long_Term_Forecast(args)
    with torch.no_grad():
        metrics = experiment.test(args.model_id, test=1)
        output = Path(args.output_dir) / args.protocol / args.model_id
        load_model_checkpoint(experiment.model, checkpoint, experiment.device)
        experiment.model.eval()
        data, _ = experiment._get_data('test')
        pred = np.load(output / 'pred.npy', mmap_mode='r')
        batch = args.eval_batch_size
        groups = [list(range(batch)), list(range((len(data)-1)//batch*batch, len(data)))]
        error = 0.
        for indices in groups:
            actual, _, _ = experiment._forward_batch(default_collate([data[i] for i in indices]))
            error = max(error, float(np.max(np.abs(actual.cpu().numpy() - pred[indices]))))
        assert error < 1e-5, error
    receipt = accept_and_record(output, specific_checks=dict(checkpoint_GPU_replay=dict(batches=groups, max_abs_error=error, tolerance=1e-5, gpu=gpu)))
    assert receipt['status'] == 'passed'
    save(REPORT / 'test_result.json', dict(state='accepted_formal_test', metrics=metrics,
        checkpoint=str(checkpoint), output_dir=str(output), acceptance=receipt))


def publish(status):
    result = status['test_result']; metrics = result['metrics']; baseline = status['reference_test']['metrics']
    status['test_changes_vs_lr2e4'] = {k: dict(absolute=metrics[k]-baseline[k], relative_percent=(metrics[k]/baseline[k]-1)*100) for k in ('mae', 'rmse', 'r2')}
    table = ROOT / 'reports/experiment_summary.csv'
    with table.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
    row = next(r for r in rows if r['models'] == 'Cross-Unet')
    for field, key in zip(fields[2:5], ('mae', 'rmse', 'r2')):
        row[field] = f'{metrics[key]:.6f}'
    with table.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    doc = ROOT / 'docs/EXPERIMENT_STATUS.md'; text = doc.read_text()
    old = next(line for line in text.splitlines() if line.startswith('| Cross-Unet |'))
    text = text.replace(old, '| ' + ' | '.join(row[k] for k in fields) + ' |', 1)
    name = status['selected_by_validation']['candidate']
    rates = {'lr2e4': '0.0002', 'lr3e4': '0.0003', 'lr4e4': '0.0004'}
    text = text.replace('Cross-Unet in_domain使用lr0.0002', f'Cross-Unet in_domain使用lr{rates[name]}')
    text += '\nCross-Unet本轮补充调参及测试验收完成。验证RMSE：' + '；'.join(f"{t['candidate']}={t['result']['metrics']['rmse']:.9f}" for t in status['trials'])
    text += f"；已有lr0.0002={status['reference']['metrics']['rmse']:.9f}。验证选定{name}，测试MAE/RMSE/R²={metrics['mae']:.9f}/{metrics['rmse']:.9f}/{metrics['r2']:.9f}。"
    delta = status['test_changes_vs_lr2e4']['rmse']
    text += f"相对上一轮测试RMSE变化{delta['absolute']:+.9f}（{delta['relative_percent']:+.4f}%）。"
    text += '主表按验证选优结果更新，不按测试反馈改选。若沿用原版则复用已验收测试，没有重复推理。完整日志、配置、权重、选模、预测与验收路径见reports/CrossUnet_lr_extension_in_domain_seed42_20261016/runs.json；无新增图表。单种子、测试已被观察及未来NWP可用性假设的限制保留。当前两组完成后停止，不扩展其他参数或协议。\n'
    doc.write_text(text)


def idle_gpu():
    from utils.resources import require_idle_gpus
    while True:
        for gpu in range(8):
            try:
                require_idle_gpus(str(gpu))
                return gpu
            except ValueError:
                pass
        time.sleep(60)


def coordinate():
    status = json.loads(INDEX.read_text()); assert status['state'] == 'prepared'
    try:
        for i, item in enumerate(status['trials']):
            status.update(state='waiting_for_idle_gpu', active_trial=i); save(INDEX, status)
            gpu = idle_gpu()
            status.update(state='training'); item.update(state='running', gpu=gpu); save(INDEX, status)
            with (REPORT / f'{i}_training.log').open('x') as log:
                subprocess.run([sys.executable, '-u', __file__, '--trial', str(i), '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT, check=True)
            item.update(state='accepted_validation_only', result=json.loads((REPORT / f'{i}_result.json').read_text()))
            save(INDEX, status)
        winner = min([status['reference']] + [t['result'] for t in status['trials']], key=lambda r: r['metrics']['rmse'])
        status.update(state='selected_by_validation', selected_by_validation=winner, active_trial=None)
        save(INDEX, status)  # Freeze selection before any new test access.
        if winner['candidate'] == 'lr2e4':
            status['test_result'] = status['reference_test']
            status['test_reused'] = True
        else:
            gpu = idle_gpu()
            with (REPORT / 'formal_test.log').open('x') as log:
                subprocess.run([sys.executable, '-u', __file__, '--test', '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT, check=True)
            status['test_result'] = json.loads((REPORT / 'test_result.json').read_text())
            status['test_reused'] = False
        publish(status)
        status['state'] = 'completed_formal_test_acceptance'
    except Exception as error:
        status.update(state='failed_stopped', error=repr(error))
        raise
    finally:
        save(INDEX, status)


if __name__ == '__main__':
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--trial', type=int)
    parser.add_argument('--test', action='store_true'); parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.trial is not None:
        previous.REPORT = REPORT
        previous.trial(args.trial, args.gpu)
    elif args.test: evaluate(args.gpu)
    else: coordinate()
