"""A03-a/b/c seed42 zero-shot: checks, serial training, test and acceptance only."""

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012'
RECORD = REPORT / 'runs.json'
SOURCE_FILES = ('layers/chronos2_history_weather_adapter.py', 'models/Chronos2HistoryWeatherAdapter.py',
                'exp/exp_chronos2_history_weather_adapter.py', 'scripts/run_chronos2_history_weather_adapter.py')


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def configure(gpu):
    os.sched_setaffinity(0, [9, 10])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from utils.resources import require_idle_gpus
    stable = 0
    for attempt in range(12):
        try:
            require_idle_gpus(str(gpu))
            stable += 1
            if stable >= 2:
                break
        except ValueError:
            stable = 0
        if attempt == 11:
            raise RuntimeError('GPU did not become stably idle; stop before model loading')
        time.sleep(1)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return np, torch


def make_args(variant, gpu):
    from run import parse_args
    args = parse_args(['--model', 'Chronos2WeatherAdapter', '--protocol', 'zeroshot_v1',
                      '--is_training', '1', '--seed', '42', '--use_gpu', '1', '--gpu', str(gpu),
                      '--num_workers', '0'])
    args.model = 'Chronos2HistoryWeatherAdapter'
    args.history_variant = variant
    args.adapter_version = f'b131_linear_history_weather_{variant}_v1'
    args.experiment_label = f'A03-{variant}'
    args.baseline_config = 'history_weather_adapter_variant'
    args.model_id = f'Chronos2HistoryWeatherAdapter_A03{variant}_seed42_sl24_pl24_history_future'
    args.a03_source_sha256 = source_identity()
    args.protocol_limit = 'A02 gridwise bidirectional NWP interpolation and assumed issue-time availability retained by explicit user decision'
    return args


def reseed(torch, np):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def protect_originals():
    review = json.loads((REPORT / 'review.json').read_text())
    for name, expected in review['files_reviewed'].items():
        # Framework acceptance/isolation was updated; retain numerical-model/data protection.
        if not name.startswith(('models/', 'layers/', 'data_provider/')):
            continue
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original implementation changed after review: {name}')
    for protocol in ('in_domain', 'zeroshot_v1'):
        row = review['a02'][protocol]
        for prefix in ('checkpoint', 'base_checkpoint'):
            if hashlib.sha256(Path(row[prefix]).read_bytes()).hexdigest() != row[f'{prefix}_sha256']:
                raise ValueError('A01/A02 checkpoint changed after review')


def preflight(gpu):
    np, torch = configure(gpu)
    from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter
    from layers.chronos2_history_weather_adapter import HistoricalRelationConditioner, weather_power_correlation
    from models.Chronos2WeatherAdapter import Model as A02Model
    from scripts.check_chronos2_weather_adapter import state_digest
    protect_originals()
    result = dict(status='checking', stage='real_GPU_correctness_checks', full_training_complete=False,
                  formal_test_complete=False, source_sha256=source_identity(), variants=[])
    # Known signed correlation and constant/missing/low-variance edge cases.
    p = torch.arange(7, dtype=torch.float32).view(1, 7, 1).to(f'cuda:{gpu}')
    w = torch.cat((p, -p, torch.ones_like(p), torch.full_like(p, torch.nan),
                   p * 1e-8), dim=-1)
    corr = weather_power_correlation(p, w)
    assert torch.allclose(corr[0, :5], torch.tensor([1., -1., 0., 0., 0.], device=p.device))
    assert torch.isfinite(corr).all()
    for v in 'abc':
        module = HistoricalRelationConditioner(v, 5, 32).to(p.device)
        assert torch.isfinite(module(p, w)).all()
        assert torch.isfinite(module(p * torch.nan, w * torch.nan)).all()
        # Nonstandard lengths/channel counts are handled by the relation module.
        assert module(p[:, :3], w[:, :3]).shape == (1, 32)
    result['synthetic_checks'] = dict(signed_pearson_reference=True, constants_missing_low_variance_finite=True,
                                    dynamic_lengths_channels=True, variance_epsilon=1e-8)
    reseed(torch, np)
    reference = A02Model(make_args('a', gpu)).float().to(p.device)
    initial_weather_sha = state_digest(reference.weather_adapter)
    del reference
    torch.cuda.empty_cache()
    for variant in 'abc':
        reseed(torch, np)
        exp = Exp_Chronos2_History_Weather_Adapter(make_args(variant, gpu))
        model = exp.model
        assert state_digest(model.weather_adapter) == initial_weather_sha
        assert all(not par.requires_grad for par in model.base.parameters())
        model.train()
        assert not any(m.training for m in model.base.modules())
        before = state_digest(model.base)
        # Build all caches with full window identity, scaler and historical-boundary checks.
        split_evidence = []
        for flag in ('train', 'val', 'test'):
            ds, loader = exp._get_data(flag)
            complete = exp.history_dir / flag / 'complete.json'
            split_evidence.append(dict(split=flag, windows=len(ds), manifest_sha256=hashlib.sha256(complete.read_bytes()).hexdigest()))
        ds, loader = exp._get_data('train')
        tensors = loader.dataset.tensors
        indices = [100, 1000, len(ds) // 2, len(ds) - 1]
        batch = [v[indices] for v in tensors]
        embeds, loc, scale, target, _, _, power, history, future = [v.to(exp.device) for v in batch]
        with torch.no_grad():
            a02 = A02Model.forward_features(model, embeds, loc, scale, future)
            prediction, _, _ = exp._forward_batch(batch)
            assert torch.equal(prediction, a02)
            native = model.base.encode_inputs(power[..., 0], history, future)
            direct = model.forward_features(*native, future, power, history)
            difference = float((direct - prediction).abs().max())
            assert difference <= 2e-6, difference
            # Decoder content (including future target values) is ignored.
            cov = dict(historical_nwp=history, future_nwp=future)
            ordinary = model(power, None, torch.zeros_like(target), None, cov)
            changed = model(power, None, target + 1000, None, cov)
            assert torch.equal(ordinary, changed)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=exp.args.learning_rate, weight_decay=0.05, betas=(0.9, 0.999))
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction, targets, _ = exp._forward_batch(batch)
            loss = torch.nn.functional.l1_loss(prediction, targets)
            assert prediction.shape == targets.shape == (len(indices), exp.args.pred_len, 1)
            assert torch.isfinite(prediction).all() and torch.isfinite(loss)
            loss.backward()
            gradients = {name: float(par.grad.abs().sum()) for name, par in model.named_parameters() if par.requires_grad
                         and par.grad is not None and torch.isfinite(par.grad).all()}
            assert len(gradients) == sum(par.requires_grad for par in model.parameters())
            assert all(par.grad is None for par in model.base.parameters())
            if step == 1:
                assert all(value > 0 for name, value in gradients.items() if name.startswith('history_relation.'))
                assert gradients['weather_adapter.cov_out.weight'] > 0
            steps.append(dict(step=step + 1, loss=float(loss.detach()), gradient_l1=gradients))
            optimizer.step()
        assert state_digest(model.base) == before
        with torch.no_grad():
            initial = model.history_relation(power, history)
            absent_power, absent_weather = power.clone(), history.clone()
            absent_power[:, 0] = torch.nan; absent_weather[:, 3, 0] = torch.nan
            changed_output = model.forward_features(embeds, loc, scale, future, absent_power, absent_weather)
            assert torch.isfinite(changed_output).all()
            # Marginal statistics intentionally ignore power/weather pairing.
            # Correlation and the joint learned encoder must respond to pairing.
            other_history = history + 1 if variant == 'a' else history.flip(1)
            changed_relation = model.history_relation(power, other_history)
            assert not torch.equal(initial, changed_relation)
            if variant == 'a':
                assert torch.allclose(initial, model.history_relation(power, history.flip(1)), atol=1e-6)
            future_changed = future.clone(); future_changed[:, 5, 0] += 5
            a = model.forward_features(embeds, loc, scale, future, power, history)
            b = model.forward_features(embeds, loc, scale, future_changed, power, history)
            delta = b - a
            assert delta[:, 5].abs().max() > 0
            delta[:, 5] = 0
            assert torch.count_nonzero(delta) == 0  # Hourly weather alignment.
            assert torch.equal(initial, model.history_relation(power, history))
        result['variants'].append(dict(variant=variant, base_provenance=model.base_provenance,
            trainable_parameters=sum(par.numel() for par in model.parameters() if par.requires_grad),
            relation_parameters=sum(par.numel() for par in model.history_relation.parameters()),
            frozen_parameters=sum(par.numel() for par in model.base.parameters()),
            initial_weather_weights_equal_seed42_A02=True, epoch0_output_exactly_initialized_A02=True,
            native_cache_max_abs_difference=difference, frozen_state_sha256=before,
            frozen_gradients_absent_weights_unchanged=True, finite_output_all_gradients=True,
            new_relation_receives_nonzero_gradients_step2=True,
            historical_relation_responds_to_input=True,
            pairing_check='marginal statistics permutation invariant' if variant=='a' else 'joint pairing sensitivity passed',
            no_future_power_or_decoder_access=True, no_historical_index_crosses_origin=True,
            hourly_future_weather_alignment=True, missing_inputs_finite=True,
            full_splits_window_cache_alignment=split_evidence, transient_steps=steps))
        del exp, model, optimizer, loader, tensors, batch
        torch.cuda.empty_cache()
    result['status'] = 'passed'
    save(REPORT / 'preflight_gpu.json', result)
    print(json.dumps(dict(status='passed', parameters=[(v['variant'],v['trainable_parameters']) for v in result['variants']]), ensure_ascii=False), flush=True)


def evaluate_details(pred, truth, future, thresholds):
    import numpy as np
    from utils.metrics import metric
    error = pred.astype(np.float64) - truth.astype(np.float64)
    scores = np.square(error).mean(axis=(1,2))
    worst = np.argsort(-scores, kind='stable')[:int(np.ceil(len(pred) * 0.1))]
    intensity = np.abs(np.diff(future.astype(np.float64), axis=1)).mean(axis=(1,2))
    groups = np.searchsorted(thresholds, intensity, side='right')
    return dict(overall=metric(pred, truth),
                horizons=[dict(hour=i + 1, mae=float(np.abs(error[:, i]).mean()),
                               rmse=float(np.sqrt(np.square(error[:, i]).mean()))) for i in range(pred.shape[1])],
                worst10percent=dict(definition='ceil(10% windows), ranked by each model window RMSE; ties use original order',
                                    windows=len(worst), metrics=metric(pred[worst], truth[worst])),
                weather_change=dict(definition='Mean absolute adjacent-hour differences of 15 train-standardized future NWP channels; training-only tertile boundaries',
                                    train_thresholds=list(thresholds),
                                    groups=[dict(label=name, metrics=metric(pred[groups == i],truth[groups == i]))
                                            for i,name in enumerate(('low','medium','high'))]),
                _worst_indices=worst)


def accept(variant, gpu):
    np, torch = configure(gpu)
    from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from scripts.check_chronos2_weather_adapter import state_digest
    from utils.metrics import metric
    protect_originals()
    from types import SimpleNamespace
    saved_directory = ROOT / 'outputs/zeroshot_v1' / f'Chronos2HistoryWeatherAdapter_A03{variant}_seed42_sl24_pl24_history_future'
    saved_config = json.loads((saved_directory / 'config.json').read_text())
    for name, expected in saved_config['a03_source_sha256'].items():
        if name.startswith(('models/', 'layers/')) and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Numerical implementation changed: {name}')
    args = SimpleNamespace(**saved_config)
    args.gpu = gpu  # Replays may use a different idle device; historical config stays intact.
    exp = Exp_Chronos2_History_Weather_Adapter(args)
    out, ck = Path(args.output_dir)/args.protocol/args.model_id, Path(args.checkpoints)/args.protocol/args.model_id
    from utils.acceptance import accept_and_record
    accept_and_record(out)
    config = json.loads((out/'config.json').read_text())
    assert config == saved_config == json.loads((ck/'config.json').read_text())
    selected = exp.load_selected(ck/'checkpoint.pth')
    summary = json.loads((ck/'training_summary.json').read_text())
    with (ck/'training_history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    assert [int(row['epoch']) for row in history] == list(range(len(history)))
    assert all(np.isfinite(float(row[k])) for row in history for k in ('mae','rmse','mse'))
    assert all(np.isfinite(float(row['train_mae'])) for row in history[1:])
    best = min(history, key=lambda row: float(row['mae']))
    assert selected['epoch'] == int(best['epoch']) == summary['best_epoch']
    assert selected['validation_metrics']['mae'] == float(best['mae'])
    assert summary['completed_epochs'] == len(history)-1
    assert len(history)-1 == args.train_epochs or len(history)-1-selected['epoch'] == args.patience
    arrays = {name:np.load(out/f'{name}.npy') for name in ('pred','true','site_ids','forecast_timestamps')}
    pred, truth = arrays['pred'], arrays['true']
    assert pred.shape == truth.shape == (25450,args.pred_len,1)
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    a02_out = ROOT/'outputs/zeroshot_v1/Chronos2WeatherAdapter_A02_seed42_sl24_pl24_history_future'
    for name in ('true','site_ids','forecast_timestamps'):
        assert np.array_equal(arrays[name],np.load(a02_out/f'{name}.npy')), name
    saved = json.loads((out/'metrics.json').read_text())
    for key,value in metric(pred,truth).items():
        assert value == saved[key], key
    protocol = json.loads((out/'data_protocol.json').read_text())
    assert protocol == json.loads((a02_out/'data_protocol.json').read_text())
    for directory in (out,ck):
        with np.load(directory/'scalers.npz') as current,np.load(a02_out/'scalers.npz') as old:
            assert set(current.files)==set(old.files) and all(np.array_equal(current[k],old[k]) for k in old.files)
    training_protocol = json.loads((ck/'data_protocol.json').read_text())
    assert training_protocol['sites']==training_protocol['train_sites']==list(range(10,20))
    assert protocol['test_sites']==list(range(10)) and protocol['windows']==dict(train=76570,validation=25450,test=25450)
    for key in ('version','train_sites','validation_sites','test_sites','windows','target_time_ranges','scaler_fit','data_interface','modalities'):
        assert training_protocol[key]==protocol[key], key
    for site in protocol['test_sites']:
        values=pred[arrays['site_ids']==site]
        assert len(values)==2545 and np.any(values!=0)
    for state in ('weather_state_dict','relation_state_dict'):
        assert all(torch.isfinite(v).all() for v in selected[state].values())
    exp.model.eval()
    before=state_digest(exp.model.base)
    check = json.loads((REPORT/'preflight_gpu.json').read_text())
    checked=next(row for row in check['variants'] if row['variant']==variant)
    assert before==checked['frozen_state_sha256']
    ds, loader=exp._get_data('test')
    indices=[0,1,2545,25449]
    batch=[value[indices] for value in loader.dataset.tensors]
    with torch.no_grad():
        replay,_,_=exp._forward_batch(batch)
    difference=float(np.abs(replay.cpu().numpy()-pred[indices]).max())
    assert difference<=2e-6, difference
    assert all(not par.requires_grad and par.grad is None for par in exp.model.base.parameters())
    assert state_digest(exp.model.base)==before
    record=json.loads(RECORD.read_text())
    run=next(row for row in record['runs'] if row['variant']==variant)
    log=Path(run['log']).read_text()
    assert len(re.findall('Complete test metrics:',log))==1
    assert [int(x) for x in re.findall(r'A03-'+variant+r' epoch (\d+)',log)]==list(range(1,len(history)))
    future=np.load(exp.history_dir/'test/future_nwp.npy',mmap_mode='r')
    train_future=np.load(exp.history_dir/'train/future_nwp.npy',mmap_mode='r')
    train_intensity=np.abs(np.diff(train_future.astype(np.float64),axis=1)).mean(axis=(1,2))
    thresholds=np.quantile(train_intensity,[1/3,2/3])
    baseline=np.load(a02_out/'pred.npy')
    reference=evaluate_details(baseline,truth,future,thresholds)
    details=evaluate_details(pred,truth,future,thresholds)
    common=reference.pop('_worst_indices')
    details.pop('_worst_indices')
    reference['A02_worst10_common_subset']=metric(baseline[common],truth[common])
    details['A02_worst10_common_subset']=metric(pred[common],truth[common])
    details['comparison_to_A02']={k:dict(absolute_change=details['overall'][k]-reference['overall'][k],
        improvement_percent=(reference['overall'][k]-details['overall'][k])/reference['overall'][k]*100)
        for k in ('mae','rmse')}
    details['comparison_to_A02']['r2']=dict(absolute_change=saved['r2']-reference['overall']['r2'])
    details['limits']=['seed42 only; no mechanism or stability proof',
        'A03 has about 9% more trainable parameters than A02; capacity-matched control not run',
        'overlapping windows; worst-window sets differ by model, also report common A02 subset',
        'weather intensity is forecast variability, not observed weather uncertainty',args.protocol_limit]
    save(out/'analysis.json',details)
    if not (REPORT/'A02_reference_analysis.json').exists():
        save(REPORT/'A02_reference_analysis.json',reference)
    Exp_Long_Term_Forecast._record_result(exp,out,saved, specific_checks={'source': __file__})
    table=ROOT/'reports/zeroshot_v1/results.csv'
    with table.open() as handle:
        rows=[r for r in csv.DictReader(handle) if r['result_dir']==str(out.resolve())]
    assert len(rows)==1 and all(float(rows[0][k])==saved[k] for k in ('mae','rmse','r2'))
    item=dict(status='passed_full_training_test_artifact_acceptance',variant=variant,metrics=saved,
        completed_epochs=summary['completed_epochs'],best_epoch=selected['epoch'],
        trainable_parameters=summary['trainable_parameters'],relation_parameters=summary['relation_parameters'],
        checkpoint_sha256=hashlib.sha256((ck/'checkpoint.pth').read_bytes()).hexdigest(),
        frozen_state_sha256=before,selected_checkpoint_replay_max_abs_difference=difference,
        base_provenance=exp.model.base_provenance, analysis_path=str(out/'analysis.json'),
        comparison_to_A02=details['comparison_to_A02'],accepted_at=now())
    save(REPORT/f'acceptance_{variant}.json',item)
    run.update(state='completed_artifact_acceptance_passed',metrics=saved,acceptance=item)
    save(RECORD,record)
    print(json.dumps(item,ensure_ascii=False),flush=True)


def sync_table(record):
    table=ROOT/'reports/chronos2_adapter_metrics.csv'
    with table.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames; rows=list(reader)
    doc=ROOT/'docs/EXPERIMENT_STATUS.md'
    contents=doc.read_text()
    inputs=dict(a='历史功率+历史/未来NWP（显式历史统计）',b='历史功率+历史/未来NWP（显式历史相关性）',c='历史功率+历史/未来NWP（学习型历史关系）')
    for run in record['runs']:
        if run['state']!='completed_artifact_acceptance_passed':
            continue
        label=f'A03-{run["variant"]} (History-Aware Weather Adapter)'
        row=next((r for r in rows if r['models']==label),None)
        if row is None:
            row={key:'' for key in fields};row.update(models=label,input=inputs[run['variant']]);rows.append(row)
        for key,metric in (('MAE','mae'),('RMSE','rmse'),('R^2','r2')):
            row[f'MMSP zeroshot {key}']=f'{run["metrics"][metric]:.6f}'
        line='| '+' | '.join(row[key] or '—' for key in fields)+' |'
        pattern=r'^\| '+re.escape(label)+r' \|.*$'
        if re.search(pattern,contents,flags=re.MULTILINE):
            contents=re.sub(pattern,lambda m:line,contents,flags=re.MULTILINE)
        else:
            anchor = ('A02 (A01 + Weather Adapter)' if run['variant']=='a' else
                      f'A03-{chr(ord(run["variant"])-1)} (History-Aware Weather Adapter)')
            contents=re.sub(r'(^\| '+re.escape(anchor)+r' \|.*$)',lambda m:m[1]+'\n'+line,contents,count=1,flags=re.MULTILINE)
    with table.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    heading='## A03 History-Aware Weather Adapter（2026-10-12）'
    block=(heading+'\n\n用户批准保留A02协议并记录限制，继续三个独立A03版本：a为历史均值/标准差/有效比例，b为成对有效的有符号Pearson/有效比例/低方差有效性，c为逐小时小MLP对历史功率、天气、乘积、有效标记与相对位置编码后池化。三者均以32维关系向量的1+tanh门条件化A02未来天气编码，不串联，不复制Cross-Unet。\n\n'
        '仅zeroshot_v1、seed42、24→24；复用冻结A01，Weather Adapter从seed42重新初始化并与新分支共同训练，max100/patience10、batch512/eval1024、FP32、AdamW lr0.000016/wd0.05、L1、验证MAE选权重含epoch0。初始输出等于同样初始化的A02/A01，不从已拟合A02继续增加训练预算。a/c新增2624、共31873参数，b新增2528、共31777参数，三者总可训练参数相差0.31%，比A02多约9%；未开展参数匹配控制，不能将改善归因于历史关系机制。\n\n'
        '保留既有分窗前NWP双向插值与未来NWP发布时间缺失限制。原CSV共29项缺失，站点0–19涉及23项；全网格与仅前缀插值在9项原缺失值上不同。新增分支只使用起点前历史，检查不等同于证明数据在线可用。审查/协议核对见同目录review.json，GPU形状、有限梯度、冻结哈希、初始一致性、原生/缓存、时间边界及窗口/站点/scaler核对见preflight_gpu.json。\n\n'
        f'当前状态：`{record["state"]}`；'+ '；'.join(f'A03-{r["variant"]}={r["state"]}' for r in record['runs'])+'。\n\n'
        '唯一入口scripts/run_chronos2_history_weather_adapter.py，唯一记录reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/runs.json；每组正式测试和验收后才登记既有Adapter独立表，不改A01/A02、官方主表或其他队列。逐horizon、各自最差10%及A02共同最差10%窗口、按训练NWP变动强度三分位分层与相对A02改善保存于各输出analysis.json；仅为单seed探索证据。42/2026/2027稳定性与参数匹配实验本次不自动启动，三组完成后停止。\n')
    if heading in contents:
        start=contents.index(heading);end=contents.find('\n## ',start+len(heading))
        contents=contents[:start]+block+(contents[end:] if end>=0 else '')
    else:
        contents=contents.rstrip()+'\n\n'+block
    doc.write_text(contents)


def execute(stage, gpu, log, variant=None, utilization=None):
    # Utilization can lag behind process exit for one device sampling interval.
    # Wait for true readiness; never ignore another process or start on a busy GPU.
    from utils.resources import require_idle_gpus
    for attempt in range(12):
        try:
            require_idle_gpus(str(gpu))
            break
        except ValueError:
            if attempt == 11:
                raise
            time.sleep(1)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage,'--gpu',str(gpu)]
    if variant:
        command+=['--variant',variant]
    with log.open('x') as handle:
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if utilization:
            with utilization.open('x',newline='') as samples:
                writer=csv.writer(samples);writer.writerow(['timestamp','gpu','memory_mib','utilization_percent'])
                while process.poll() is None:
                    value=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                    writer.writerow([now(),gpu,*[v.strip() for v in value]]);samples.flush()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
        else:
            process.wait()
    if process.returncode:
        raise RuntimeError(f'{stage} {variant or ""} exited {process.returncode}; see {log}')
    return process.pid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--stage',choices=['run','check','worker','accept'],default='run')
    parser.add_argument('--variant',choices=list('abc'))
    parser.add_argument('--continue-after-idle',action='store_true',
                        help='Explicitly finish acceptance after the recorded post-exit idle-check failure; no retraining')
    opts=parser.parse_args()
    if opts.stage=='check':
        preflight(opts.gpu);return
    if opts.stage=='accept':
        if opts.variant is None:parser.error('--variant required')
        accept(opts.variant,opts.gpu);return
    if opts.stage=='worker':
        if opts.variant is None:parser.error('--variant required')
        np,torch=configure(opts.gpu)
        from exp.exp_chronos2_history_weather_adapter import Exp_Chronos2_History_Weather_Adapter
        protect_originals();reseed(torch,np)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',stream=sys.stdout)
        exp=Exp_Chronos2_History_Weather_Adapter(make_args(opts.variant,opts.gpu))
        exp.train(exp.args.model_id);exp.test(exp.args.model_id);return
    os.sched_setaffinity(0,[9,10])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        continuation = None
        if RECORD.exists():
            if not opts.continue_after_idle:
                raise FileExistsError('A03 record exists; no automatic duplicate or restart')
            continuation = json.loads(RECORD.read_text())
            first, third = continuation['runs'][0], continuation['runs'][2]
            a_idle = (continuation.get('failure','').startswith('accept a exited 1;') and first['state']=='failed'
                      and first.get('returncode')==0 and all(r['state']=='pending' for r in continuation['runs'][1:])
                      and (REPORT/'a_acceptance_idle_attempt1.log').exists()
                      and 'GPU 0 is occupied' in (REPORT/'a_acceptance_idle_attempt1.log').read_text())
            c_idle = (continuation.get('failure','').startswith('worker c exited 1;') and third['state']=='failed'
                      and all(r['state']=='completed_artifact_acceptance_passed' for r in continuation['runs'][:2])
                      and not (Path(third['checkpoint_dir'])/'checkpoint.pth').exists()
                      and not Path(third['output_dir']).exists()
                      and (REPORT/'c_startup_idle_attempt1.log').exists()
                      and 'GPU 0 is occupied' in (REPORT/'c_startup_idle_attempt1.log').read_text())
            if not (a_idle or c_idle):
                raise ValueError('This is not the recorded idle-handoff failure; do not implicitly restart')
        protect_originals()
        runs=[]
        for variant in 'abc':
            args=make_args(variant,opts.gpu)
            for root in (args.output_dir,args.checkpoints):
                if not continuation and (Path(root)/args.protocol/args.model_id).exists():
                    raise FileExistsError('A03 artifacts exist; do not repeat')
            runs.append(dict(variant=variant,state='pending',model_id=args.model_id,
                output_dir=str(Path(args.output_dir)/args.protocol/args.model_id),
                checkpoint_dir=str(Path(args.checkpoints)/args.protocol/args.model_id),
                log=str(REPORT/f'{variant}_training.log')))
        record=dict(state='preflight',started_at=now(),coordinator_pid=os.getpid(),gpu=opts.gpu,
                    cpu_cores=[9,10],threads=2,num_workers=0,source_sha256=source_identity(),
                    protocol='zeroshot_v1',seed=42,authorization='User: preserve A02 protocol and record limitations, continue A03',
                    multi_seed_started=False,parameter_matched_control_started=False,runs=runs)
        if continuation:
            record = continuation
            record.setdefault('lifecycle_corrections',[]).append(dict(
                at=now(), failure=record.pop('failure'), failed_at=record.pop('failed_at'),
                previous_coordinator_pid=record['coordinator_pid'],
                correction=('Bounded readiness check at GPU phase handoff; complete a acceptance without retraining/full reinference, then run pending b/c'
                            if a_idle else 'Require two stable idle checks inside worker before loading; c was not trained, a/b are accepted and not repeated'),
                numerical_code_unchanged=True, gpu_preflight_repeated=False,
                previous_runner=str(REPORT/'runner_before_idle_handoff_fix.py'),
                current_execution_source_sha256=source_identity()))
            record.update(state='preflight',coordinator_pid=os.getpid(),gpu=opts.gpu)
            if a_idle:
                record['runs'][0]['state']='complete_pending_acceptance'
            else:
                record['runs'][2]['state']='pending'
        save(RECORD,record);sync_table(record)
        try:
            preflight_path = REPORT/'preflight_gpu.json'
            if preflight_path.exists():
                checked = json.loads(preflight_path.read_text())
                if checked['status']!='passed' or checked['source_sha256']!=source_identity():
                    raise ValueError('Existing preflight is unresolved or uses different code')
            else:
                execute('check',opts.gpu,REPORT/'preflight_gpu.log')
            record.update(state='running');save(RECORD,record)
            for variant in 'abc':
                record=json.loads(RECORD.read_text())
                run=next(row for row in record['runs'] if row['variant']==variant)
                if run['state']=='completed_artifact_acceptance_passed':
                    continue
                if run['state']!='complete_pending_acceptance':
                    run.update(state='training',started_at=now());save(RECORD,record);sync_table(record)
                    print(f'{now()} A03-{variant}: training started GPU{opts.gpu}',flush=True)
                    run['pid']=execute('worker',opts.gpu,Path(run['log']),variant,REPORT/f'{variant}_gpu_utilization.csv')
                    run.update(state='complete_pending_acceptance',returncode=0,finished_at=now());save(RECORD,record)
                execute('accept',opts.gpu,REPORT/f'{variant}_acceptance.log',variant)
                record=json.loads(RECORD.read_text());sync_table(record)
                item=next(row for row in record['runs'] if row['variant']==variant)
                print(f'{now()} A03-{variant}: accepted {item["metrics"]}',flush=True)
            record.update(state='completed_artifact_acceptance_passed',completed_at=now(),
                          acceptance=[row['acceptance'] for row in record['runs']])
        except Exception as error:
            record=json.loads(RECORD.read_text())
            record.update(state='failed',failure=str(error),failed_at=now())
            for row in record['runs']:
                if row['state'] in ('training','complete_pending_acceptance'):row['state']='failed'
            save(RECORD,record);sync_table(record);raise
        save(RECORD,record);sync_table(record)


if __name__=='__main__':
    main()
