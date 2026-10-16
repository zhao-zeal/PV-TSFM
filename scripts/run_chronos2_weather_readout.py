"""Approved A04-a/b: seed42 two protocols, matched A02 budget, serial checks/run/accept."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chronos2_history_weather_adapter import configure, reseed, now, save, evaluate_details

REPORT = ROOT / 'reports/Chronos2WeatherReadout_two_protocols_seed42_20261012'
RECORD = REPORT / 'runs.json'
SOURCE_FILES = ('layers/chronos2_weather_readout.py', 'models/Chronos2WeatherReadout.py',
                'exp/exp_chronos2_weather_readout.py', 'scripts/run_chronos2_weather_readout.py',
                'exp/exp_chronos2_history_weather_adapter.py', 'scripts/run_chronos2_history_weather_adapter.py')


def source_identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def protect_originals():
    review = json.loads((ROOT / 'reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/review.json').read_text())
    for name, expected in review['files_reviewed'].items():
        # Framework acceptance/isolation was updated; retain numerical-model/data protection.
        if not name.startswith(('models/', 'layers/', 'data_provider/')):
            continue
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original implementation changed: {name}')
    for protocol in ('in_domain', 'zeroshot_v1'):
        for prefix in ('checkpoint', 'base_checkpoint'):
            row = review['a02'][protocol]
            if hashlib.sha256(Path(row[prefix]).read_bytes()).hexdigest() != row[f'{prefix}_sha256']:
                raise ValueError('Protected A01/A02 checkpoint changed')


def make_args(variant, gpu, protocol):
    from run import parse_args
    args = parse_args(['--model', 'Chronos2WeatherAdapter', '--protocol', protocol,
                      '--is_training', '1', '--seed', '42', '--use_gpu', '1', '--gpu', str(gpu), '--num_workers', '0'])
    args.model = 'Chronos2WeatherReadout'
    args.memory_variant = variant
    args.adapter_version = f'b131_linear_weather_readout_{variant}_v1'
    args.experiment_label = f'A04-{variant}'
    args.baseline_config = 'weather_conditioned_chronos_readout_variant'
    args.model_id = f'Chronos2WeatherReadout_A04{variant}_seed42_sl24_pl24_history_future'
    args.a04_source_sha256 = source_identity()
    args.protocol_limit = 'A02 gridwise bidirectional NWP interpolation and assumed issue-time availability retained by explicit user decision'
    return args


def weather_embedding(model, future):
    import torch
    missing = future.isnan()
    return model.weather_adapter.cov_out(torch.cat((future.masked_fill(missing, 0), missing.to(future.dtype)), -1))


def preflight(gpu):
    np, torch = configure(gpu)
    from exp.exp_chronos2_weather_readout import Exp_Chronos2_Weather_Readout
    from models.Chronos2WeatherAdapter import Model as A02Model
    from scripts.check_chronos2_weather_adapter import state_digest
    protect_originals()
    result = dict(status='checking', stage='real_GPU_correctness_checks', full_training_complete=False,
                  formal_test_complete=False, source_sha256=source_identity(), variants=[])
    reseed(torch, np)
    reference = A02Model(make_args('a', gpu, 'zeroshot_v1')).float().to(f'cuda:{gpu}')
    initial_weather_sha = state_digest(reference.weather_adapter)
    del reference
    torch.cuda.empty_cache()
    for protocol, variant in ((p,v) for p in ('in_domain','zeroshot_v1') for v in 'ab'):
        reseed(torch, np)
        exp = Exp_Chronos2_Weather_Readout(make_args(variant, gpu, protocol))
        model = exp.model
        assert state_digest(model.weather_adapter) == initial_weather_sha
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 29249
        model.train()
        assert not any(m.training for m in model.base.modules())
        before = state_digest(model.base)
        # Small native check precedes any expensive full context-cache generation.
        from data_provider.data_factory import data_provider
        ds, _ = data_provider(exp.args, 'train')
        batch = [ds[i] for i in (100,1000,len(ds)//2,len(ds)-1)]
        from torch.utils.data._utils.collate import default_collate
        power, target, _, _, cov = default_collate(batch)
        power, history, future = [v.to(exp.device) for v in (power,cov['historical_nwp'],cov['future_nwp'])]
        with torch.no_grad():
            native = model.base.encode_inputs(power[...,0], history, future)
            extended = model.encode_memory(power[...,0], history, future)
            assert all(torch.equal(a,b) for a,b in zip(native,extended[:3]))
            assert extended[3].shape == (len(power),2,768)
            output, attention = model.forward_features(*extended[:3], future, extended[3], return_attention=True)
            initialized = A02Model.forward_features(model,*native,future)
            assert torch.equal(output,initialized)
            assert attention.shape == (len(power),24,2 if variant=='a' else 4)
            assert torch.allclose(attention.sum(-1),torch.ones_like(attention[...,0]),atol=1e-6)
            changed = model(power,None,target.to(exp.device)+1000,None,dict(historical_nwp=history,future_nwp=future))
            assert torch.equal(output,changed)
        splits=[]
        for flag in ('train','val','test'):
            ds, loader = exp._get_data(flag)
            splits.append(dict(split=flag,windows=len(ds),input_manifest_sha256=hashlib.sha256((exp.history_dir/flag/'complete.json').read_bytes()).hexdigest()))
            print(f'{protocol} A04-{variant}: checked {flag} cache ({len(ds)} windows)',flush=True)
        ds, loader = exp._get_data('train')
        indices=[100,1000,len(ds)//2,len(ds)-1]
        batch=[v[indices] for v in loader.dataset.tensors]
        values=[v.to(exp.device) for v in batch]
        embeds,loc,scale,target,sites,times,power,history,future=values[:9]
        context=values[9] if len(values)==10 else None
        with torch.no_grad():
            cached=model.forward_features(embeds,loc,scale,future,context)
            native=model(power,None,None,None,dict(historical_nwp=history,future_nwp=future))
            difference=float((native-cached).abs().max())
            assert difference<=2e-6,difference
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=exp.args.learning_rate,weight_decay=.05,betas=(.9,.999))
        steps=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output,attention=model.forward_features(embeds,loc,scale,future,context,return_attention=True)
            attention.retain_grad()
            loss=torch.nn.functional.l1_loss(output,target)
            assert output.shape==target.shape and torch.isfinite(output).all() and torch.isfinite(loss)
            loss.backward()
            gradients={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.requires_grad
                       and p.grad is not None and torch.isfinite(p.grad).all()}
            assert len(gradients)==sum(p.requires_grad for p in model.parameters())
            assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
            if step==1:
                assert all(v>0 for v in gradients.values())
                assert attention.grad is not None and torch.isfinite(attention.grad).all() and attention.grad.abs().sum()>0
            steps.append(dict(step=step+1,loss=float(loss.detach()),gradient_l1=gradients,
                              attention_gradient_l1=float(attention.grad.abs().sum())))
            optimizer.step()
        with torch.no_grad():
            query=weather_embedding(model,future)
            ordinary,weights=model.forward_features(embeds,loc,scale,future,context,return_attention=True)
            changed,other_weights=model.forward_features(embeds,loc,scale,future,context,query_weather=torch.zeros_like(query),return_attention=True)
            assert float((weights-other_weights).abs().max())>1e-6
            query_delta=float((ordinary-changed).abs().max())
            assert query_delta>0
            missing=future.clone();missing[:,0,0]=torch.nan
            assert torch.isfinite(model.forward_features(embeds,loc,scale,missing,context)).all()
            assert float((weights[:,0]-weights[:,1]).abs().max())>0  # Horizon encoding can distinguish queries.
        assert state_digest(model.base)==before
        result['variants'].append(dict(protocol=protocol,variant=variant,base_provenance=model.base_provenance,
            trainable_parameters=29249,frozen_parameters=sum(p.numel() for p in model.base.parameters()),
            memory_shape=[4,2 if variant=='a' else 4,768],attention_shape=list(weights.shape),
            initial_weather_weights_equal_seed42_A02=True,epoch0_output_exactly_initialized_A02=True,
            native_encode_forecast_tokens_loc_scale_exactly_equal=True,native_cache_max_abs_difference=difference,
            frozen_state_sha256=before,frozen_gradients_absent_weights_unchanged=True,finite_output_all_gradients=True,
            trainable_readout_and_attention_receive_nonzero_gradients_step2=True,
            query_only_intervention_prediction_max_difference=query_delta,missing_weather_finite=True,
            no_future_power_or_decoder_access=True,full_splits_window_cache_alignment=splits,transient_steps=steps))
        del exp,model,optimizer,loader,batch,values
        torch.cuda.empty_cache()
    result['status']='passed'
    save(REPORT/'preflight_gpu.json',result)
    print('A04 preflight passed: 29249 trainable parameters per variant',flush=True)


def analyze_attention(exp, loader, out, pred, truth, common):
    import numpy as np
    import torch
    from utils.metrics import metric
    attention_batches=[]
    changed_predictions={name:[] for name in ('zero_weather_query','permuted_weather_query','uniform_attention')}
    change_sums={name:0. for name in changed_predictions}
    cosine=[]
    exp.model.eval()
    with torch.no_grad():
        for batch in loader:
            v=[t.to(exp.device) for t in batch]
            embeds,loc,scale=v[:3];future=v[8];context=v[9] if len(v)==10 else None
            prediction,attention=exp.model.forward_features(embeds,loc,scale,future,context,return_attention=True)
            attention_batches.append(attention.cpu().numpy())
            query=weather_embedding(exp.model,future)
            for name,options in (('zero_weather_query',dict(query_weather=torch.zeros_like(query))),
                    ('permuted_weather_query',dict(query_weather=query.roll(1,0))),('uniform_attention',dict(uniform=True))):
                changed,other=exp.model.forward_features(embeds,loc,scale,future,context,return_attention=True,**options)
                changed_predictions[name].append(changed.cpu().numpy())
                change_sums[name]+=float((other-attention).abs().sum())
            adapted=exp.model.base.adapter(embeds)
            memory=torch.cat((context,adapted),1) if context is not None else adapted
            projected=exp.model.weather_adapter.hidden_state_out(memory)
            normalized=torch.nn.functional.normalize(projected,dim=-1)
            mask=torch.triu(torch.ones(memory.size(1),memory.size(1),dtype=torch.bool,device=exp.device),diagonal=1)
            cosine.append((normalized@normalized.transpose(-1,-2))[:,mask].cpu().numpy())
    weights=np.concatenate(attention_batches)
    np.save(out/'attention.npy',weights)
    entropy=-(weights.astype(np.float64)*np.log(np.maximum(weights,1e-30))).sum(-1)/np.log(weights.shape[-1])
    diagnostics={}
    for name,batches in changed_predictions.items():
        changed=np.concatenate(batches)
        diagnostics[name]=dict(metrics=metric(changed,truth),A02_worst10_common_subset=metric(changed[common],truth[common]),
            prediction_mean_abs_change=float(np.abs(changed-pred).mean()),prediction_max_abs_change=float(np.abs(changed-pred).max()),
            attention_mean_abs_change=change_sums[name]/weights.size)
    return dict(memory_order=['adapted_forecast_1','adapted_forecast_2'] if exp.args.memory_variant=='a' else
                ['raw_context_1','raw_context_2','adapted_forecast_1','adapted_forecast_2'],
        mean_weights_by_horizon=weights.mean(0).tolist(),A02_worst10_mean_weights_by_horizon=weights[common].mean(0).tolist(),
        mean_normalized_entropy=float(entropy.mean()),entropy_by_horizon=entropy.mean(0).tolist(),
        max_mean_abs_deviation_from_uniform=float(np.abs(weights-1/weights.shape[-1]).mean()),
        across_sample_weight_std_by_horizon=weights.std(0).tolist(),
        mean_pairwise_projected_token_cosine=float(np.concatenate(cosine).mean()),
        query_only_interventions=diagnostics,
        limitation='Diagnostic interventions only: memory and direct weather branch fixed; permutation is within eval batches. Attention is not causal attribution; context positions already contain known future NWP.')


def accept(variant, gpu, protocol):
    np, torch = configure(gpu)
    from exp.exp_chronos2_weather_readout import Exp_Chronos2_Weather_Readout
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from scripts.check_chronos2_weather_adapter import state_digest
    from utils.metrics import metric
    protect_originals()
    from types import SimpleNamespace
    saved_directory = ROOT / 'outputs' / protocol / f'Chronos2WeatherReadout_A04{variant}_seed42_sl24_pl24_history_future'
    saved_config = json.loads((saved_directory / 'config.json').read_text())
    for name, expected in saved_config['a04_source_sha256'].items():
        if name.startswith(('models/', 'layers/')) and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Numerical implementation changed: {name}')
    args = SimpleNamespace(**saved_config)
    args.gpu = gpu  # Replays may use a different idle device; historical config stays intact.
    exp = Exp_Chronos2_Weather_Readout(args)
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
    a02_out = ROOT/'outputs'/args.protocol/'Chronos2WeatherAdapter_A02_seed42_sl24_pl24_history_future'
    for name in ('true','site_ids','forecast_timestamps'):
        assert np.array_equal(arrays[name],np.load(a02_out/f'{name}.npy')), name
    saved = json.loads((out/'metrics.json').read_text())
    for key,value in metric(pred,truth).items():
        assert value == saved[key], key
    test_protocol = json.loads((out/'data_protocol.json').read_text())
    assert test_protocol == json.loads((a02_out/'data_protocol.json').read_text())
    for directory in (out,ck):
        with np.load(directory/'scalers.npz') as current,np.load(a02_out/'scalers.npz') as old:
            assert set(current.files)==set(old.files) and all(np.array_equal(current[k],old[k]) for k in old.files)
    training_protocol = json.loads((ck/'data_protocol.json').read_text())
    expected_sites = list(range(10,20)) if protocol=='zeroshot_v1' else list(range(10))
    assert training_protocol['sites']==training_protocol['train_sites']==expected_sites
    assert test_protocol['test_sites']==list(range(10)) and test_protocol['windows']==dict(train=76570,validation=25450,test=25450)
    for key in ('version','train_sites','validation_sites','test_sites','windows','target_time_ranges','scaler_fit','data_interface','modalities'):
        assert training_protocol[key]==test_protocol[key], key
    for site in test_protocol['test_sites']:
        values=pred[arrays['site_ids']==site]
        assert len(values)==2545 and np.any(values!=0)
    for state in ('weather_state_dict',):
        assert all(torch.isfinite(v).all() for v in selected[state].values())
    exp.model.eval()
    before=state_digest(exp.model.base)
    check = json.loads((REPORT/'preflight_gpu.json').read_text())
    checked=next(row for row in check['variants'] if row['variant']==variant and row['protocol']==protocol)
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
    run=next(row for row in record['runs'] if row['variant']==variant and row['protocol']==protocol)
    log=Path(run['log']).read_text()
    assert len(re.findall('Complete test metrics:',log))==1
    assert [int(x) for x in re.findall(r'A04-'+variant+r' epoch (\d+)',log)]==list(range(1,len(history)))
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
        'A04 and A02 have exactly 29249 trainable parameters; fixed position encoding adds none',
        'overlapping windows; worst-window sets differ by model, also report common A02 subset',
        'weather intensity is forecast variability, not observed weather uncertainty',args.protocol_limit]
    details['attention_analysis'] = analyze_attention(exp, loader, out, pred, truth, common)
    save(out/'analysis.json',details)
    if not (REPORT/f'{protocol}_A02_reference_analysis.json').exists():
        save(REPORT/f'{protocol}_A02_reference_analysis.json',reference)
    Exp_Long_Term_Forecast._record_result(exp,out,saved, specific_checks={'source': __file__})
    table=ROOT/'reports'/args.protocol/'results.csv'
    with table.open() as handle:
        rows=[r for r in csv.DictReader(handle) if r['result_dir']==str(out.resolve())]
    assert len(rows)==1 and all(float(rows[0][k])==saved[k] for k in ('mae','rmse','r2'))
    item=dict(status='passed_full_training_test_artifact_acceptance',protocol=protocol,variant=variant,metrics=saved,
        completed_epochs=summary['completed_epochs'],best_epoch=selected['epoch'],
        trainable_parameters=summary['trainable_parameters'],
        checkpoint_sha256=hashlib.sha256((ck/'checkpoint.pth').read_bytes()).hexdigest(),
        frozen_state_sha256=before,selected_checkpoint_replay_max_abs_difference=difference,
        base_provenance=exp.model.base_provenance, analysis_path=str(out/'analysis.json'),
        comparison_to_A02=details['comparison_to_A02'],accepted_at=now())
    save(REPORT/f'{protocol}_{variant}_acceptance.json',item)
    run.update(state='completed_artifact_acceptance_passed',metrics=saved,acceptance=item)
    save(RECORD,record)
    print(json.dumps(item,ensure_ascii=False),flush=True)


def sync_table(record):
    table=ROOT/'reports/chronos2_adapter_metrics.csv'
    with table.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    doc=ROOT/'docs/EXPERIMENT_STATUS.md';contents=doc.read_text()
    for run in record['runs']:
        if run['state']!='completed_artifact_acceptance_passed':continue
        label=f'A04-{run["variant"]} (Weather-Conditioned Chronos Readout)'
        row=next((r for r in rows if r['models']==label),None)
        if row is None:
            row={key:'' for key in fields}
            row.update(models=label,input='历史功率+历史/未来NWP（天气Query读取'+('预测Token）' if run['variant']=='a' else '历史上下文及预测Token）'))
            rows.append(row)
        prefix='MMSP zeroshot' if run['protocol']=='zeroshot_v1' else 'MMSP in_domain'
        for key,value in (('MAE','mae'),('RMSE','rmse'),('R^2','r2')):
            row[f'{prefix} {key}']=f'{run["metrics"][value]:.6f}'
        line='| '+' | '.join(row[key] or '—' for key in fields)+' |'
        pattern=r'^\| '+re.escape(label)+r' \|.*$'
        if re.search(pattern,contents,flags=re.MULTILINE):
            contents=re.sub(pattern,lambda m:line,contents,flags=re.MULTILINE)
        else:
            anchor='A03-c (History-Aware Weather Adapter)' if run['variant']=='a' else 'A04-a (Weather-Conditioned Chronos Readout)'
            contents=re.sub(r'(^\| '+re.escape(anchor)+r' \|.*$)',lambda m:m[1]+'\n'+line,contents,count=1,flags=re.MULTILINE)
    with table.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    heading='## A04 Weather-Conditioned Chronos Readout（2026-10-12）'
    block=(heading+'\n\n用户确认首轮方案并要求in_domain也做：两个协议、seed42、24→24，单GPU串行四组。A04-a天气Query读取B131后的两个预测Token；A04-b按原encode提取两个功率历史上下文Token，与两个适配后预测Token组成Memory，排除REG，不增加预测patch或上下文长度，不将B131用于历史Token。历史位置状态经双向时间/分组注意力，已包含已知未来NWP，不能称纯历史观测编码。\n\n'
        '未来NWP值及缺失标记[ B,24,30 ]经A02 cov_out得到32维天气表示，加固定正弦小时位置构成Query。共享A02 LayerNorm/Linear投影Memory为K=V，单头32维，softmax(QKᵀ/√32)读取[ B,24,32 ]，与原天气表示拼接后走原残差MLP；最终Y_A01+Delta。两版均29249可训练参数，与A02完全相同；A01全部冻结。两个预测Token的注意力仅为两向量之间的动态混合，不能解释为24个独立Chronos小时状态。\n\n'
        'Weather Readout复用seed42的A02初始权重与随机数流，从头训练，不热启动拟合A02。FP32、batch512/eval1024、AdamW lr0.000016/wd0.05、L1、max100/patience10，验证MAE选权重含epoch0。沿用原始站点、窗口、scaler及A02 NWP双向插值/发布时间缺失限制，不更改公共接口、A01/A02实现或checkpoint，不依据测试成绩调参。\n\n'
        f'当前状态：`{record["state"]}`；GPU{record["gpu"] if record["gpu"]>=0 else "待空闲"}，CPU9,10两线程、num_workers0；'+ '；'.join(f'{r["protocol"]}/A04-{r["variant"]}={r["state"]}' for r in record['runs'])+'。\n\n'
        '唯一入口scripts/run_chronos2_weather_readout.py；唯一记录reports/Chronos2WeatherReadout_two_protocols_seed42_20261012/runs.json。检查证据为同目录preflight_gpu.json；b历史Token缓存单独保存并逐窗口核对原forecast、A01预测、全部输入/标签/站点/时间及历史边界。各输出保留attention.npy、analysis.json：全量MAE/RMSE/R²、24个horizon、各自及固定A02最差10%窗口、注意力熵/均值/样本差异/Memory相似度；零天气Query、同批天气Query置换、均匀注意力均保持Memory及直接天气支路不变，只用于诊断。注意力不是因果解释。全部正式测试及产物验收后登记既有Adapter独立表，完成即停，不扩展种子或调参。\n')
    if heading in contents:
        start=contents.index(heading);end=contents.find('\n## ',start+len(heading))
        contents=contents[:start]+block+(contents[end:] if end>=0 else '')
    else:
        contents=contents.rstrip()+'\n\n'+block
    authorization='- 2026-10-12 A04：用户批准实现并运行A04-a/b，随后补充in_domain也要做；两个协议seed42、A02预算，单GPU串行四组，验收后停止。证据为reports/Chronos2WeatherReadout_two_protocols_seed42_20261012/runs.json。\n'
    if authorization not in contents:contents=contents.replace('## 当前授权\n','## 当前授权\n\n'+authorization,1)
    doc.write_text(contents)


def wait_for_idle(gpu):
    from utils.resources import require_idle_gpus
    first = True
    while True:
        candidates = range(8) if gpu < 0 else (gpu,)
        for candidate in candidates:
            try:
                require_idle_gpus(str(candidate))
                time.sleep(1)
                require_idle_gpus(str(candidate))
                return candidate
            except ValueError:
                continue
        if first:
            print(f'{now()} waiting for an idle GPU; no CUDA allocated', flush=True)
            first = False
        time.sleep(30)


def execute(stage,gpu,log,protocol=None,variant=None,utilization=None):
    wait_for_idle(gpu)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage,'--gpu',str(gpu)]
    if protocol:command+=['--protocol',protocol,'--variant',variant]
    with log.open('x') as handle:
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if utilization:
            with utilization.open('x',newline='') as samples:
                writer=csv.writer(samples);writer.writerow(['timestamp','gpu','memory_mib','utilization_percent'])
                while process.poll() is None:
                    value=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                    writer.writerow([now(),gpu,*[v.strip() for v in value]]);samples.flush()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
        else:process.wait()
    if process.returncode:raise RuntimeError(f'{stage} {protocol or ""} {variant or ""} exited {process.returncode}; see {log}')
    return process.pid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=-1,help='-1: wait for any completely idle GPU; otherwise fixed GPU')
    parser.add_argument('--stage',choices=['run','check','worker','accept'],default='run')
    parser.add_argument('--variant',choices=list('ab'))
    parser.add_argument('--protocol',choices=['in_domain','zeroshot_v1'])
    opts=parser.parse_args()
    if opts.stage=='check':preflight(opts.gpu);return
    if opts.stage in ('worker','accept') and (opts.variant is None or opts.protocol is None):
        parser.error('--protocol and --variant required')
    if opts.stage=='accept':accept(opts.variant,opts.gpu,opts.protocol);return
    if opts.stage=='worker':
        np,torch=configure(opts.gpu)
        from exp.exp_chronos2_weather_readout import Exp_Chronos2_Weather_Readout
        protect_originals();reseed(torch,np)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',stream=sys.stdout)
        exp=Exp_Chronos2_Weather_Readout(make_args(opts.variant,opts.gpu,opts.protocol))
        exp.train(exp.args.model_id);exp.test(exp.args.model_id);return
    os.sched_setaffinity(0,[9,10])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if RECORD.exists():raise FileExistsError('A04 record exists; no automatic duplicate or restart')
        protect_originals()
        runs=[]
        for protocol in ('in_domain','zeroshot_v1'):
            for variant in 'ab':
                args=make_args(variant,opts.gpu,protocol)
                for root in (args.output_dir,args.checkpoints):
                    if (Path(root)/protocol/args.model_id).exists():raise FileExistsError('A04 artifacts exist; do not repeat')
                runs.append(dict(protocol=protocol,variant=variant,state='pending',model_id=args.model_id,
                    output_dir=str(Path(args.output_dir)/protocol/args.model_id),
                    checkpoint_dir=str(Path(args.checkpoints)/protocol/args.model_id),
                    log=str(REPORT/f'{protocol}_{variant}_training.log')))
        record=dict(state='waiting_for_idle_gpu',started_at=now(),coordinator_pid=os.getpid(),gpu=opts.gpu,
            cpu_cores=[9,10],threads=2,num_workers=0,source_sha256=source_identity(),seed=42,
            authorization='User approved A04 proposal, then explicitly included in_domain; four serial runs',
            multi_seed_started=False,hyperparameter_search_started=False,runs=runs)
        save(RECORD,record);sync_table(record)
        try:
            opts.gpu=wait_for_idle(opts.gpu)
            record.update(state='preflight',gpu=opts.gpu)
            save(RECORD,record);sync_table(record)
            execute('check',opts.gpu,REPORT/'preflight_gpu.log')
            record.update(state='running');save(RECORD,record)
            for protocol in ('in_domain','zeroshot_v1'):
                for variant in 'ab':
                    record=json.loads(RECORD.read_text())
                    run=next(row for row in record['runs'] if row['protocol']==protocol and row['variant']==variant)
                    run.update(state='training',started_at=now());save(RECORD,record);sync_table(record)
                    print(f'{now()} {protocol}/A04-{variant}: training started GPU{opts.gpu}',flush=True)
                    run['pid']=execute('worker',opts.gpu,Path(run['log']),protocol,variant,REPORT/f'{protocol}_{variant}_gpu_utilization.csv')
                    run.update(state='complete_pending_acceptance',returncode=0,finished_at=now());save(RECORD,record)
                    execute('accept',opts.gpu,REPORT/f'{protocol}_{variant}_acceptance.log',protocol,variant)
                    record=json.loads(RECORD.read_text());sync_table(record)
                    item=next(row for row in record['runs'] if row['protocol']==protocol and row['variant']==variant)
                    print(f'{now()} {protocol}/A04-{variant}: accepted {item["metrics"]}',flush=True)
            record.update(state='completed_artifact_acceptance_passed',completed_at=now(),acceptance=[row['acceptance'] for row in record['runs']])
            protect_originals()
        except Exception as error:
            record=json.loads(RECORD.read_text());record.update(state='failed',failure=str(error),failed_at=now())
            for row in record['runs']:
                if row['state'] in ('training','complete_pending_acceptance'):row['state']='failed'
            save(RECORD,record);sync_table(record);raise
        save(RECORD,record);sync_table(record)


if __name__=='__main__':main()
