"""One frozen validation-selected gate checkpoint: full test and artifact acceptance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/A01_gate_two_protocols_seed42_20261015'


def evaluate(protocol,mode,gpu):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,default_collate
    from scripts.run_a01_dynamic_gate import setup,raw_batch
    from scripts.run_a01_gate_two_protocols import save
    from data_provider.data_factory import data_provider
    from utils.resources import require_idle_gpus
    from utils.metrics import metric
    from utils.acceptance import _check_samples,_check_scalers,_update_index
    require_idle_gpus(str(gpu))
    directory=REPORT/protocol/mode
    state=json.loads((directory/'runs.json').read_text())
    assert state['acceptance']['status']=='passed_validation_artifact_acceptance'
    source=Path(state['output_dir']); cp=Path(state['checkpoint_dir'])/'checkpoint.pth'
    w=torch.load(cp,map_location='cpu',weights_only=True)
    config=json.loads((source/'config.json').read_text());assert config==w['config']
    history=json.loads((source/'linear_history.json').read_text())
    best=min(history,key=lambda row:row['mae']);assert best['epoch']==w['epoch']
    assert abs(best['mae']-w['validation_metrics']['mae'])<1e-12
    base=torch.load(cp.parent/'selector_b131.pth',map_location='cpu',weights_only=True)
    base_history=json.loads((source/'selector_b131_history.json').read_text())
    assert base['epoch']==min(base_history,key=lambda row:row['mae'])['epoch']
    for key in ('selector_state_dict','adapter_state_dict'):
        assert all(torch.equal(v,w[key][k]) for k,v in base[key].items())
    for key in ('selector_state_dict','adapter_state_dict','readout_state_dict'):
        assert all(torch.isfinite(t).all() for t in w[key].values())
    name=state['run_id']+'_formal_test_20261016'
    output=ROOT/'outputs'/protocol/name
    output.mkdir(exist_ok=False)
    device=f'cuda:{gpu}'
    args,model=setup(device,mode,config['fixed_gate_input'],protocol)
    for attr in ('selector','adapter','readout'):getattr(model,attr).load_state_dict(w[attr+'_state_dict'])
    model.stage='b131_linear_v1';model.eval()
    dataset,_=data_provider(args,'test')
    protocol_data=json.loads(json.dumps(dataset.save_protocol(output)))
    training_protocol=json.loads((source/'data_protocol.json').read_text())
    assert all(protocol_data.get(k)==training_protocol.get(k) for k in protocol_data if k!='sites')
    scalers=_check_scalers(output,protocol_data)
    training_scalers=_check_scalers(source,training_protocol)
    assert scalers.keys()==training_scalers.keys() and all(np.array_equal(v,training_scalers[k]) for k,v in scalers.items())
    test_config=dict(config,model='Chronos2DynamicGate',baseline_config=mode+'_gate',model_id=name,
                     is_training=0,checkpoint=str(cp),gpu=gpu,source_training_output=str(source),
                     evaluation_split='test',selection='validation MAE; frozen before test',
                     test_policy='One formal test authorized 2026-10-16; no test-based selection',
                     source_training_test_policy=config['test_policy'])
    save(output/'config.json',test_config)
    arrays={k:[] for k in ['pred','true','site_ids','forecast_timestamps']}
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0):
            p,h,f,y=raw_batch(batch,device)
            pred=model.forward_features(*model.encode_inputs(p,h,f))
            assert torch.isfinite(pred).all()
            for key,value in [('pred',pred.cpu().numpy()),('true',y.cpu().numpy()),
                              ('site_ids',batch[4]['site_id'].numpy()),('forecast_timestamps',batch[4]['forecast_timestamps'].numpy())]:
                arrays[key].append(value)
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    for k,v in arrays.items():np.save(output/(k+'.npy'),v)
    metrics=metric(arrays['pred'],arrays['true'])
    metrics.update(model=test_config['model'],baseline_config=test_config['baseline_config'],seed=42,
                   data_protocol=protocol,input_len=24,output_len=24,llm_model='',
                   **{k:protocol_data[k] for k in ['train_sites','validation_sites','test_sites','split_version','scaler_version']})
    save(output/'metrics.json',metrics)
    # Re-read saved arrays, independently check raw CSV labels/sites/times and metric arithmetic.
    saved=[np.load(output/(k+'.npy')) for k in ['pred','true','site_ids','forecast_timestamps']]
    _check_samples(test_config,protocol_data,*saved)
    recomputed=metric(saved[0],saved[1]);assert all(recomputed[k]==metrics[k] for k in recomputed)
    # Reload every trained module from the selected file before original-batch GPU replay.
    replay=torch.load(cp,map_location=device,weights_only=True)
    for attr in ('selector','adapter','readout'):getattr(model,attr).load_state_dict(replay[attr+'_state_dict'])
    batches=[list(range(32,40)),list(range(len(dataset)-len(dataset)%8,len(dataset)))]
    error=0.
    with torch.no_grad():
        for indices in batches:
            p,h,f,y=raw_batch(default_collate([dataset[i] for i in indices]),device)
            pred=model.forward_features(*model.encode_inputs(p,h,f)).cpu().numpy()
            error=max(error,float(np.abs(pred-saved[0][indices]).max()))
    assert error<1e-5
    receipt=dict(status='passed_formal_test_artifact_acceptance',metrics=metrics,
                 source_checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                 selected_epoch=w['epoch'],selection='minimum validation MAE',
                 validation_artifact_acceptance=state['acceptance'],
                 checks=['raw_CSV_full_test_labels_sites_times','finite_nondegenerate_predictions','saved_metric_recomputation',
                         'training_test_protocol_and_scalers','both_stage_validation_selection','stage2_frozen_weights','GPU_checkpoint_replay'],
                 replay_device=device,replay_precision='fp32',replay_batches=batches,
                 prediction_max_abs_error=error,prediction_tolerance=1e-5,
                 test_windows=len(dataset),source_training_config_preserved=True,
                 limitation='Previously observed benchmark test; single seed; future NWP issue-time availability assumed',output_dir=str(output))
    save(output/'acceptance.json',receipt)
    _update_index(output,metrics,ROOT/'reports')
    save(directory/'test_acceptance.json',receipt)
    state.update(test_acceptance={'status':receipt['status'],'evidence':str(directory/'test_acceptance.json')},
                 test_output_dir=str(output),test_accessed=True)
    save(directory/'runs.json',state)
    print(json.dumps(receipt,indent=2),flush=True)


def sync_table(mode):
    receipts=[json.loads((REPORT/p/mode/'test_acceptance.json').read_text()) for p in ['in_domain','zeroshot_v1']]
    assert all(x['status']=='passed_formal_test_artifact_acceptance' for x in receipts)
    name=('固定门控' if mode=='fixed' else '动态门控')+'（正式测试）'
    row=[name,'历史功率+历史/未来NWP（'+('全局固定' if mode=='fixed' else '逐样本逐小时动态')+'软门控）']
    for a in receipts:row += [f"{a['metrics'][k]:.6f}" for k in ['mae','rmse','r2']]
    p=ROOT/'reports/chronos2_adapter_metrics.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
    prefix='固定门控' if mode=='fixed' else '动态门控'
    rows=[r for r in rows if not r[0].startswith(prefix)];rows.append(row)
    with p.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
    p=ROOT/'docs/EXPERIMENT_STATUS.md';s=p.read_text();lines=s.splitlines()
    lines=[line for line in lines if not line.startswith('| '+prefix)]
    section = next(i for i,line in enumerate(lines) if line.startswith('## Chronos-2 Adapter迁移'))
    header = next(i for i in range(section,len(lines)) if lines[i].startswith('| models |'))
    end = header
    while end < len(lines) and lines[end].startswith('|'):
        end += 1
    lines.insert(end, '| '+' | '.join(row)+' |')
    lines = [('门控行已按2026-10-16授权登记正式测试成绩；验证成绩保留在原validation_review.json，不能与测试成绩混用。测试证据为reports/A01_gate_two_protocols_seed42_20261015/{in_domain,zeroshot_v1}/{fixed,dynamic}/test_acceptance.json（仅已完成版本）。'
              if line.startswith('2026-10-16用户明确批准新增“固定门控') else line) for line in lines]
    s='\n'.join(lines)+'\n'
    s=s.replace('新增固定门控行按用户授权单独标注为验证成绩。','门控行已按2026-10-16授权替换为正式测试成绩。')
    s += f'\n{prefix}两协议正式测试及产物验收完成，已同步Adapter表；原验证成绩仍保留在各validation_review.json。证据reports/A01_gate_two_protocols_seed42_20261015/{{in_domain,zeroshot_v1}}/{mode}/test_acceptance.json。\n'
    p.write_text(s)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['in_domain','zeroshot_v1']);parser.add_argument('--mode',choices=['fixed','dynamic'],required=True);parser.add_argument('--gpu',type=int);parser.add_argument('--sync-table',action='store_true');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        from utils.resources import require_idle_gpus
        from scripts.run_a01_gate_two_protocols import accept
        require_idle_gpus(str(args.gpu))
        q=json.loads((REPORT/args.protocol/'queue.json').read_text())
        run=next(r for r in q['runs'] if r['gate_mode']==args.mode)
        accept(run, device=f'cuda:{args.gpu}')
    elif args.sync_table:sync_table(args.mode)
    else:evaluate(args.protocol,args.mode,args.gpu)
