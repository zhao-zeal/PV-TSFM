"""Six fixed in-domain candidates, validation only; wait for the gate task to end."""
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
REPORT = ROOT / 'reports/Baseline_limited_tuning_in_domain_seed42_20261015'
GATE = ROOT / 'reports/A01_gate_two_protocols_seed42_20261015/runs.json'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def prepare():
    from run import parse_args
    REPORT.mkdir(exist_ok=False)
    refs = {
        'TimeXer': ROOT / 'checkpoints/in_domain/TimeXer_unified100_pat10_rmse_seed42_sl24_pl24_20261014T141649',
        'CrossUnet': ROOT / 'checkpoints/in_domain/CrossUnet_official_seed42_sl24_pl24_power_nwp_full',
        'ChronosX': ROOT / 'checkpoints/in_domain/ChronosX_unified100_pat10_rmse_seed42_sl24_pl24_20261014T141649',
    }
    candidates = [('TimeXer', 'width128', {'d_model':128,'d_ff':128}),
                  ('TimeXer', 'dropout020', {'dropout':.2}),
                  ('CrossUnet', 'lr5e5', {'learning_rate':.00005}),
                  ('CrossUnet', 'lr2e4', {'learning_rate':.0002}),
                  ('ChronosX', 'lr0003', {'learning_rate':.003}),
                  ('ChronosX', 'lr0001', {'learning_rate':.001})]
    baselines = {}
    for model, directory in refs.items():
        if model == 'CrossUnet':
            previous = json.loads((ROOT / 'reports/CrossUnet_two_protocols_seed42_20261010/runs.json').read_text())
            selected = next(r for r in previous['runs'] if r['protocol']=='in_domain')
            val = selected['best_validation_metrics']
        else:
            rows = list(csv.DictReader((directory / 'training_history.csv').open()))
            best = min(rows, key=lambda row: float(row['rmse']))
            val = {k:float(best[k]) for k in ('mae','rmse','mse')}
        baselines[model] = dict(checkpoint_dir=str(directory), validation=val)
    trials = []
    for model, name, changes in candidates:
        old = json.loads((refs[model] / 'config.json').read_text())
        argv = ['--model',model,'--protocol','in_domain','--is_training','1',
                '--baseline_config','unified_prediction','--seed','42','--seq_len','24','--pred_len','24',
                '--train_epochs','100','--patience','10','--monitor','rmse','--num_workers','0','--use_gpu','1',
                '--model_id',f'{model}_limited_tuning_{name}_seed42_sl24_pl24_20261015']
        keys = ['batch_size','eval_batch_size','learning_rate','lradj','lr_patience','precision',
                'd_model','d_ff','e_layers','n_heads','factor','dropout','patch_len','seg_len']
        if model == 'ChronosX':
            keys += ['gradient_accumulation_steps','num_samples','covariate_injection','injection_hidden_dim','injection_layers','model_path']
        for key in keys:
            if old.get(key) is not None:
                argv += ['--'+key, str(changes.get(key,old[key]))]
        config = vars(parse_args(argv))
        # Check all architecture/optimization/input fields against the accepted reference.
        allowed = set(changes) | {'baseline_config','model_id','monitor','gpu','print_config'}
        ignored = {'checkpoint','resume'} | ({'max_steps','eval_steps'} if model == 'ChronosX' else set())
        differences = {k:[v,config.get(k)] for k,v in old.items()
                       if k in config and v != config[k] and k not in ignored}
        assert set(differences) <= allowed, (model,differences)
        trials.append(dict(model=model,candidate=name,changes=changes,argv=argv,
                           parsed_config=config,state='pending',config_differences=differences))
    save(REPORT/'runs.json',dict(state='prepared',scope='Six candidates only; in_domain seed42 24->24; no test evaluation',
         authorization='User requested limited tuning of TimeXer, Cross-Unet, ChronosX; in_domain or Site1 allowed. Selected in_domain and two candidates/model.',
         baselines=baselines,trials=trials,selection='Lowest validation RMSE among existing baseline and two candidates per model; no test-based choice',
         resources='Wait until gate queue completes successfully; then one idle GPU, CPU9/10, threads2, workers0, serial',
         failure_policy='Stop on failure; no automatic rerun or extra search',test_accessed=False))


def trial(index, gpu):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from run import parse_args, configure_resources
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.losses import forecasting_loss
    status = json.loads((REPORT/'runs.json').read_text())
    item = status['trials'][index]
    args = parse_args(item['argv'] + ['--gpu',str(gpu)])
    configure_resources(args)
    def seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
    seed()
    experiment = Exp_Long_Term_Forecast(args)
    data, _ = experiment._get_data('train')
    batch = default_collate([data[32], data[len(data)-1]])
    experiment.model.train()
    if args.model == 'ChronosX':
        x, _, _, _, cov, target = experiment._prepare_batch(batch)
        with experiment._autocast():
            loss = experiment.model.token_loss(x, target, cov, augment=True)
        assert all(not p.requires_grad for n,p in experiment.model.named_parameters() if 'injection' not in n)
    else:
        pred,target,_ = experiment._forward_batch(batch)
        assert pred.shape==target.shape==(2,24,1) and torch.isfinite(pred).all()
        loss=forecasting_loss(pred,target,args.model)
    loss.backward()
    grads=[p.grad for p in experiment.model.parameters() if p.requires_grad and p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads)>0
    save(REPORT/f'{index}_preflight.json',dict(status='passed_real_gpu_train_batch',loss=loss.item(),gpu=gpu,
         sample_indices=[32,len(data)-1],test_accessed=False))
    # Recreate model/seed so the correctness check cannot update training state.
    del experiment, loss, grads
    if args.model != 'ChronosX':
        del pred
    import gc
    gc.collect(); torch.cuda.empty_cache(); seed()
    experiment=Exp_Long_Term_Forecast(args)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    experiment.train(args.model_id)  # Intentionally never call run.main or experiment.test.
    checkpoint=Path(args.checkpoints)/args.protocol/args.model_id
    w=torch.load(checkpoint/'checkpoint.pth',map_location='cpu',weights_only=False)
    rows=list(csv.DictReader((checkpoint/'training_history.csv').open()))
    best=min(rows,key=lambda row:float(row['rmse']))
    assert int(best['epoch'])==w['epoch']
    valdata, loader=experiment._get_data('val')
    metrics=experiment.vali(loader)
    assert len(valdata)==25450
    assert all(abs(metrics[k]-w['validation_metrics'][k])<1e-6 for k in ('mae','rmse','mse'))
    output=Path(args.output_dir)/args.protocol/args.model_id
    experiment._save_config(output,valdata)
    result=dict(state='accepted_validation_only',model=args.model,candidate=item['candidate'],
                metrics=metrics,best_epoch=w['epoch'],completed_epochs=len(rows),validation_windows=len(valdata),
                checkpoint=str(checkpoint/'checkpoint.pth'),output_dir=str(output),test_accessed=False,
                acceptance='Best-checkpoint full validation replay agrees with selected epoch; real GPU gradients finite',
                limitation='Validation tuning result only, not formal test acceptance')
    save(output/'validation_metrics.json',result)
    save(REPORT/f'{index}_result.json',result)


def coordinate():
    from utils.resources import require_idle_gpus
    path=REPORT/'runs.json'; status=json.loads(path.read_text())
    assert status['state']=='prepared'
    status.update(state='waiting_for_gate_completion',coordinator_pid=os.getpid()); save(path,status)
    try:
        while True:
            gate=json.loads(GATE.read_text())
            if gate['state']=='completed_formal_test_acceptance':break
            if gate['state'].startswith('failed'):
                raise RuntimeError('Gate queue failed; do not automatically start tuning')
            time.sleep(60)
        for i,item in enumerate(status['trials']):
            gpu=None
            while gpu is None:
                for candidate in range(8):
                    try: require_idle_gpus(str(candidate)); gpu=candidate; break
                    except ValueError: pass
                if gpu is None:time.sleep(60)
            require_idle_gpus(str(gpu))
            status.update(state='running',active_trial=i); item.update(state='running',gpu=gpu);save(path,status)
            with (REPORT/f'{i}_training.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',__file__,'--trial',str(i),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
                item['pid']=child.pid;save(path,status)
                code=child.wait()
            if code:raise RuntimeError(f'Trial {i} exited {code}; stop without retry')
            item['result']=json.loads((REPORT/f'{i}_result.json').read_text());item['state']='accepted_validation_only';save(path,status)
        winners={}
        for model,base in status['baselines'].items():
            choices=[dict(candidate='existing_baseline',metrics=base['validation'])]
            choices += [t['result'] for t in status['trials'] if t['model']==model]
            winners[model]=min(choices,key=lambda r:r['metrics']['rmse'])
        status.update(state='completed_validation_tuning',selected_by_validation=winners,active_trial=None)
        with (ROOT/'docs/EXPERIMENT_STATUS.md').open('a') as f:
            f.write('\n有限baseline调参六组完成，验证选参及验收见reports/Baseline_limited_tuning_in_domain_seed42_20261015/runs.json；未做测试，正式主表不改。\n')
    except Exception as error:
        status.update(state='failed_stopped',error=repr(error));raise
    finally:save(path,status)


if __name__=='__main__':
    os.chdir(ROOT)
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--trial',type=int);parser.add_argument('--gpu',type=int)
    options=parser.parse_args()
    if options.prepare:prepare()
    elif options.trial is not None:trial(options.trial,options.gpu)
    else:coordinate()
