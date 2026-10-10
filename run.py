"""TSLib-style forecasting entry; configuration inspection never starts a run."""

import argparse
import json
import logging
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
MODEL_ALIASES = {'Chronos-2': 'Chronos2', 'Cross-Unet': 'CrossUnet'}


def build_parser():
    parser = argparse.ArgumentParser(description='PV-TSFM: unified MMSP forecasting')
    parser.add_argument('--task_name', default='long_term_forecast', choices=['long_term_forecast'])
    parser.add_argument('--is_training', type=int, choices=[0, 1], help='1: train then test; 0: test only')
    parser.add_argument('--model', required=True, choices=[
        'DLinear', 'PatchTST', 'FusionSF', 'CrossUnet', 'Cross-Unet', 'TimeXer',
        'Chronos2', 'Chronos-2', 'TimesFM3', 'ChronosX', 'Ours'])
    parser.add_argument('--model_id', help='Explicit run name; choose a new name for a separate run')
    parser.add_argument('--data', default='MMSP', choices=['MMSP'])
    parser.add_argument('--root_path', default=str(ROOT / 'datasets/MMSP/data'))
    parser.add_argument('--protocol', required=True, choices=['in_domain', 'zeroshot_v1'])
    parser.add_argument('--modalities', choices=['power', 'power_nwp', 'satellite', 'all'])
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--label_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--train_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoints', default=str(ROOT / 'checkpoints'))
    parser.add_argument('--output_dir', default=str(ROOT / 'outputs'))
    parser.add_argument('--checkpoint', help='Existing BasicTS or project model checkpoint for test-only mode')
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=0)
    parser.add_argument('--gpu', type=int, default=0, help='Explicit idle GPU index')
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--d_ff', type=int)
    parser.add_argument('--e_layers', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--factor', type=int)
    parser.add_argument('--patch_len', type=int)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seg_len', type=int, default=12)
    parser.add_argument('--moving_avg', type=int, default=13)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--preset', choices=['script', 'experiment'], default='script')
    parser.add_argument('--fusion_modalities', type=int, choices=[2, 3], default=3)
    parser.add_argument('--model_path', help='Local pretrained checkpoint directory')
    parser.add_argument('--nwp_mode', choices=['none', 'history', 'future', 'history_future'], default='none')
    parser.add_argument('--print_config', action='store_true')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = MODEL_ALIASES.get(args.model, args.model)
    if args.model in ('ChronosX', 'Ours'):
        parser.error('ChronosX 尚未接入；Ours 为待设计方法，当前没有实现。')
    defaults = {
        'seed': 2021 if args.model in ('CrossUnet', 'TimeXer') else 42,
        'batch_size': {'DLinear': 64, 'PatchTST': 16, 'FusionSF': 16, 'CrossUnet': 128}.get(args.model, 32),
        'train_epochs': 50 if args.model in ('DLinear', 'PatchTST') else 100,
        'patience': 10 if args.model in ('DLinear', 'CrossUnet', 'TimeXer') else 20,
        'learning_rate': {'DLinear': 0.0016, 'PatchTST': 0.0002, 'FusionSF': 0.0016}.get(args.model, 0.0001),
        'd_model': 256 if args.model == 'CrossUnet' else 512,
        'd_ff': 2048 if args.model == 'PatchTST' else 512,
        'e_layers': 2 if args.model == 'TimeXer' else 3,
        'n_heads': 4 if args.model == 'CrossUnet' else 8,
        'dropout': 0.1 if args.model == 'TimeXer' else 0.05,
        'factor': {'CrossUnet': 10, 'TimeXer': 3}.get(args.model, 5),
        'patch_len': 12 if args.model == 'TimeXer' else 16,
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    required_modalities = 'power'
    if args.model == 'FusionSF':
        required_modalities = 'all' if args.fusion_modalities == 3 else 'satellite'
    elif args.model in ('CrossUnet', 'TimeXer') or (args.model == 'Chronos2' and args.nwp_mode != 'none'):
        required_modalities = 'power_nwp'
    args.modalities = args.modalities or required_modalities
    if (required_modalities == 'all' and args.modalities != 'all') or (
        required_modalities == 'satellite' and args.modalities not in ('satellite', 'all')) or (
        required_modalities == 'power_nwp' and args.modalities not in ('power_nwp', 'all')):
        parser.error(f'{args.model} requires {required_modalities} inputs')
    args.model_path = args.model_path or str(ROOT / 'pretrained' / (
        'timesfm-3.0-pytorch' if args.model == 'TimesFM3' else 'chronos-2'))
    args.model_id = args.model_id or f'{args.model}_seed{args.seed}_sl{args.seq_len}_pl{args.pred_len}_{args.modalities}_{args.nwp_mode}'
    if Path(args.model_id).name != args.model_id:
        parser.error('model_id must be a directory name, without path separators')
    if not args.print_config:
        if args.is_training is None:
            parser.error('Select --is_training 0 or 1 explicitly')
        frozen = args.model in ('Chronos2', 'TimesFM3')
        if args.is_training and frozen:
            parser.error('Frozen models support test-only mode: --is_training 0')
        if not args.is_training and not frozen and not args.checkpoint:
            parser.error('Test-only mode for trainable models requires --checkpoint')
    if not args.use_gpu and args.num_workers != 0:
        parser.error('CPU runs require --num_workers 0')
    return args


def configure_resources(args):
    cores = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cores)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(len(cores))
    if args.use_gpu:
        from utils.resources import require_idle_gpus
        require_idle_gpus(str(args.gpu))
    import torch
    torch.set_num_threads(len(cores))
    torch.set_num_interop_threads(1)
    # Match the accepted BasicTS runs' numerical backend settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(args.use_gpu)


def main(argv=None):
    args = parse_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2))
        return
    configure_resources(args)
    import numpy as np
    import torch
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    directory = Path(args.output_dir) / args.protocol / args.model_id
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(), logging.FileHandler(directory / 'run.log')],
        format='%(asctime)s %(message)s')
    experiment = Exp_Long_Term_Forecast(args)
    if args.is_training:
        experiment.train(args.model_id)
    experiment.test(args.model_id, test=not args.is_training)


if __name__ == '__main__':
    main()
