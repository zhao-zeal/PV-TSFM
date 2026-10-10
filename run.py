"""TSLib-style forecasting entry; configuration inspection never starts a run."""

import argparse
import json
import logging
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
MODEL_ALIASES = {'Chronos-2': 'Chronos2', 'Cross-Unet': 'CrossUnet',
                 'Time-LLM': 'TimeLLM', 'Time-VLM': 'TimeVLM', 'Dlinear': 'DLinear'}


def build_parser():
    parser = argparse.ArgumentParser(description='PV-TSFM: unified MMSP forecasting')
    parser.add_argument('--task_name', default='long_term_forecast', choices=['long_term_forecast'])
    parser.add_argument('--is_training', type=int, choices=[0, 1], help='1: train then test; 0: test only')
    parser.add_argument('--model', required=True, choices=[
        'DLinear', 'PatchTST', 'FusionSF', 'CrossUnet', 'Cross-Unet', 'TimeXer',
        'Chronos2', 'Chronos-2', 'TimesFM3', 'ChronosX', 'TimeLLM', 'Time-LLM',
        'TimeVLM', 'Time-VLM', 'Dlinear', 'Ours'])
    parser.add_argument('--baseline_config', choices=['official', 'legacy_fusionsf'], default='official')
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
    parser.add_argument('--eval_batch_size', type=int)
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
    parser.add_argument('--moving_avg', type=int)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--fc_dropout', type=float)
    parser.add_argument('--head_dropout', type=float)
    parser.add_argument('--padding_patch', default='end', choices=['end'])
    parser.add_argument('--revin', type=int, default=1, choices=[0, 1])
    parser.add_argument('--affine', type=int, default=0, choices=[0, 1])
    parser.add_argument('--subtract_last', type=int, default=0, choices=[0, 1])
    parser.add_argument('--decomposition', type=int, default=0, choices=[0, 1])
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--lradj', choices=['type1', 'type3', 'type1_onecycle_init',
                                         'fusion_cosine', 'linear', 'plateau'])
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'])
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--covariate_injection', choices=['IIB', 'OIB', 'IIB+OIB'], default='IIB+OIB')
    parser.add_argument('--injection_hidden_dim', type=int, default=256)
    parser.add_argument('--injection_layers', type=int, default=1)
    parser.add_argument('--llm_model', choices=['LLAMA', 'GPT2', 'BERT'], default='LLAMA')
    parser.add_argument('--llm_layers', type=int)
    parser.add_argument('--llm_path', help='Project-local Llama/GPT-2/BERT directory')
    parser.add_argument('--vlm_path', help='Project-local CLIP directory')
    parser.add_argument('--preset', choices=['script', 'experiment'], default='script')
    parser.add_argument('--fusion_modalities', type=int, choices=[2, 3], default=3)
    parser.add_argument('--guide_channels', type=int, choices=[15, 17], default=17,
                        help='FusionSF official NWP input: rounded lat/lon + 15 weather variables')
    parser.add_argument('--model_path', help='Local pretrained checkpoint directory')
    parser.add_argument('--nwp_mode', choices=['none', 'history', 'future', 'history_future'], default='none')
    parser.add_argument('--print_config', action='store_true')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = MODEL_ALIASES.get(args.model, args.model)
    sources = json.loads((ROOT / 'models/baseline_sources.json').read_text())
    args.official_source = sources.get(args.model)
    if args.model == 'Ours':
        parser.error('Ours 为待设计方法，当前没有实现。')
    from models.official_presets import PRESETS, LEGACY
    defaults = {
        'seed': 42, 'batch_size': 32, 'train_epochs': 10, 'patience': 3,
        'learning_rate': 0.0001, 'd_model': 512, 'd_ff': 512, 'e_layers': 3,
        'n_heads': 8, 'dropout': 0.05, 'factor': 5, 'patch_len': 16,
        'moving_avg': 25, 'fc_dropout': 0.05, 'head_dropout': 0.,
        'precision': 'fp32', 'optimizer': 'adam', 'lradj': 'type1',
        'monitor': 'mse', 'drop_last': False, 'max_steps': 5000, 'eval_steps': 100,
        'gradient_accumulation_steps': 1,
    }
    defaults.update(PRESETS.get(args.model, {}))
    if args.baseline_config == 'legacy_fusionsf':
        if args.model not in LEGACY:
            parser.error('legacy_fusionsf is only for archived DLinear/PatchTST runs')
        defaults.update(LEGACY[args.model])
        defaults.update(optimizer='adamw_fusionsf', lradj='plateau', monitor='mae',
                        drop_last=False, eval_batch_size=64)
    defaults.setdefault('eval_batch_size', defaults['batch_size'])
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    args.enc_in = args.dec_in = args.c_out = 1
    args.llm_dim = 4096 if args.llm_model == 'LLAMA' else 768
    args.llm_layers = args.llm_layers or (32 if args.llm_model == 'LLAMA' else 12)
    llm_directory = {'LLAMA': 'llama-7b', 'GPT2': 'gpt2', 'BERT': 'bert-base-uncased'}[args.llm_model]
    args.llm_path = args.llm_path or str(ROOT / 'pretrained' / llm_directory)
    args.vlm_path = args.vlm_path or str(ROOT / 'pretrained/clip-vit-base-patch32')
    args.prompt_domain = 1
    args.content = 'MMSP hourly photovoltaic power generation at solar power plants.'
    args.vlm_type = 'clip'
    args.image_size, args.periodicity, args.norm_const = 56, 24, 0.4
    args.three_channel_image, args.finetune_vlm = True, False
    args.learnable_image, args.save_images = True, False
    args.patch_memory_size, args.top_k, args.padding = 100, 5, args.stride
    args.chronosx_drop_prob = 0.2
    args.max_grad_norm = 1.0 if args.model == 'ChronosX' else None
    args.effective_precision = 'fp32' if args.precision == 'fp16' and not args.use_gpu else args.precision
    if args.model == 'ChronosX':
        args.nwp_mode = 'history_future'
    if args.model == 'DLinear' and args.baseline_config == 'official' and args.moving_avg != 25:
        parser.error('Official DLinear fixes its decomposition kernel to 25')
    required_modalities = 'power'
    if args.model == 'FusionSF':
        required_modalities = 'all' if args.fusion_modalities == 3 else 'satellite'
    elif args.model in ('CrossUnet', 'TimeXer', 'ChronosX') or (args.model == 'Chronos2' and args.nwp_mode != 'none'):
        required_modalities = 'power_nwp'
    args.modalities = args.modalities or required_modalities
    if (required_modalities == 'all' and args.modalities != 'all') or (
        required_modalities == 'satellite' and args.modalities not in ('satellite', 'all')) or (
        required_modalities == 'power_nwp' and args.modalities not in ('power_nwp', 'all')):
        parser.error(f'{args.model} requires {required_modalities} inputs')
    args.model_path = args.model_path or str(ROOT / 'pretrained' / (
        'timesfm-3.0-pytorch' if args.model == 'TimesFM3' else
        'chronos-t5-small' if args.model == 'ChronosX' else 'chronos-2'))
    variant = ''
    if args.model == 'TimeLLM':
        variant = f'_{args.llm_model}_layers{args.llm_layers}'
    elif args.model == 'FusionSF':
        variant = f'_{args.preset}{args.fusion_modalities}_guide{args.guide_channels}'
    elif args.model == 'ChronosX':
        variant = f'_{args.covariate_injection}'
    args.model_id = args.model_id or f'{args.model}_{args.baseline_config}{variant}_seed{args.seed}_sl{args.seq_len}_pl{args.pred_len}_{args.modalities}_{args.nwp_mode}'
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
