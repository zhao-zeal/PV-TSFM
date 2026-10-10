"""Run one explicitly selected FusionSF experiment with BasicTS."""

import argparse
from pvtsfm.resources import require_idle_gpus

from pvtsfm.launcher import PVLauncher as BasicTSLauncher




def main():
    from .fusionsf_config import build_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fusionSF_2modal', 'fusionSF_3modal'], default='fusionSF_3modal')
    parser.add_argument('--preset', choices=['experiment', 'script'])
    parser.add_argument('--protocol', choices=['in_domain', 'zeroshot_v1', 'paper_main_v1', 'fixed_v1', 'paper_main'], required=True)
    parser.add_argument('--gpus', required=True, help='Physical GPU IDs; check nvidia-smi before launch')
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--checkpoint', help='Evaluate an existing BasicTS FusionSF checkpoint')
    args = parser.parse_args()
    preset = args.preset or ('script' if args.model == 'fusionSF_3modal' else 'experiment')
    cfg = build_config(args.model, preset, args.gpus, args.data_dir, args.output_dir,
                       args.seed, args.num_epochs, args.batch_size, args.num_workers, args.protocol)
    if args.print_config:
        print(cfg)
        return
    require_idle_gpus(args.gpus)
    if args.checkpoint:
        BasicTSLauncher.launch_evaluation(cfg, args.checkpoint, gpus=args.gpus)
    else:
        BasicTSLauncher.launch_training(cfg)


if __name__ == '__main__':
    main()
