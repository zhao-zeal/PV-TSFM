"""Dispatch a single explicitly selected experiment."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['fusionsf', 'baseline', 'crossunet', 'timexer', 'zeroshot'])
    args = parser.parse_args(sys.argv[1:2])
    modules = {'baseline': 'baselines'}
    sys.argv = [f'{sys.argv[0]} {args.experiment}', *sys.argv[2:]]
    module = importlib.import_module(f'pvtsfm.experiments.{modules.get(args.experiment, args.experiment)}')
    module.main()
