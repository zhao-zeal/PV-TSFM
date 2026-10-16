"""Repeat only the frozen Full versus seven-variable deletion comparison."""
import json
import os
from pathlib import Path

import run_site0_a01_covariate_ablation as base


def main():
    original = base.ROOT / 'reports/A01_site0_covariate_ablation_seed42_20261014'
    report = base.ROOT / 'reports/A01_site0_static_multiseed_20261014'
    report.mkdir(exist_ok=False)
    selection = json.loads((original / 'stage2_decision.json').read_text())
    fixed = [('full', list(range(15))), (selection['selected_subset'], selection['selected_indices'])]
    status = dict(state='running', coordinator_pid=os.getpid(), created_at=base.now(),
                  seeds=[42, 2026, 2027], reused_seed42=str(original / 'summary.json'),
                  fixed_subsets=dict(fixed), selection_policy='Frozen before new seeds; no reselection',
                  scope='Four new training runs; one idle GPU at a time, two CPU cores, workers0',
                  protocol='site0_v1', model='A01 Chronos-2 + B131 + Linear',
                  training='Original A01 100/10; checkpoint validation MAE; comparison RMSE', runs=[])
    base.save(report / 'runs.json', status)
    stamp = base.datetime.now().strftime('%Y%m%dT%H%M%S')
    try:
        for seed in (2026, 2027):
            base.SEED = seed
            directory = report / f'seed{seed}'
            directory.mkdir()
            stage = dict(state='running', seed=seed, active=[], tasks=[])
            for name, indices in fixed:
                train = base.task(name, indices, stamp)
                stage['tasks'].append(train)
                base.run_group(directory, [train], stage)
                result = base.load_result(directory, train)
                test = base.task(name, indices, stamp, kind='test', checkpoint=result['checkpoint'])
                stage['tasks'].append(test)
                base.run_group(directory, [test], stage)
                status['runs'].append(dict(seed=seed, subset=name, train_result=train['result'],
                                          test_result=test['result']))
                base.save(report / 'runs.json', status)
            stage.update(state='completed_artifact_acceptance_passed', completed_at=base.now())
            base.save(directory / 'runs.json', stage)
        summaries = []
        for seed, directory in [(42, original), (2026, report / 'seed2026'), (2027, report / 'seed2027')]:
            metrics = [json.loads((directory / f'{name}_test_result.json').read_text())['metrics'] for name, _ in fixed]
            summaries.append(dict(seed=seed, full=metrics[0], selected=metrics[1],
                differences={key: dict(absolute=metrics[1][key]-metrics[0][key],
                    relative_percent=100*(metrics[1][key]/metrics[0][key]-1)) for key in ('rmse', 'mae', 'r2')}))
        base.save(report / 'summary.json', dict(results=summaries,
            interpretation='Paired training seeds on the same test set; not independent test periods',
            dynamic_design='Pending review of multiseed and time-block uncertainty; not automatically started'))
        status.update(state='completed_artifact_acceptance_passed', completed_at=base.now())
    except Exception as error:
        status.update(state='failed_stopped', error=str(error))
        raise
    finally:
        base.save(report / 'runs.json', status)


if __name__ == '__main__':
    main()
