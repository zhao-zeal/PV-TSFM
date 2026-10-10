"""FusionSF experiment YAML defaults and the original three-modal shell preset."""

from pathlib import Path
from pvtsfm.paths import MMSP_DIR

from torch.optim import AdamW

from basicts.configs import BasicTSForecastingConfig
from pvtsfm.data.mmsp import MMSPForecastingDataset, canonical_protocol
from pvtsfm.models.fusionsf import FusionSF2Modal, FusionSF3Modal, FusionSFConfig, fusionsf_loss
from pvtsfm.models.fusionsf.scheduler import CosineWarmupScheduler
from basicts.runners.callback import EarlyStopping

from .callbacks import MMSPResults, protocol_output_dir


ROOT = Path(__file__).resolve().parents[3]


def build_config(model='fusionSF_3modal', preset='script', gpus=None,
                 data_dir=None, output_dir=None, seed=42, num_epochs=100,
                 batch_size=16, num_workers=None, protocol='in_domain'):
    protocol = canonical_protocol(protocol)
    if num_workers is None:
        num_workers = 0 if protocol == 'zeroshot_v1' else 4
    is_three = model == 'fusionSF_3modal'
    if model not in ('fusionSF_2modal', 'fusionSF_3modal'):
        raise ValueError(f'Unknown FusionSF model: {model}')
    if preset not in ('experiment', 'script') or (preset == 'script' and not is_three):
        raise ValueError('script preset is available only for fusionSF_3modal')
    architecture = FusionSFConfig(mlp_ratio=4 if is_three else 1, guide_channels=17 if protocol == 'paper_main_v1' else 15)
    if preset == 'script':
        architecture.ctx_masking_ratio = 0.99
        architecture.vq_in_ts = True
        architecture.vq_in_ctx = True
    return BasicTSForecastingConfig(
        model=FusionSF3Modal if is_three else FusionSF2Modal,
        model_config=architecture, dataset_name='MMSP',
        dataset_type=MMSPForecastingDataset,
        dataset_params={
            'data_dir': str(Path(data_dir or MMSP_DIR).resolve()),
            'input_len': 24, 'output_len': 24, 'num_sites': 20 if protocol == 'zeroshot_v1' else 10,
            'num_ignored_sites': 10 if protocol == 'zeroshot_v1' else 0,
            'protocol': protocol, 'modality_mode': 'all' if is_three else 'satellite',
            'train_ratio': 0.6, 'valid_ratio': 0.2, 'test_ratio': 0.2,
        },
        input_len=24, output_len=24, gpus=gpus, seed=seed,
        scaler=None, rescale=False, null_val=float('nan'),
        loss=fusionsf_loss, optimizer=AdamW,
        optimizer_params={'lr': 0.0016, 'weight_decay': 0.05, 'betas': [0.9, 0.95]},
        lr_scheduler=CosineWarmupScheduler,
        lr_scheduler_params={'warmup': 5, 'max_iters': num_epochs},
        num_epochs=num_epochs, batch_size=None,
        train_batch_size=batch_size, val_batch_size=batch_size,
        test_batch_size=64 if protocol == 'zeroshot_v1' else batch_size,
        metrics=['MAE', 'RMSE'], target_metric='MAE', best_metric='min',
        callbacks=[EarlyStopping(patience=20 if preset == 'script' else 35), MMSPResults()],
        train_data_num_workers=num_workers, val_data_num_workers=num_workers,
        test_data_num_workers=num_workers, test_interval=num_epochs + 1,
        ddp_find_unused_parameters=True, eval_after_train=True, save_results=True,
        ckpt_save_dir=protocol_output_dir(output_dir, protocol, ROOT / 'checkpoints' / protocol / model / preset / f'seed_{seed}'),
        data_protocol=protocol, fusionsf_preset=preset,
    )
