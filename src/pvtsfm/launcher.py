"""Launch the project runner using BasicTS and EasyTorch's training machinery."""
from easytorch.device import set_device_type
from easytorch.launcher.dist_wrap import dist_wrap
from easytorch.utils import set_visible_devices

from .runner import PVRunner


def training_func(cfg):
    runner = PVRunner(cfg)
    runner.init_logger(logger_name='PV-TSFM-training', log_file_name='training_log')
    runner.train()


class PVLauncher:
    @staticmethod
    def launch_training(cfg, node_rank=0):
        if node_rank == 0:
            cfg.save()
        set_device_type('gpu' if cfg.gpus else 'cpu')
        if cfg.gpus:
            set_visible_devices(cfg.gpus)
        dist_wrap(training_func, node_num=cfg.get('dist_node_num', 1),
                  device_num=cfg.gpu_num if cfg.gpus else 0, node_rank=node_rank,
                  dist_backend=cfg.get('dist_backend'),
                  init_method=cfg.get('dist_init_method'))(cfg)

    @staticmethod
    def launch_evaluation(cfg, ckpt_path, gpus=None, batch_size=None):
        set_device_type('gpu' if gpus else 'cpu')
        if gpus:
            set_visible_devices(gpus)
        cfg.gpus = gpus
        cfg.gpu_num = len(gpus.split(',')) if gpus else 0
        if batch_size is not None:
            cfg.test_batch_size = batch_size
        cfg.save()
        runner = PVRunner(cfg)
        runner.init_logger(logger_name='PV-TSFM-evaluation', log_file_name='evaluation_log')
        runner.eval(ckpt_path)
