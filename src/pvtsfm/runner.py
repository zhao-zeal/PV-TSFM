"""BasicTS runner integration for plateau scheduling and standard NumPy exports."""
import os
import time
from typing import Dict

import numpy as np
import torch
from easytorch.utils import master_only
from basicts.runners import BasicTSRunner


class PVRunner(BasicTSRunner):
    def on_epoch_end(self, epoch: int, step: int) -> None:
        """
        Callback at the end of each epoch to handle validation and testing.

        Args:
            epoch (int): The current epoch number.
        """

        if self.lr_scheduler is not None:
            self.update_meter("train/lr", self.optimizer.param_groups[0]["lr"])
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()
        self.update_meter("train/time", time.time() - self.epoch_start_time)

        # print training meters
        self.print_meters("train")
        # plot training meters to TensorBoard
        self.plt_meters("train", epoch)
        # perform validation if configured
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor = self.cfg.get('lr_scheduler_monitor', self.target_metric)
                self.lr_scheduler.step(self.meter_pool.get_value(f'val/{monitor}'))
        # perform testing if configured
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self._test(train_epoch=epoch)
        # save the model checkpoint
        self._save_model(epoch)
        # reset epoch meters
        self.reset_meters()

        # estimate training finish time
        if not self.should_training_stop and self.epoch < self.num_epochs:
            expected_end_time = self.train_time_predictor.get_expected_end_time(step)
            self.logger.info("The estimated training finish time is {}".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expected_end_time))))

    @master_only
    def _save_results(self, batch_idx: int, batch_data: Dict[str, torch.Tensor]) -> None:

        """
        Save the test results to disk.

        Args:
            batch_idx (int): The index of the current batch.
            batch_data (Dict[np.ndarray]): The test results:{
                "inputs": np.ndarray,
                "prediction": np.ndarray,
                "targets": np.ndarray,
            }
        """

        inputs = batch_data["inputs"].detach().cpu().numpy()
        prediction = batch_data["prediction"].detach().cpu().numpy()
        targets = batch_data["targets"].detach().cpu().numpy()

        total_samples = len(self.test_data_loader.dataset)

        save_dir = os.path.join(self.ckpt_save_dir, "test_results")
        os.makedirs(save_dir, exist_ok=True)
        inputs_path = os.path.join(save_dir, "inputs.npy")
        pred_path = os.path.join(save_dir, "prediction.npy")
        targets_path = os.path.join(save_dir, "targets.npy")

        # create memmap files
        if batch_idx == 0:
            self._inputs_memmap = np.lib.format.open_memmap(inputs_path, dtype=inputs.dtype, mode="w+",
                                    shape=(total_samples, *inputs.shape[1:]))
            self._prediction_memmap = np.lib.format.open_memmap(pred_path, dtype=prediction.dtype, mode="w+",
                                    shape=(total_samples, *prediction.shape[1:]))
            self._targets_memmap = np.lib.format.open_memmap(targets_path, dtype=targets.dtype, mode="w+",
                                shape=(total_samples, *targets.shape[1:]))

        start = batch_idx * self.cfg.test_batch_size
        end = start + inputs.shape[0]

        self._inputs_memmap[start:end] = inputs
        self._prediction_memmap[start:end] = prediction
        self._targets_memmap[start:end] = targets
        if end == total_samples:
            self._inputs_memmap.flush()
            self._prediction_memmap.flush()
            self._targets_memmap.flush()

