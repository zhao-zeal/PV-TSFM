import pandas as pd
import pytest

from pv_tsfm.training.config import TrainingConfig
from pv_tsfm.training.runtime import validate_run_gpu_policy
from pv_tsfm.training.sampling import build_window_sampling_manifest


def test_frozen_training_configuration():
    config = TrainingConfig("E4_shared_lora_cpt", 11)
    assert config.learning_rate == 1e-5
    assert config.microbatch * config.gradient_accumulation == config.effective_batch == 128
    assert TrainingConfig("E5_full_cpt", 22).learning_rate == 1e-6
    with pytest.raises(ValueError):
        TrainingConfig("E4_shared_lora_cpt", 11, effective_batch=256)
    reduced = TrainingConfig("E5_full_cpt", 11, microbatch=8, gradient_accumulation=16)
    assert reduced.effective_batch == 128
    with pytest.raises(ValueError, match="no larger than"):
        TrainingConfig("E5_full_cpt", 11, microbatch=64, gradient_accumulation=2)
    with pytest.raises(ValueError, match="must equal"):
        TrainingConfig("E5_full_cpt", 11, microbatch=8, gradient_accumulation=8)


def test_e4_e5_same_seed_sampling_manifest_is_identical():
    windows = pd.DataFrame({
        "original_source_id": ["a", "a", "b", "b"],
        "physical_location_id": ["l1", "l1", "l2", "l3"],
        "physical_site_id": ["s1", "s1", "s2", "s3"],
        "window_id": ["w1", "w2", "w3", "w4"],
    })
    e4 = build_window_sampling_manifest(windows, seed=11, optimizer_steps=2, effective_batch=4)
    e5 = build_window_sampling_manifest(windows, seed=11, optimizer_steps=2, effective_batch=4)
    pd.testing.assert_frame_equal(e4, e5)


def test_gpu_policy_rejects_third_gpu_and_unjustified_two_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    with pytest.raises(RuntimeError, match="only physical GPUs"):
        validate_run_gpu_policy(method="E4_shared_lora_cpt", requested_gpu_count=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="reserved for Full CPT"):
        validate_run_gpu_policy(method="E4_shared_lora_cpt", requested_gpu_count=2)
