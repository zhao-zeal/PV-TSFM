import pandas as pd
import pytest

from pv_tsfm.data.manifests import (
    assign_mmsp_group_roles,
    assign_stategrid_roles,
    validate_split_integrity,
)


def test_mmsp_expected_counts_and_stable_hash_order():
    groups = [f"group-{index:02d}" for index in range(88)]
    first = assign_mmsp_group_roles(groups)
    second = assign_mmsp_group_roles(reversed(groups))
    pd.testing.assert_frame_equal(first, second)
    assert first.group_role.value_counts().to_dict() == {"train": 64, "validation": 12, "test": 12}


def test_mmsp_fallback_counts_floor_first_two():
    split = assign_mmsp_group_roles([f"g{index}" for index in range(10)])
    assert split.group_role.value_counts().to_dict() == {"train": 7, "test": 2, "validation": 1}


def test_stategrid_refuses_unverified_group_count():
    with pytest.raises(ValueError, match="exactly eight"):
        assign_stategrid_roles([f"g{index}" for index in range(7)])


def test_duplicate_group_rows_are_rejected():
    split = pd.DataFrame({
        "canonical_physical_location_group_id": ["same", "same"],
        "group_role": ["train", "test"],
    })
    with pytest.raises(ValueError, match="multiple split rows"):
        validate_split_integrity({"site-a": "same", "site-b": "same"}, split)
