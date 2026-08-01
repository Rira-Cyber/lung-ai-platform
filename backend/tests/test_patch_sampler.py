import numpy as np
import pytest
from backend.preprocessing.patch_sampler import PatchSampler


def test_extract_at_center_is_deterministic() -> None:
    image = np.arange(
        1000,
        dtype=np.float32,
    ).reshape(
        10,
        10,
        10,
    )

    mask = np.zeros(
        (
            10,
            10,
            10,
        ),
        dtype=np.uint8,
    )

    mask[
        4:6,
        4:6,
        4:6,
    ] = 1

    sampler = PatchSampler(
        patch_size=(
            4,
            4,
            4,
        ),
    )

    first = sampler.extract_at_center(
        image=image,
        mask=mask,
        center=(5, 5, 5),
    )

    second = sampler.extract_at_center(
        image=image,
        mask=mask,
        center=(5, 5, 5),
    )

    np.testing.assert_array_equal(
        first["image"],
        second["image"],
    )

    np.testing.assert_array_equal(
        first["mask"],
        second["mask"],
    )


def test_patch_size_is_correct() -> None:
    image = np.zeros(
        (
            30,
            30,
            30,
        ),
        dtype=np.float32,
    )

    mask = np.zeros_like(image)

    sampler = PatchSampler(
        patch_size=(
            12,
            12,
            12,
        ),
    )

    sample = sampler.extract_at_center(
        image=image,
        mask=mask,
        center=(15, 15, 15),
    )

    assert sample["image"].shape == (
        12,
        12,
        12,
    )

    assert sample["mask"].shape == (
        12,
        12,
        12,
    )


def test_boundary_padding() -> None:
    sampler = PatchSampler(
        patch_size=(
            8,
            8,
            8,
        ),
    )

    image = np.ones(
        (
            5,
            5,
            5,
        ),
        dtype=np.float32,
    )

    mask = np.zeros_like(
        image,
        dtype=np.uint8,
    )

    sample = sampler.extract_at_center(
        image=image,
        mask=mask,
        center=(0, 0, 0),
    )

    assert sample["image"].shape == (
        8,
        8,
        8,
    )

    assert sample["mask"].shape == (
        8,
        8,
        8,
    )


def test_positive_sampling_without_positive_voxel_raises() -> None:
    sampler = PatchSampler()

    mask = np.zeros(
        (
            20,
            20,
            20,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        RuntimeError,
    ):
        sampler._sample_positive_center(
            mask,
        )


def test_force_negative_sampling() -> None:
    image = np.zeros(
        (
            20,
            20,
            20,
        ),
        dtype=np.float32,
    )

    lung = np.ones_like(
        image,
        dtype=np.uint8,
    )

    nodules = np.zeros_like(
        image,
        dtype=np.uint8,
    )

    nodules[
        8:12,
        8:12,
        8:12,
    ] = 1

    sampler = PatchSampler()

    sample = sampler.sample(
        image=image,
        nodule_mask=nodules,
        lung_mask=lung,
        force_negative=True,
    )

    assert sample["is_positive"] is False


def test_sample_batch_size() -> None:
    image = np.zeros(
        (32, 32, 32),
        dtype=np.float32,
    )

    lung = np.ones_like(
        image,
        dtype=np.uint8,
    )

    nodules = np.zeros_like(
        image,
        dtype=np.uint8,
    )

    sampler = PatchSampler(
        positive_ratio=0.0,
        patches_per_volume=5,
    )

    samples = sampler.sample_batch(
        image=image,
        nodule_mask=nodules,
        lung_mask=lung,
    )

    assert len(samples) == 5
    assert all(sample["is_positive"] is False for sample in samples)


def test_center_outside_volume_is_clamped() -> None:
    image = np.ones(
        (16, 16, 16),
        dtype=np.float32,
    )

    mask = np.zeros(
        (16, 16, 16),
        dtype=np.uint8,
    )

    sampler = PatchSampler(
        patch_size=(8, 8, 8),
    )

    sample = sampler.extract_at_center(
        image=image,
        mask=mask,
        center=(25, -10, 40),
    )

    assert tuple(sample["center"]) == (
        15,
        0,
        15,
    )

    assert sample["image"].shape == (
        8,
        8,
        8,
    )

    assert sample["mask"].shape == (
        8,
        8,
        8,
    )
