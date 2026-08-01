from __future__ import annotations

import numpy as np
import pytest

from backend.datasets.validation_metadata import (
    ValidationMetadataGenerator,
)


class FakeLIDCProcessor:
    """
    Lightweight processor for metadata generator tests.
    """

    def __init__(
        self,
        dataset_path,
    ) -> None:
        self.dataset_path = dataset_path
        self.patient_id: str | None = None

    def load_patient(
        self,
        patient_id: str,
    ) -> None:
        self.patient_id = patient_id

    def hu_volume(
        self,
    ) -> np.ndarray:
        return np.ones(
            (16, 16, 16),
            dtype=np.float32,
        )

    def lung_mask(
        self,
    ) -> np.ndarray:
        return np.ones(
            (16, 16, 16),
            dtype=np.uint8,
        )

    def nodule_mask(
        self,
    ) -> np.ndarray:
        mask = np.zeros(
            (16, 16, 16),
            dtype=np.uint8,
        )

        if self.patient_id == "positive-patient":
            mask[
                7:9,
                7:9,
                7:9,
            ] = 1

        return mask


def create_generator(
    tmp_path,
    *,
    patches_per_patient: int = 2,
    seed: int = 42,
) -> ValidationMetadataGenerator:
    return ValidationMetadataGenerator(
        dataset_path=tmp_path,
        patch_size=(8, 8, 8),
        positive_ratio=1.0,
        patches_per_patient=patches_per_patient,
        seed=seed,
        processor_factory=FakeLIDCProcessor,
    )


def test_generator_creates_expected_patch_count(
    tmp_path,
) -> None:
    generator = create_generator(
        tmp_path,
        patches_per_patient=3,
    )

    metadata = generator.generate(
        [
            "positive-patient",
            "negative-patient",
        ]
    )

    assert len(metadata.patches) == 6


def test_generator_assigns_patch_indexes_per_patient(
    tmp_path,
) -> None:
    generator = create_generator(
        tmp_path,
        patches_per_patient=3,
    )

    metadata = generator.generate(["positive-patient"])

    assert [patch.patch_index for patch in metadata.patches] == [
        0,
        1,
        2,
    ]


def test_generator_preserves_patient_ids(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata = generator.generate(
        [
            "positive-patient",
            "negative-patient",
        ]
    )

    assert metadata.patient_ids == (
        "negative-patient",
        "positive-patient",
    )


def test_patient_without_nodule_is_always_negative(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata = generator.generate(["negative-patient"])

    assert all(patch.is_positive is False for patch in metadata.patches)


def test_positive_patient_uses_positive_sampling(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata = generator.generate(["positive-patient"])

    assert all(patch.is_positive is True for patch in metadata.patches)


def test_generator_is_reproducible_with_same_seed(
    tmp_path,
) -> None:
    first_generator = create_generator(
        tmp_path,
        seed=42,
    )

    second_generator = create_generator(
        tmp_path,
        seed=42,
    )

    patient_ids = [
        "positive-patient",
        "negative-patient",
    ]

    first = first_generator.generate(patient_ids)

    second = second_generator.generate(patient_ids)

    assert first == second


def test_generate_rejects_empty_patient_ids(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        generator.generate([])


def test_generate_rejects_duplicate_patient_ids(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        generator.generate(
            [
                "positive-patient",
                "positive-patient",
            ]
        )


def test_generate_or_load_saves_metadata(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata_path = tmp_path / "validation_metadata.json"

    metadata = generator.generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    assert metadata_path.exists()
    assert len(metadata.patches) == 2


def test_generate_or_load_reuses_existing_metadata(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata_path = tmp_path / "validation_metadata.json"

    first = generator.generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    second = generator.generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    assert second == first


def test_existing_metadata_rejects_different_patient_split(
    tmp_path,
) -> None:
    generator = create_generator(tmp_path)

    metadata_path = tmp_path / "validation_metadata.json"

    generator.generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    with pytest.raises(
        ValueError,
        match="patient split",
    ):
        generator.generate_or_load(
            patient_ids=["negative-patient"],
            metadata_path=metadata_path,
        )


def test_existing_metadata_rejects_different_seed(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "validation_metadata.json"

    create_generator(
        tmp_path,
        seed=42,
    ).generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    with pytest.raises(
        ValueError,
        match="different seed",
    ):
        create_generator(
            tmp_path,
            seed=7,
        ).generate_or_load(
            patient_ids=["positive-patient"],
            metadata_path=metadata_path,
        )


def test_existing_metadata_rejects_different_patch_count(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "validation_metadata.json"

    create_generator(
        tmp_path,
        patches_per_patient=2,
    ).generate_or_load(
        patient_ids=["positive-patient"],
        metadata_path=metadata_path,
    )

    with pytest.raises(
        ValueError,
        match="patches_per_patient",
    ):
        create_generator(
            tmp_path,
            patches_per_patient=3,
        ).generate_or_load(
            patient_ids=["positive-patient"],
            metadata_path=metadata_path,
        )
