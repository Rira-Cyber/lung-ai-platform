import pytest

from backend.datasets.dataset_splitter import PatientSplitter


def test_split_is_reproducible() -> None:
    patient_ids = [f"LIDC-IDRI-{index:04d}" for index in range(20)]

    first_split = PatientSplitter(
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    ).split(patient_ids)

    second_split = PatientSplitter(
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    ).split(patient_ids)

    assert first_split == second_split


def test_split_contains_all_patients() -> None:
    patient_ids = [f"LIDC-IDRI-{index:04d}" for index in range(20)]

    split = PatientSplitter(
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    ).split(patient_ids)

    combined_ids = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)

    assert combined_ids == set(patient_ids)


def test_split_has_no_patient_overlap() -> None:
    patient_ids = [f"LIDC-IDRI-{index:04d}" for index in range(20)]

    split = PatientSplitter(
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    ).split(patient_ids)

    train_ids = set(split.train_ids)
    val_ids = set(split.val_ids)
    test_ids = set(split.test_ids)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_zero_test_ratio_returns_empty_test_split() -> None:
    patient_ids = [f"LIDC-IDRI-{index:04d}" for index in range(10)]

    split = PatientSplitter(
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
    ).split(patient_ids)

    assert split.test_ids == ()
    assert len(split.val_ids) == 2
    assert len(split.train_ids) == 8


def test_non_zero_ratio_allocates_at_least_one_patient() -> None:
    patient_ids = [
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
        "LIDC-IDRI-0003",
    ]

    split = PatientSplitter(
        val_ratio=0.01,
        seed=42,
    ).split(patient_ids)

    assert len(split.val_ids) == 1
    assert len(split.train_ids) == 2


def test_empty_patient_ids_raise_error() -> None:
    splitter = PatientSplitter(
        val_ratio=0.2,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        splitter.split([])


def test_duplicate_patient_ids_raise_error() -> None:
    splitter = PatientSplitter(
        val_ratio=0.2,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        splitter.split(
            [
                "LIDC-IDRI-0001",
                "LIDC-IDRI-0001",
                "LIDC-IDRI-0002",
            ]
        )


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio"),
    [
        (-0.1, 0.0),
        (1.0, 0.0),
        (0.2, -0.1),
        (0.2, 1.0),
    ],
)
def test_invalid_individual_ratios_raise_error(
    val_ratio: float,
    test_ratio: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be in the range",
    ):
        PatientSplitter(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio"),
    [
        (0.5, 0.5),
        (0.8, 0.2),
        (0.9, 0.2),
    ],
)
def test_invalid_combined_ratios_raise_error(
    val_ratio: float,
    test_ratio: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be less than 1.0",
    ):
        PatientSplitter(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )


def test_split_that_leaves_no_training_patients_raises_error() -> None:
    splitter = PatientSplitter(
        val_ratio=0.75,
        test_ratio=0.0,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="leave no training patients",
    ):
        splitter.split(
            [
                "LIDC-IDRI-0001",
                "LIDC-IDRI-0002",
            ]
        )
