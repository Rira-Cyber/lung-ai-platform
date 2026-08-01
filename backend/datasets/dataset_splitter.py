from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class PatientSplit:
    """
    Immutable patient-level dataset split.

    Ensures that train, validation, and test sets contain
    mutually exclusive patient IDs.
    """

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        train = set(self.train_ids)
        val = set(self.val_ids)
        test = set(self.test_ids)

        if train & val:
            raise ValueError("Train and validation patient IDs overlap.")

        if train & test:
            raise ValueError("Train and test patient IDs overlap.")

        if val & test:
            raise ValueError("Validation and test patient IDs overlap.")


class PatientSplitter:
    """
    Creates deterministic patient-level dataset splits.

    Splitting is performed before dataset creation to guarantee
    that no patient appears in more than one subset.
    """

    def __init__(
        self,
        val_ratio: float,
        test_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        self._validate_ratios(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        patient_ids: Sequence[str],
    ) -> PatientSplit:
        """
        Split patient IDs into train, validation and test subsets.
        """

        unique_ids = list(dict.fromkeys(patient_ids))

        if len(unique_ids) != len(patient_ids):
            raise ValueError("patient_ids contains duplicate values.")

        if not unique_ids:
            raise ValueError("patient_ids cannot be empty.")

        shuffled = unique_ids.copy()
        random.Random(self.seed).shuffle(shuffled)

        total = len(shuffled)

        test_count = self._split_size(
            total,
            self.test_ratio,
        )

        val_count = self._split_size(
            total,
            self.val_ratio,
        )

        if test_count + val_count >= total:
            raise ValueError("Validation and test splits leave no training patients.")

        test_ids = tuple(shuffled[:test_count])
        val_ids = tuple(shuffled[test_count : test_count + val_count])
        train_ids = tuple(shuffled[test_count + val_count :])

        return PatientSplit(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

    @staticmethod
    def _split_size(
        total: int,
        ratio: float,
    ) -> int:
        """
        Compute the number of samples assigned to a split.
        """

        if ratio == 0.0:
            return 0

        return max(1, round(total * ratio))

    @staticmethod
    def _validate_ratios(
        val_ratio: float,
        test_ratio: float,
    ) -> None:
        """
        Validate split ratios.
        """

        for name, ratio in (
            ("val_ratio", val_ratio),
            ("test_ratio", test_ratio),
        ):
            if not 0.0 <= ratio < 1.0:
                raise ValueError(f"{name} must be in the range [0.0, 1.0).")

        if val_ratio + test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1.0.")
