from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.preprocessing.patch_sampler import PatchSampler


VALIDATION_METADATA_VERSION = 1


@dataclass(frozen=True)
class ValidationPatch:
    """
    Metadata describing one fixed validation patch.
    """

    patient_id: str
    patch_index: int
    center: tuple[int, int, int]
    is_positive: bool

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id cannot be empty.")

        if self.patch_index < 0:
            raise ValueError("patch_index cannot be negative.")

        if len(self.center) != 3:
            raise ValueError("center must contain exactly three coordinates.")

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> ValidationPatch:
        """
        Create ValidationPatch from JSON-compatible data.
        """

        required_fields = {
            "patient_id",
            "patch_index",
            "center",
            "is_positive",
        }

        missing_fields = required_fields - data.keys()

        if missing_fields:
            raise ValueError(
                "Validation patch metadata is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        center = tuple(int(value) for value in data["center"])

        return cls(
            patient_id=str(data["patient_id"]),
            patch_index=int(data["patch_index"]),
            center=center,
            is_positive=bool(data["is_positive"]),
        )


@dataclass(frozen=True)
class ValidationMetadata:
    """
    Immutable collection of validation patch metadata.
    """

    version: int
    patch_size: tuple[int, int, int]
    seed: int
    patches: tuple[ValidationPatch, ...]

    def __post_init__(self) -> None:
        if self.version != VALIDATION_METADATA_VERSION:
            raise ValueError(
                f"Unsupported validation metadata version: {self.version}."
            )

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must contain exactly three values.")

        if any(value <= 0 for value in self.patch_size):
            raise ValueError("All patch_size values must be greater than zero.")

        if not self.patches:
            raise ValueError("Validation metadata cannot be empty.")

        identities = [
            (
                patch.patient_id,
                patch.patch_index,
            )
            for patch in self.patches
        ]

        if len(identities) != len(set(identities)):
            raise ValueError("Validation metadata contains duplicate patch identities.")

    @property
    def patient_ids(
        self,
    ) -> tuple[str, ...]:
        """
        Return unique patient IDs represented in metadata.
        """

        return tuple(sorted({patch.patient_id for patch in self.patches}))

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save metadata atomically as JSON.
        """

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": self.version,
            "patch_size": list(self.patch_size),
            "seed": self.seed,
            "patches": [
                {
                    **asdict(patch),
                    "center": list(patch.center),
                }
                for patch in self.patches
            ],
        }

        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        temporary_path.replace(output_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> ValidationMetadata:
        """
        Load and validate metadata from JSON.
        """

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(f"Validation metadata file not found: {input_path}")

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        required_fields = {
            "version",
            "patch_size",
            "seed",
            "patches",
        }

        missing_fields = required_fields - payload.keys()

        if missing_fields:
            raise ValueError(
                "Validation metadata is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        return cls(
            version=int(payload["version"]),
            patch_size=tuple(int(value) for value in payload["patch_size"]),
            seed=int(payload["seed"]),
            patches=tuple(
                ValidationPatch.from_dict(patch_data)
                for patch_data in payload["patches"]
            ),
        )


class ValidationMetadataGenerator:
    """
    Generate deterministic validation patch metadata.

    Patch centers are generated once and persisted. A processor
    factory can be injected for lightweight testing.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        patch_size: tuple[int, int, int],
        positive_ratio: float,
        patches_per_patient: int,
        seed: int,
        processor_factory=None,
    ) -> None:
        if patches_per_patient <= 0:
            raise ValueError("patches_per_patient must be greater than zero.")

        self.dataset_path = Path(dataset_path)
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.patches_per_patient = patches_per_patient
        self.seed = seed
        self.processor_factory = processor_factory

    def _create_processor(self):
        """
        Create the configured medical processor.

        LIDCProcessor is imported lazily so metadata models and tests
        do not require pylidc unless real metadata generation is used.
        """

        if self.processor_factory is not None:
            return self.processor_factory(self.dataset_path)

        from backend.medical.lidc_processor import LIDCProcessor

        return LIDCProcessor(self.dataset_path)

    def generate(
        self,
        patient_ids: Sequence[str],
    ) -> ValidationMetadata:
        """
        Generate fixed patch metadata for validation patients.
        """

        resolved_patient_ids = list(patient_ids)

        if not resolved_patient_ids:
            raise ValueError("Validation patient IDs cannot be empty.")

        if len(resolved_patient_ids) != len(set(resolved_patient_ids)):
            raise ValueError("Validation patient IDs contain duplicates.")

        processor = self._create_processor()

        sampler = PatchSampler(
            patch_size=self.patch_size,
            positive_ratio=self.positive_ratio,
            random_seed=self.seed,
        )

        patches: list[ValidationPatch] = []

        for patient_id in resolved_patient_ids:
            processor.load_patient(patient_id)

            image = processor.hu_volume()
            nodule_mask = processor.nodule_mask()
            lung_mask = processor.lung_mask()

            force_negative = not bool(nodule_mask.any())

            for patch_index in range(self.patches_per_patient):
                sample = sampler.sample(
                    image=image,
                    nodule_mask=nodule_mask,
                    lung_mask=lung_mask,
                    force_negative=force_negative,
                )

                patches.append(
                    ValidationPatch(
                        patient_id=patient_id,
                        patch_index=patch_index,
                        center=tuple(int(value) for value in sample["center"]),
                        is_positive=bool(sample["is_positive"]),
                    )
                )

        return ValidationMetadata(
            version=VALIDATION_METADATA_VERSION,
            patch_size=self.patch_size,
            seed=self.seed,
            patches=tuple(patches),
        )

    def generate_or_load(
        self,
        patient_ids: Sequence[str],
        metadata_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ValidationMetadata:
        """
        Load existing compatible metadata or generate new metadata.
        """

        output_path = Path(metadata_path)

        expected_patient_ids = set(patient_ids)

        if output_path.exists() and not overwrite:
            metadata = ValidationMetadata.load(output_path)

            if metadata.patch_size != self.patch_size:
                raise ValueError(
                    "Existing validation metadata uses a different patch_size."
                )

            if metadata.seed != self.seed:
                raise ValueError("Existing validation metadata uses a different seed.")

            actual_patient_ids = set(metadata.patient_ids)

            if actual_patient_ids != expected_patient_ids:
                raise ValueError(
                    "Existing validation metadata does not match "
                    "the current validation patient split."
                )

            expected_patch_count = len(expected_patient_ids) * self.patches_per_patient

            if len(metadata.patches) != expected_patch_count:
                raise ValueError(
                    "Existing validation metadata uses a different "
                    "patches_per_patient value."
                )

            return metadata

        metadata = self.generate(patient_ids=patient_ids)

        metadata.save(output_path)

        return metadata
