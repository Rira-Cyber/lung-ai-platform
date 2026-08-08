from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.preprocessing.patient_preprocessor import (
    PatientPreprocessor,
)
from backend.preprocessing.store import (
    PreprocessedPatientStore,
)


PATIENT_ID = "LIDC-IDRI-0001"


class FakeScan:
    def __init__(
        self,
        dicom_path: Path,
    ) -> None:
        self.dicom_path = dicom_path

    def get_path_to_dicom_files(
        self,
    ) -> str:
        return str(self.dicom_path)


class FakeProcessor:
    def __init__(
        self,
        dicom_path: Path,
    ) -> None:
        self.scan = FakeScan(dicom_path)

        self.build_calls = 0

    def load_patient(
        self,
        patient_id: str,
    ) -> FakeScan:
        return self.scan

    def hu_volume(
        self,
    ) -> np.ndarray:
        self.build_calls += 1

        return np.zeros(
            (4, 4, 4),
            dtype=np.float32,
        )

    def lung_mask(
        self,
    ) -> np.ndarray:
        return np.ones(
            (4, 4, 4),
            dtype=np.uint8,
        )

    def nodule_mask(
        self,
    ) -> np.ndarray:
        return np.zeros(
            (4, 4, 4),
            dtype=np.uint8,
        )


def create_preprocessor(
    tmp_path: Path,
) -> tuple[
    PatientPreprocessor,
    FakeProcessor,
]:
    dicom_path = tmp_path / "raw" / PATIENT_ID

    dicom_path.mkdir(parents=True)

    (dicom_path / "slice.dcm").write_bytes(b"fake-dicom")

    store = PreprocessedPatientStore(tmp_path / "processed")

    processor = FakeProcessor(dicom_path)

    preprocessor = PatientPreprocessor(
        dataset_path=tmp_path / "raw",
        store=store,
        processor=processor,
    )

    return (
        preprocessor,
        processor,
    )


def test_cache_miss_builds_artifact(
    tmp_path: Path,
) -> None:
    (
        preprocessor,
        processor,
    ) = create_preprocessor(tmp_path)

    artifact = preprocessor.get(PATIENT_ID)

    assert artifact.patient_id == PATIENT_ID
    assert processor.build_calls == 1

    assert preprocessor.store.exists(PATIENT_ID)


def test_cache_hit_does_not_rebuild(
    tmp_path: Path,
) -> None:
    (
        preprocessor,
        processor,
    ) = create_preprocessor(tmp_path)

    preprocessor.get(PATIENT_ID)

    preprocessor.get(PATIENT_ID)

    assert processor.build_calls == 1


def test_force_rebuild_reprocesses_patient(
    tmp_path: Path,
) -> None:
    (
        preprocessor,
        processor,
    ) = create_preprocessor(tmp_path)

    preprocessor.get(PATIENT_ID)

    preprocessor.rebuild(PATIENT_ID)

    assert processor.build_calls == 2


def test_raw_source_change_invalidates_cache(
    tmp_path: Path,
) -> None:
    (
        preprocessor,
        processor,
    ) = create_preprocessor(tmp_path)

    preprocessor.get(PATIENT_ID)

    dicom_path = Path(processor.scan.get_path_to_dicom_files())

    (dicom_path / "new_slice.dcm").write_bytes(b"changed-source")

    preprocessor.get(PATIENT_ID)

    assert processor.build_calls == 2


def test_preprocessing_change_invalidates_cache(
    tmp_path: Path,
) -> None:
    (
        first,
        processor,
    ) = create_preprocessor(tmp_path)

    first.get(PATIENT_ID)

    second = PatientPreprocessor(
        dataset_path=tmp_path / "raw",
        store=first.store,
        consensus_level=0.6,
        processor=processor,
    )

    second.get(PATIENT_ID)

    assert processor.build_calls == 2


def test_is_cached_reports_valid_cache(
    tmp_path: Path,
) -> None:
    (
        preprocessor,
        _,
    ) = create_preprocessor(tmp_path)

    assert not preprocessor.is_cached(PATIENT_ID)

    preprocessor.get(PATIENT_ID)

    assert preprocessor.is_cached(PATIENT_ID)
