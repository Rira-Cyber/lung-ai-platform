from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.configs import DEV_CONFIG, PROD_CONFIG
from backend.configs.config import Config
from backend.datasets.patient_manifest import (
    load_patient_manifest,
)
from backend.preprocessing.patient_preprocessor import (
    PatientPreprocessor,
)
from backend.preprocessing.store import (
    PreprocessedPatientStore,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build the incremental preprocessing cache for LIDC patients.")
    )

    parser.add_argument(
        "--config",
        choices=(
            "development",
            "production",
        ),
        default="development",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional patient manifest. "
            "If omitted, all patient directories "
            "in the configured dataset are used."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=("Force rebuilding artifacts even when the existing cache is valid."),
    )

    return parser.parse_args()


def resolve_config(
    config_name: str,
) -> Config:
    if config_name == "development":
        return DEV_CONFIG

    if config_name == "production":
        return PROD_CONFIG

    raise ValueError(f"Unsupported config: {config_name}")


def resolve_patient_ids(
    dataset_path: Path,
    manifest_path: Path | None,
) -> tuple[str, ...]:
    if manifest_path is not None:
        return load_patient_manifest(manifest_path)

    patient_ids = tuple(
        sorted(folder.name for folder in dataset_path.iterdir() if folder.is_dir())
    )

    if not patient_ids:
        raise RuntimeError(f"No patient directories found in: {dataset_path}")

    return patient_ids


def main() -> None:
    arguments = parse_arguments()

    config = resolve_config(arguments.config)

    dataset_path = Path(config.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    patient_ids = resolve_patient_ids(
        dataset_path=dataset_path,
        manifest_path=arguments.manifest,
    )

    processed_store_path = dataset_path.parent / "processed" / "patients"

    store = PreprocessedPatientStore(processed_store_path)

    preprocessor = PatientPreprocessor(
        dataset_path=dataset_path,
        store=store,
    )

    total = len(patient_ids)

    built = 0
    reused = 0
    failed = 0

    print("=" * 60)
    print("LIDC Patient Cache Builder")
    print("=" * 60)

    print(f"Dataset           : {dataset_path}")

    print(f"Cache             : {processed_store_path}")

    print(f"Patients requested: {total}")

    print(f"Force rebuild     : {arguments.force}")

    print("-" * 60)

    start_time = time.perf_counter()

    for index, patient_id in enumerate(
        patient_ids,
        start=1,
    ):
        patient_start = time.perf_counter()

        try:
            if arguments.force:
                preprocessor.rebuild(patient_id)

                status = "BUILT"
                built += 1

            else:
                was_valid = preprocessor.is_cached(patient_id)

                preprocessor.get(patient_id)

                if was_valid:
                    status = "HIT"
                    reused += 1
                else:
                    status = "BUILT"
                    built += 1

            elapsed = time.perf_counter() - patient_start

            print(f"[{index:03d}/{total:03d}] {patient_id} {status:<5} {elapsed:.2f}s")

        except Exception as error:
            failed += 1

            elapsed = time.perf_counter() - patient_start

            print(
                f"[{index:03d}/{total:03d}] "
                f"{patient_id} "
                f"FAILED "
                f"{elapsed:.2f}s "
                f"- {error}"
            )

    elapsed_total = time.perf_counter() - start_time

    print("-" * 60)

    print(f"Built             : {built}")

    print(f"Cache hits        : {reused}")

    print(f"Failed            : {failed}")

    print(f"Elapsed           : {elapsed_total:.2f}s")

    print("=" * 60)

    if failed:
        raise RuntimeError(f"Cache build completed with {failed} failed patient(s).")


if __name__ == "__main__":
    main()
