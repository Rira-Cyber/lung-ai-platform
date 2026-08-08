from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.configs import DEV_CONFIG
from backend.datasets.patient_manifest import (
    load_patient_manifest,
)
from backend.training.dataloader import (
    create_train_validation_loaders,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Lung AI DataLoader performance."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=("Optional patient manifest used to limit the benchmark to a subset."),
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Maximum number of training batches to benchmark.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.batches <= 0:
        raise ValueError("--batches must be greater than zero.")

    patient_ids = None

    if arguments.manifest is not None:
        patient_ids = load_patient_manifest(arguments.manifest)

    print("=" * 60)
    print("DataLoader Benchmark")
    print("=" * 60)

    setup_start = time.perf_counter()

    loaders = create_train_validation_loaders(
        dataset_path=DEV_CONFIG.dataset_path,
        batch_size=DEV_CONFIG.batch_size,
        num_workers=DEV_CONFIG.num_workers,
        patch_size=DEV_CONFIG.patch_size,
        positive_ratio=DEV_CONFIG.positive_ratio,
        patches_per_patient=(DEV_CONFIG.patches_per_patient),
        val_ratio=DEV_CONFIG.val_ratio,
        test_ratio=DEV_CONFIG.test_ratio,
        split_seed=DEV_CONFIG.split_seed,
        pin_memory=DEV_CONFIG.pin_memory,
        shuffle=DEV_CONFIG.shuffle,
        patient_ids=patient_ids,
    )

    setup_time = time.perf_counter() - setup_start

    print(f"Loader setup time     : {setup_time:.2f}s")

    print(f"Train patients        : {len(loaders.patient_split.train_ids)}")

    print(f"Validation patients   : {len(loaders.patient_split.val_ids)}")

    print(f"Reserved test patients: {len(loaders.patient_split.test_ids)}")

    print("-" * 60)

    batch_times: list[float] = []

    iterator = iter(loaders.train_loader)

    for batch_index in range(arguments.batches):
        start = time.perf_counter()

        try:
            batch = next(iterator)
        except StopIteration:
            break

        elapsed = time.perf_counter() - start

        batch_times.append(elapsed)

        print(
            f"Batch {batch_index + 1:02d}"
            f" | "
            f"{elapsed:.3f}s"
            f" | "
            f"shape={tuple(batch['image'].shape)}"
        )

    if not batch_times:
        raise RuntimeError("No training batches were produced.")

    total_time = sum(batch_times)

    average_time = total_time / len(batch_times)

    fastest_time = min(batch_times)

    slowest_time = max(batch_times)

    print("-" * 60)

    print(f"Batches measured      : {len(batch_times)}")

    print(f"Average batch time    : {average_time:.3f}s")

    print(f"Fastest batch         : {fastest_time:.3f}s")

    print(f"Slowest batch         : {slowest_time:.3f}s")

    print(f"Approx batches/sec    : {1.0 / average_time:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
