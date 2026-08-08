from __future__ import annotations

import pytest

from backend.training.early_stopping import (
    EarlyStopping,
)


def test_first_metric_is_improvement() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    should_stop = early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    assert should_stop is False
    assert early_stopping.best_score == pytest.approx(1.0)
    assert early_stopping.bad_epochs == 0


def test_min_mode_detects_improvement() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    early_stopping.step(
        {
            "val_loss": 0.8,
        }
    )

    assert early_stopping.best_score == pytest.approx(0.8)
    assert early_stopping.bad_epochs == 0
    assert early_stopping.should_stop is False


def test_max_mode_detects_improvement() -> None:
    early_stopping = EarlyStopping(
        monitor="val_dice",
        patience=3,
        mode="max",
    )

    early_stopping.step(
        {
            "val_dice": 0.5,
        }
    )

    early_stopping.step(
        {
            "val_dice": 0.7,
        }
    )

    assert early_stopping.best_score == pytest.approx(0.7)
    assert early_stopping.bad_epochs == 0


def test_stops_after_patience_is_reached() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    assert (
        early_stopping.step(
            {
                "val_loss": 1.1,
            }
        )
        is False
    )

    assert (
        early_stopping.step(
            {
                "val_loss": 1.2,
            }
        )
        is False
    )

    assert (
        early_stopping.step(
            {
                "val_loss": 1.3,
            }
        )
        is True
    )

    assert early_stopping.bad_epochs == 3
    assert early_stopping.should_stop is True


def test_improvement_resets_bad_epochs() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    early_stopping.step(
        {
            "val_loss": 1.1,
        }
    )

    early_stopping.step(
        {
            "val_loss": 1.2,
        }
    )

    early_stopping.step(
        {
            "val_loss": 0.8,
        }
    )

    assert early_stopping.bad_epochs == 0
    assert early_stopping.should_stop is False
    assert early_stopping.best_score == pytest.approx(0.8)


def test_min_delta_filters_small_improvement() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
        min_delta=0.1,
    )

    early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    early_stopping.step(
        {
            "val_loss": 0.95,
        }
    )

    assert early_stopping.best_score == pytest.approx(1.0)
    assert early_stopping.bad_epochs == 1


def test_missing_monitor_metric_raises_error() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
    )

    with pytest.raises(
        KeyError,
        match="val_loss",
    ):
        early_stopping.step(
            {
                "train_loss": 1.0,
            }
        )


def test_state_dict_round_trip() -> None:
    first = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        min_delta=0.01,
    )

    first.step(
        {
            "val_loss": 1.0,
        }
    )

    first.step(
        {
            "val_loss": 1.1,
        }
    )

    state = first.state_dict()

    second = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        min_delta=0.01,
    )

    second.load_state_dict(state)

    assert second.state_dict() == state


def test_incompatible_state_raises_error() -> None:
    source = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    state = source.state_dict()

    target = EarlyStopping(
        monitor="val_dice",
        patience=3,
        mode="max",
    )

    with pytest.raises(
        ValueError,
        match="incompatible",
    ):
        target.load_state_dict(state)


def test_reset_clears_runtime_state() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=1,
    )

    early_stopping.step(
        {
            "val_loss": 1.0,
        }
    )

    early_stopping.step(
        {
            "val_loss": 2.0,
        }
    )

    assert early_stopping.should_stop is True

    early_stopping.reset()

    assert early_stopping.best_score is None
    assert early_stopping.bad_epochs == 0
    assert early_stopping.should_stop is False


@pytest.mark.parametrize(
    ("monitor", "patience", "mode", "min_delta"),
    [
        ("", 3, "min", 0.0),
        ("val_loss", 0, "min", 0.0),
        ("val_loss", -1, "min", 0.0),
        ("val_loss", 3, "invalid", 0.0),
        ("val_loss", 3, "min", -0.1),
    ],
)
def test_invalid_configuration_raises_error(
    monitor,
    patience,
    mode,
    min_delta,
) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
        )


def test_invalid_state_dict_type_raises_error() -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
    )

    with pytest.raises(
        TypeError,
        match="must be a dictionary",
    ):
        early_stopping.load_state_dict([])
