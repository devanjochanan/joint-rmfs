"""Validation helpers for RTS-RL port contracts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

RAW_THRESHOLD_FEATURE_NAMES = {
    "paper_replenishment_threshold",
    "threshold",
    "global_threshold_inv_level",
}


def validate_no_raw_threshold_features(feature_names: Sequence[str]) -> None:
    present = RAW_THRESHOLD_FEATURE_NAMES.intersection(str(name) for name in feature_names)
    if present:
        raise ValueError(f"raw threshold feature names are forbidden: {sorted(present)}")


def validate_action_mask_shape(mask: Sequence[int], action_count: int) -> None:
    if len(tuple(mask)) != int(action_count):
        raise ValueError("RTS-RL action mask shape mismatch")


def validate_feature_matrix_shape(matrix: np.ndarray, feature_names: Sequence[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError("RTS-RL feature matrix must be 2D")
    if matrix.shape[1] != len(tuple(feature_names)):
        raise ValueError("RTS-RL feature matrix width does not match feature names")


def validate_stock_matrix_shape(matrix: np.ndarray, stock_feature_names: Sequence[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError("RTS-RL stock matrix must be 2D")
    if matrix.shape[1] != len(tuple(stock_feature_names)):
        raise ValueError("RTS-RL stock matrix width does not match stock feature names")


def validate_reward_result(result) -> None:
    if result.reward_computed and result.reward_value is None:
        raise ValueError("RTS-RL reward marked computed without reward_value")


def validate_model_output_shape(logits, values, batch_size: int, action_count: int) -> None:
    if tuple(logits.shape) != (int(batch_size), int(action_count)):
        raise ValueError("RTS-RL logits shape mismatch")
    if tuple(values.shape) != (int(batch_size),):
        raise ValueError("RTS-RL value shape mismatch")
