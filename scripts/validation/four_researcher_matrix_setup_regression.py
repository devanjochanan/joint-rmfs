#!/usr/bin/env python3
"""Focused regression for the four-researcher compatibility matrix setup.

Covers:
  * exactly 16 unique conditions with stable ids;
  * each condition has its own isolated input root (never data/input/base);
  * default concurrency is exactly eight workers;
  * requested/actual mode fields are present for every factor;
  * censor-rate denominator logic (null when no RTS decisions occurred).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.experiments.four_researcher_compatibility_matrix as matrix


def _args(output_root):
    return argparse.Namespace(
        ticks=2000,
        max_workers=8,
        seed=42,
        output_root=str(output_root),
        baseline_bundle=matrix.DEFAULT_BASELINE_BUNDLE,
        lukman_bundle=matrix.DEFAULT_LUKMAN_BUNDLE,
        prepare_only=True,
        execute=False,
        keep_runtime_artifacts=False,
    )


def test_sixteen_unique_isolated_conditions():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = _args(out)
        conditions = [
            matrix.build_condition(d, p, l, s, args=args, output_root=out, commit="c", branch="b")
            for d, p, l, s in itertools.product((False, True), repeat=4)
        ]
        assert len(conditions) == 16
        ids = {c["condition_id"] for c in conditions}
        assert len(ids) == 16, ids
        assert "D0_P0_L0_S0" in ids and "D1_P1_L1_S1" in ids

        input_roots = {c["isolation"]["input_root"] for c in conditions}
        assert len(input_roots) == 16, "each condition must have its own isolated input root"
        base = str((REPO_ROOT / "data" / "input" / "base").resolve())
        for c in conditions:
            ir = c["isolation"]["input_root"]
            assert base not in ir, "no condition may target the mutable data/input/base"
            assert str(out) in ir, "input roots must live under the isolated output root"
        # Requested/actual present for every factor.
        for c in conditions:
            for factor in ("dewa", "devan", "lukman", "salsa"):
                assert c[factor]["requested"], factor
                assert c[factor]["actual"], factor
        print("PASS test_sixteen_unique_isolated_conditions")


def test_default_concurrency_is_eight():
    parser = matrix.build_arg_parser()
    ns = parser.parse_args([])
    assert ns.max_workers == 8
    assert ns.ticks == 2000
    assert ns.seed == 42
    print("PASS test_default_concurrency_is_eight")


def test_censor_rate_denominator_logic():
    # No RTS decisions -> null, not zero.
    assert matrix.censored_rate(0, 0) is None
    # finalized = completed + censored.
    assert matrix.censored_rate(3, 1) == 0.25
    assert matrix.censored_rate(0, 4) == 1.0
    assert matrix.censored_rate(4, 0) == 0.0
    print("PASS test_censor_rate_denominator_logic")


def test_factor_mode_resolution():
    # Dewa ON is interface-compatibility (no trained checkpoint).
    dewa_on = matrix.resolve_dewa(True)
    assert dewa_on.detail["uses_trained_checkpoint"] is False
    assert dewa_on.detail["rts_policy_mode"] == "random_valid"
    # Devan ON with no checkpoint -> random interface compatibility mode.
    devan_on = matrix.resolve_devan(True)
    assert devan_on.actual in {"pps_ppo_strict", "pps_random_interface_compatibility_mode"}
    # Salsa ON must resolve through the production charging config with cells.
    salsa_on = matrix.resolve_salsa(True)
    assert salsa_on.detail["charging_enabled"] is True
    assert salsa_on.detail.get("charger_cells_present") is True
    print("PASS test_factor_mode_resolution")


def main():
    test_sixteen_unique_isolated_conditions()
    test_default_concurrency_is_eight()
    test_censor_rate_denominator_logic()
    test_factor_mode_resolution()
    print("\nALL FOUR-RESEARCHER MATRIX SETUP REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
