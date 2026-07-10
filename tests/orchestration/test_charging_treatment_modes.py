import json
from types import SimpleNamespace

from src.rmfs.app.netlogo_api import _configure_charging_treatment, _uses_grid_charger_cells
from src.rmfs.orchestration.run_spec import RunSpec


def _warehouse():
    return SimpleNamespace(charger_cells={(99, 99)}, active_charger_cells={(99, 99)})


def _config(tmp_path):
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps({
        "num_chargers": 2, "charger_positions": [[1, 2], [3, 4]],
        "battery_low_pct": 18, "battery_charged_pct": 60,
        "battery_interrupt_pct": 50, "disable_active_charging": False,
    }), encoding="utf-8")
    return path


def test_reference_off_has_no_functional_chargers(tmp_path):
    warehouse = _warehouse()
    _configure_charging_treatment(warehouse, placement_source="reference_off", config_path=None)
    assert warehouse.charging_enabled is False
    assert warehouse.charger_cells == set()
    assert warehouse.active_charger_cells == set()
    assert _uses_grid_charger_cells("reference_off") is False


def test_salsa_registers_only_generated_coordinates(tmp_path):
    warehouse = _warehouse()
    _configure_charging_treatment(warehouse, placement_source="salsa_adaptive_on", config_path=str(_config(tmp_path)))
    # JSON [row, col] is converted to runtime (x=col, y=row).
    assert warehouse.charger_cells == {(2, 1), (4, 3)}
    assert warehouse.charging_enabled is True
    assert warehouse.charging_declared_count == 2
    assert _uses_grid_charger_cells("salsa_adaptive_on") is False


def test_legacy_union_is_the_only_grid_compatibility_mode():
    assert _uses_grid_charger_cells("legacy_union") is True
    assert _uses_grid_charger_cells("reference_off") is False


def test_new_runs_default_off_and_old_json_is_legacy_union(tmp_path):
    spec = RunSpec(run_id="r", ticks=1, runtime_root=tmp_path / "r", repo_root=tmp_path, run_profile="gui")
    assert spec.charging_placement_source == "reference_off"
    old = spec.to_json_dict()
    old.pop("charging_placement_source")
    assert RunSpec.from_json_dict(old).charging_placement_source == "legacy_union"
