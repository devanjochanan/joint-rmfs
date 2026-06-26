#!/usr/bin/env python3
"""Pure smoke for RTS robot-pressure and station-load accounting."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.zone_features import build_zone_rows, infer_pressure_zone_id
from src.rmfs.rl.rts.zone_registry import build_zone_registry


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def storage(number: int, x: int, y: int, zone_id: str) -> Obj:
    return Obj(
        storage_number=number,
        pos_x=x,
        pos_y=y,
        zone_id=zone_id,
        is_empty=True,
        assigned_pod=None,
    )


def base_storages() -> list[Obj]:
    return [
        storage(1, 10, 10, "A"),
        storage(2, 20, 10, "B"),
    ]


def context_for(objects: list[Obj], *, source_station: Obj | None = None, stations: list[Obj] | None = None) -> Obj:
    picker = source_station or Obj(station_id="picker-1", station_type="picker", pos_x=0, pos_y=0)
    station_list = stations if stations is not None else [picker]
    warehouse = Obj(
        storage_manager=Obj(storages=base_storages()),
        station_manager=Obj(stations=station_list),
        pod_manager=Obj(pods=[]),
        order_manager=Obj(orders=[]),
        _objects=objects,
    )
    return Obj(
        warehouse=warehouse,
        robot=objects[0] if objects else None,
        pod=Obj(pod_id=1, skus={}),
        station=picker,
    )


def zone_rows_for(objects: list[Obj]) -> dict[str, dict]:
    rows, warnings = build_zone_rows(context_for(objects), ("A", "B"))
    assert isinstance(warnings, list)
    return {str(row["zone_id"]): row for row in rows}


def assert_zone_pressure(row: dict, *, present: float = 0.0, destination: float = 0.0) -> None:
    assert row["zone_present_robot_count"] == present
    assert row["zone_destination_robot_count"] == destination


def main() -> None:
    rows = zone_rows_for([Obj(object_type="storage", pos_x=10, pos_y=10)])
    assert_zone_pressure(rows["A"], present=0.0)

    rows = zone_rows_for([Obj(object_type="robot", pos_x=10, pos_y=10)])
    assert_zone_pressure(rows["A"], present=1.0)

    rows = zone_rows_for([Obj(object_type="robot", pos_x=11, pos_y=10)])
    assert_zone_pressure(rows["A"], present=1.0)

    far_pathway = Obj(object_type="robot", pos_x=14, pos_y=10)
    rows = zone_rows_for([far_pathway])
    assert_zone_pressure(rows["A"], present=0.0)
    assert_zone_pressure(rows["B"], present=0.0)

    rows = zone_rows_for([
        Obj(object_type="robot", pos_x=0, pos_y=0, destination=Obj(x=9, y=10)),
        Obj(object_type="robot", pos_x=0, pos_y=0, destination=Obj(x=14, y=10)),
    ])
    assert_zone_pressure(rows["A"], destination=1.0)
    assert_zone_pressure(rows["B"], destination=0.0)

    context = context_for([])
    registry = build_zone_registry(context, ("A", "B"))
    assert registry.zone_id_for_coordinate(Obj(x=14, y=10)) == "A"
    assert infer_pressure_zone_id(Obj(x=14, y=10), ("A", "B"), registry=registry) == ""

    repl = Obj(station_id="repl-1", station_type="replenishment", pos_x=30, pos_y=10)
    picker = Obj(station_id="picker-1", station_type="picker", pos_x=0, pos_y=0)
    state_context = context_for(
        [
            Obj(object_type="robot", pos_x=0, pos_y=0, station=repl),
            Obj(object_type="storage", pos_x=0, pos_y=0, station=repl),
            Obj(object_type="pod", pos_x=0, pos_y=0, destination=Obj(x=30, y=10)),
        ],
        source_station=picker,
        stations=[picker, repl],
    )
    state = build_state(state_context, ("A", "B")).state_json
    assert "selected_replenishment_station_logical_load" not in state["spatial_context"]
    assert state["zone_registry"]["robot_pressure_denominator"] == "total active warehouse robot objects"

    print("rts zone robot pressure smoke ok")


if __name__ == "__main__":
    main()
