"""Read-only RTS replenishment-station queue estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .travel_time import backend_steps_to_seconds


QUEUE_ESTIMATOR_VERSION = "rts_replenishment_queue_estimator.v1"


@dataclass(frozen=True)
class RTSQueueEstimate:
    known: bool
    estimated_wait_seconds: float | None
    status: str
    station_id: str | None
    queue_semantics: str
    active_robot_count: int
    queued_robot_count: int
    server_count: int
    active_replenishment_work_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "queue_estimator_version": QUEUE_ESTIMATOR_VERSION,
            "known": bool(self.known),
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "status": self.status,
            "station_id": self.station_id,
            "queue_semantics": self.queue_semantics,
            "active_robot_count": int(self.active_robot_count),
            "queued_robot_count": int(self.queued_robot_count),
            "server_count": int(self.server_count),
            "active_replenishment_work_seconds": float(self.active_replenishment_work_seconds),
        }


def estimate_replenishment_queue(context: Any, station: Any | None) -> RTSQueueEstimate:
    warehouse = getattr(context, "warehouse", None)
    if station is None:
        return RTSQueueEstimate(
            known=False,
            estimated_wait_seconds=None,
            status="unavailable_missing_station",
            station_id=None,
            queue_semantics="host_parallel_processing_no_serial_service_queue",
            active_robot_count=0,
            queued_robot_count=0,
            server_count=0,
            active_replenishment_work_seconds=0.0,
        )
    robot_ids = getattr(station, "robot_ids", {}) or {}
    robot_queue = getattr(station, "robot_queue", []) or []
    server_count = int(getattr(station, "max_robots", 0) or 0)
    active_count = len(robot_ids)
    queued_count = len(robot_queue)
    active_work = _active_replenishment_work_seconds(warehouse, robot_ids)
    if server_count <= 0:
        return RTSQueueEstimate(
            known=False,
            estimated_wait_seconds=None,
            status="unavailable_station_capacity",
            station_id=_station_id(station),
            queue_semantics="host_parallel_processing_no_serial_service_queue",
            active_robot_count=active_count,
            queued_robot_count=queued_count,
            server_count=server_count,
            active_replenishment_work_seconds=active_work,
        )
    if active_count >= server_count:
        return RTSQueueEstimate(
            known=False,
            estimated_wait_seconds=None,
            status="unavailable_active_capacity_full",
            station_id=_station_id(station),
            queue_semantics="host_parallel_processing_no_serial_service_queue",
            active_robot_count=active_count,
            queued_robot_count=queued_count,
            server_count=server_count,
            active_replenishment_work_seconds=active_work,
        )
    return RTSQueueEstimate(
        known=True,
        estimated_wait_seconds=0.0,
        status="available_no_serial_queue_wait",
        station_id=_station_id(station),
        queue_semantics="host_parallel_processing_no_serial_service_queue",
        active_robot_count=active_count,
        queued_robot_count=queued_count,
        server_count=server_count,
        active_replenishment_work_seconds=active_work,
    )


def _active_replenishment_work_seconds(warehouse: Any, robot_ids: Any) -> float:
    if warehouse is None:
        return 0.0
    active_ids = {str(key) for key in getattr(robot_ids, "keys", lambda: [])()}
    total = 0.0
    for obj in getattr(warehouse, "_objects", []) or []:
        if str(getattr(obj, "_id", "")) not in active_ids:
            continue
        job = getattr(obj, "job", None)
        delay = getattr(job, "replenishment_delay", 0)
        seconds = backend_steps_to_seconds(warehouse, delay)
        if seconds is not None:
            total += seconds
    return float(total)


def _station_id(station: Any) -> str:
    return str(getattr(station, "station_id", ""))
