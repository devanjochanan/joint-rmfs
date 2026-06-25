"""RTS (Return-To-Storage) decision types.

Small, boring dataclasses for the RTS decision seam.
No behavior lives here — just the data contract between
the policy and the robot controller.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RTSDestinationContext:
    """Snapshot of the data available when the robot must choose
    a storage destination for its carried pod."""
    warehouse: Any          # model.inventory.Inventory
    robot: Any              # model.robot.Robot
    pod: Any                # model.pod.Pod
    station: Any            # model.station.Station


@dataclass(frozen=True)
class RTSDecision:
    """Result returned by an RTSPolicy.

    *storage* may be ``None`` when the policy falls back
    (e.g. no empty storage found in nearest mode, or fixed-return mode
    where no Storage object is tracked).

    *destination* is always a NetLogoCoordinate that the robot will
    navigate to.
    """
    storage: Any                # Optional[model.storage.Storage]
    destination: Any            # engine.netlogo_coordinate.NetLogoCoordinate
    policy_name: str
    mode: str                   # "fixed" | "nearest" | "nearest_fallback"
    reason: str = "current_behavior"
    metadata: Mapping[str, Any] = field(default_factory=dict)
