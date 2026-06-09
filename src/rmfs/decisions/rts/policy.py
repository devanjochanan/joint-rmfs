"""RTS policy protocol / interface.

Defines the abstract contract that any RTS policy must satisfy.
The protocol is deliberately minimal: one method, one context in,
one decision out.
"""

from typing import Protocol, runtime_checkable

from .types import RTSDestinationContext, RTSDecision


@runtime_checkable
class RTSPolicy(Protocol):
    """Strategy interface for return-to-storage destination selection.

    Implementations receive an ``RTSDestinationContext`` containing
    everything the current code uses when choosing a storage slot,
    and return an ``RTSDecision`` describing the chosen slot and
    destination coordinate.

    The robot controller (``model/robot.py``) keeps ownership of all
    side effects (storage reservation, path planning, telemetry).
    The policy only *selects*; it does not *execute*.
    """

    def select_destination(self, context: RTSDestinationContext) -> RTSDecision:
        """Choose a storage destination for the pod being returned."""
        ...
