"""Default RTS policy — preserves current fixed/nearest behavior.

This policy wraps the exact same decision logic that previously lived
inline in ``model/robot.py`` (lines 730–758 at commit d690c77).  It
handles both ``return_fix`` and ``return_nearest`` modes and produces
the same destination/storage choices as the original code.

No new algorithm or heuristic is introduced.  This is a pure
delegation seam for future RTS-RL work.
"""

from engine.netlogo_coordinate import NetLogoCoordinate
from .types import RTSDestinationContext, RTSDecision

POLICY_NAME = "current_rts"


class CurrentRTSPolicy:
    """Behavior-preserving default RTS policy.

    Delegates to the exact same fixed/nearest logic that was inline
    in ``Robot.advance_state_if_needed`` before the seam was added.
    """

    def select_destination(self, context: RTSDestinationContext) -> RTSDecision:
        robot = context.robot
        pod = context.pod
        station = context.station
        warehouse = context.warehouse

        if robot.return_fix:
            # --- Fixed return: same coordinate the pod was picked from ---
            return RTSDecision(
                storage=None,
                destination=robot.job.pod_coordinate,
                policy_name=POLICY_NAME,
                mode="fixed",
                reason="return_fix is True; returning to original pod_coordinate",
            )

        if robot.return_nearest:
            # --- Nearest empty storage: identical query as old inline code ---
            nearest_storage = warehouse.storage_manager.getNearestEmptyStorageToLocation(
                location_coordinate=station.coordinate,
                robot_coordinate=robot.coordinate,
            )

            if nearest_storage is not None:
                destination = NetLogoCoordinate(
                    nearest_storage.pos_x, nearest_storage.pos_y
                )
                return RTSDecision(
                    storage=nearest_storage,
                    destination=destination,
                    policy_name=POLICY_NAME,
                    mode="nearest",
                    reason="return_nearest is True; nearest empty storage found",
                )
            else:
                # Fallback: no empty storage available — return pod to its
                # current live position (same as original code).
                destination = NetLogoCoordinate(pod.pos_x, pod.pos_y)
                return RTSDecision(
                    storage=None,
                    destination=destination,
                    policy_name=POLICY_NAME,
                    mode="nearest_fallback",
                    reason="return_nearest is True but no empty storage; "
                           "falling back to current pod position",
                )

        # Neither flag is set — this path did not exist in the original
        # code (one of the two flags was always True). Raise explicitly.
        raise RuntimeError(
            "Invalid RTS configuration: neither return_fix nor return_nearest is enabled"
        )
