from typing import List
import os
import csv

from engine.netlogo_coordinate import NetLogoCoordinate
from .tools.pod_location import upsert_pod_location


def replenishment_service_steps_for_skus(pod, skus_to_replenish=None, *, delay_per_sku=20):
    # Full-pod replenishment service time: every visit restores all SKU
    # compartments on the pod, so the service time is always sized by the number
    # of distinct SKUs physically present on the pod (not the trigger subset).
    total_skus = len(getattr(pod, "skus", {}) or {})
    return int(total_skus) * int(delay_per_sku)


class RobotJob:
    counter = 1
    def __init__(self, pod_coordinate: NetLogoCoordinate, station_id, pod):
        self.my_id = RobotJob.counter
        RobotJob.counter +=1
        self.job_id = self.my_id
        self.pod_coordinate = pod_coordinate
        self.pod_return_coordinate = NetLogoCoordinate(0,0)
        self.pod = pod
        self.station_id = station_id
        self.orders: list[tuple[int, int, int]] = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 8 # Time for handling a task
        self.picking_delay = 0
        self.replenishment_delay_per_sku = 20
        self.replenishment_delay = 0
        self.replenishment_skus = []
        # Replenishment job identity + explicit source ("proactive" | "post_pick"
        # | "rts"). Source is preserved through pending/queued/active/completion
        # states and drives capacity accounting and diagnostics.
        self.is_replenishment_job = False
        self.replenishment_source = None
        # Proactive replenishment origin reservation: a proactive replenishment
        # pod must return to the exact storage it left. The origin is recorded at
        # dispatch and kept reserved (via the existing StorageManager ownership
        # model) for the whole trip; it is NOT released when the pod is collected.
        self.proactive_origin_reserved = False
        self.proactive_origin_storage = None
        self.proactive_origin_storage_id = None
        self.proactive_origin_coordinate = None
        self.is_finished = False
        self.rts_continuation_active = False
        self.rts_decision_identity = None
        self.rts_branch = None
        self.rts_zone_id = None
        self.rts_final_storage = None
        self.rts_final_destination = None
        self.rts_replenishment_station = None
        self.rts_source_station_id = None
        self.rts_stage = None
        self.rts_storage_reserved = False
        self.rts_pending_replenishment_request_snapshot = None
        self.rts_action_index = None
        self.rts_action_context_id = None
        self.rts_action_context_version = None
        self.rts_final_storage_id = None
        self.rts_replenishment_station_id = None
        self.rts_next_job_proposal_id = None
        self.rts_committed_next_reservation_id = None
        self.committed_next_reservation_id = None
        self.committed_next_owner_robot_id = None
        self.committed_next_activated_by_robot_id = None

    def add_picking_task(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku * quantity
    
    def add_replenishment_task(self, pod, skus_to_replenish=None, source=None):
        # ``skus_to_replenish`` is retained only as diagnostic trigger metadata;
        # the physical restoration is always full-pod and the service time is
        # sized by the pod's distinct SKU count.
        self.replenishment_skus = list(skus_to_replenish or [])
        self.is_replenishment_job = True
        if source is not None:
            self.replenishment_source = source
        self.replenishment_delay += replenishment_service_steps_for_skus(
            pod,
            self.replenishment_skus,
            delay_per_sku=self.replenishment_delay_per_sku,
        )

    def is_proactive_replenishment(self) -> bool:
        return bool(self.is_replenishment_job) and self.replenishment_source == "proactive"

    def set_proactive_origin(self, storage, storage_id, coordinate) -> None:
        self.proactive_origin_reserved = True
        self.proactive_origin_storage = storage
        self.proactive_origin_storage_id = storage_id
        self.proactive_origin_coordinate = coordinate

    def clear_proactive_origin_reservation(self) -> None:
        self.proactive_origin_reserved = False
        self.proactive_origin_storage = None
        self.proactive_origin_storage_id = None
        self.proactive_origin_coordinate = None

    def is_being_processed(self):
        """Check if the job is being processed based on delays."""
        return self.picking_delay > 0 or self.replenishment_delay > 0

    def decrement_delay(self):
        """Decrement the picking or replenishment delay."""
        if self.picking_delay > 0:
            self.picking_delay -= 1
        elif self.replenishment_delay > 0:
            self.replenishment_delay -= 1

    def set_job_finish(self):
        self.is_finished = True
        upsert_pod_location(self.pod.pod_id, self.pod.pos_x, self.pod.pos_y)

    def clear_rts_continuation(self):
        self.rts_continuation_active = False
        self.rts_decision_identity = None
        self.rts_branch = None
        self.rts_zone_id = None
        self.rts_final_storage = None
        self.rts_final_destination = None
        self.rts_replenishment_station = None
        self.rts_source_station_id = None
        self.rts_stage = None
        self.rts_storage_reserved = False
        self.rts_pending_replenishment_request_snapshot = None
        self.rts_action_index = None
        self.rts_action_context_id = None
        self.rts_action_context_version = None
        self.rts_final_storage_id = None
        self.rts_replenishment_station_id = None
        self.rts_next_job_proposal_id = None
        self.rts_committed_next_reservation_id = None

    def pop_order(self):
        return self.orders.pop(0)
    
    def __str__(self):
        return f"RobotJob: {str(self.job_id)}, pid {self.pod.pod_id}, xy {self.pod_coordinate}, stid {self.station_id}, ords {self.orders}"

    def __repr__(self):
        return self.__str__()
    
    def writePodReturnReport(self, manhattan_distance):
        return
        log_file = "log_pod_return.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Job ID", "Pod", "Pod Coordinate", "Return Coordinate", "Station ID", "Manhattan Distance"])
            
            # Write the data
            writer.writerow([
                self.job_id,
                self.pod,
                self.pod_coordinate,
                self.pod_return_coordinate,
                self.station_id,
                manhattan_distance
            ])

    def record_delivery(self, tick):
        return
        log_file = "log_pod_delivery.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Tick", "Pod", "Pod Coordinate", "Pod Coordinate inside pod object", "To Station ID"])
            
            # Write the data
            writer.writerow([
                f"{tick:.2f}",
                self.pod,
                self.pod_coordinate,
                (self.pod.pos_x, self.pod.pos_y),
                self.station_id
            ])
