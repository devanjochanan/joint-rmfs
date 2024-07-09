from typing import List

from engine.netlogo_coordinate import NetLogoCoordinate


class RobotJob:
    def __init__(self, pod_coordinate: NetLogoCoordinate, station_id):
        self.job_id = id
        self.pod_coordinate = pod_coordinate
        self.station_id = station_id
        self.orders = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 100
        self.picking_delay = 0
        self.replenishment_delay_per_sku = 100
        self.replenishment_delay = 100
        self.is_finished = False

    def add_picking_task(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku

    def add_replenishment_task(self, pod):
        total_skus = len(pod.skus)
        self.replenishment_delay += total_skus * self.replenishment_delay_per_sku

    def is_being_processed(self):
        """Check if the job is being processed based on delays."""
        return self.picking_delay > 0 or self.replenishment_delay > 0

    def decrement_delay(self):
        """Decrement the picking or replenishment delay."""
        if self.picking_delay > 0:
            self.picking_delay -= 1
        elif self.replenishment_delay > 0:
            self.replenishment_delay -= 1

    def pop_order(self):
        return self.orders.pop(0)

    def __str__(self):
        return f"RobotJob: {self.job_id}, {self.pod_coordinate}, {self.station_id}, {self.orders}"

    def __repr__(self):
        return self.__str__()
