from typing import List

from engine.netlogo_coordinate import NetLogoCoordinate


class RobotJob:
    def __init__(self, pod_coordinate: NetLogoCoordinate, station_coordinate: NetLogoCoordinate, station_path: List[NetLogoCoordinate]):
        self.pod_coordinate = pod_coordinate
        self.station_coordinate = station_coordinate
        self.station_path = station_path
        self.orders = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 100
        self.picking_delay = 0
        self.is_finished = False

    def add_picking_task(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku

    def pop_order(self):
        return self.orders.pop(0)
