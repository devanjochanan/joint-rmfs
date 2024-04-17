from model.pod import Pod
from model.station import Station


class RobotJob:
    def __init__(self, pod: Pod, station: Station):
        self.designated_pod = pod
        self.station = station
        self.orders = []  # This will hold tuples of (order, sku, quantity)

    def add_order_sku(self, order, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order, sku, quantity))

    def execute_pick(self):
        """Execute the picking job for all orders associated with it."""
        for order, sku, quantity in self.orders:
            order.update_sku_quantity(sku, quantity)
            self.designated_pod.pick_sku(sku, quantity)

    def assign_station(self, station):
        self.station = station
