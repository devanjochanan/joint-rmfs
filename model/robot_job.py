from model.order import Order
from model.pod import Pod
from model.station import Station


class RobotJob:
    def __init__(self, pod: Pod, station: Station):
        self.designated_pod = pod
        self.station = station
        self.orders = []  # This will hold tuples of (order, sku, quantity)
        self.picking_delay_per_sku = 12
        self.picking_delay = 0

    def add_picking_task(self, order: Order, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku

    def execute_pick(self):
        """Execute the picking job for the first order in the list.
                If no orders are left, returns None."""
        if not self.orders:
            return None  # Return None if there are no orders to process

        # Pop the first order from the list
        order, sku, quantity = self.orders.pop(0)

        # Deliver the quantity and pick the SKU from the pod
        order.deliver_quantity(sku, quantity)
        self.designated_pod.pick_sku(sku, quantity)

    def assign_station(self, station):
        self.station = station
