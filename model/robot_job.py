from model.order import Order
from model.order_manager import OrderManager
from model.pod import Pod
from model.station import Station


class RobotJob:
    def __init__(self, pod: Pod, station: Station, order_manager: OrderManager):
        self.designated_pod = pod
        self.station = station
        self.orders = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 12
        self.picking_delay = 0
        self.is_active = True
        self.order_manager = order_manager

        self.designated_pod.is_idle = False

    def add_picking_task(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku

    def execute_pick(self):
        if not self.orders:
            return None

        order_id, sku, quantity = self.orders.pop(0)
        order = self.order_manager.get_order_by_id(order_id)
        order.deliver_quantity(sku, quantity)
        self.designated_pod.pick_sku(sku, quantity)

    def assign_station(self, station):
        self.station = station

    def mark_job_as_done(self):
        """Marks the job as done and sets the pod back to idle."""
        if self.is_active:
            self.is_active = False
            self.designated_pod.is_idle = True

    def pop_order(self):
        return self.orders.pop()