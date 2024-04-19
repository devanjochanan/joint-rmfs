import pandas as pd

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from . import order_manager
from .order import Order
from .order_manager import OrderManager
from .pod import Pod
from .pod_manager import PodManager
from .robot import Robot
from .robot_job import RobotJob
from .station_manager import StationManager


class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stop_and_go = 0
    total_energy = 0
    total_pod = 0
    total_turning = 0
    movement_channel = {}
    graph = None
    graph_pod = None

    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station", "way-direction"]
        self.tick_to_second = 0.25
        self.job_queue = []
        self.landscape = Landscape(self.dimension)
        self.pod_manager = PodManager()
        self.station_manager = StationManager()
        self.order_manager = OrderManager()
        self.next_process_tick = 0

        super().__init__()

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod + 1
            self.total_pod += 1

        super().addObject(object)

    def addTrafficPolicyHistory(self, sender, target):
        if target not in self.movement_channel:
            self.movement_channel[target] = []

        self.movement_channel[target].append(sender)

    def getTrafficPolicyHistory(self, target):
        if target not in self.movement_channel:
            return []

        return self.movement_channel[target]

    def tick(self):
        self.movement_channel = {}
        if int(self._tick) == self.next_process_tick:
            self.find_new_orders()
            self.process_orders()
            self.next_process_tick += 1
        if len(self.job_queue) > 0:
            current_distance = 1000000
            current_id = -1

            for o in self.get_movable_objects():
                if len(self.job_queue) > 0:
                    job: RobotJob = self.job_queue[0]

                    if o.object_type == "robot" and o.job is None and o.current_state == 'idle':
                        dist = calculateDistance(o.pos_x, o.pos_y, job.designated_pod.pos_x, job.designated_pod.pos_y)
                        if dist < current_distance:
                            current_id = o.id
                            current_distance = dist

                        if current_id != -1:
                            self.job_queue.pop(0)

                        for movableObject in self.get_movable_objects():
                            if movableObject.id == current_id:
                                movableObject.assign_job_and_set_move_to_take_pod(job)

        total_energy = 0
        total_turning = 0
        for o in self.get_movable_objects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

        self.total_energy = total_energy
        self.total_turning = total_turning

        self._tick += self.tick_to_second

    def find_new_orders(self):
        orders_df = pd.read_csv('generated_order.csv')

        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = orders_df[(orders_df['Order Arrival (in second)'] <= current_second) &
                               (orders_df['Order Arrival (in second)'] > previous_second)]

        grouped_orders = new_orders.groupby('Order Id')

        for order_id, group in grouped_orders:
            order_items = group[['Item Id', 'Quantity']].to_dict('records')
            order = Order(order_id=order_id, order_arrival=current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.add_sku(item['Item Id'], item['Quantity'])

            self.order_manager.add_order(order)

        return new_orders

    def assign_job_to_available_robot(self, job: RobotJob):
        current_distance = 1000000
        current_id = -1

        for o in self.get_movable_objects():
            if isinstance(o, Robot) and o.job is None and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, job.designated_pod.pos_x, job.designated_pod.pos_y)
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.job_queue.append(job)
            return

        for o in self.get_movable_objects():
            if o.id == current_id and isinstance(o, Robot):
                o.assign_job_and_set_move_to_take_pod(job)

    def get_movable_objects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def process_orders(self):
        for order in self.order_manager.orders:
            if order.station is None:
                available_station = self.station_manager.find_available_station()
                if available_station is not None:
                    order.assign_station(available_station)
                else:
                    break

            if order.process_start_time >= 0:
                continue

            order.start_processing(int(self._tick))

            for sku in order.get_remaining_skus():
                available_pod: Pod = self.pod_manager.get_available_pod(sku)
                if available_pod is None:
                    continue
                quantity_to_take = order.get_quantity_left_for_sku(sku)
                order.commit_quantity(sku, quantity_to_take)

                job = RobotJob(available_pod, order.station, order_manager=self.order_manager)
                job.add_picking_task(order.order_id, sku, quantity_to_take)
                self.job_queue.append(job)
