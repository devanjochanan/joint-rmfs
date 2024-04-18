import pandas as pd

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from .order import Order
from .pod import Pod
from .robot import Robot
from .robot_job import RobotJob


class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stations = []
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
        self.orders = []
        self.landscape = Landscape(self.dimension)

        # Dictionary mapping coordinate tuples (x, y) to Pod instances
        # Key: Tuple representing the coordinates (x, y) of the Pod
        # Value: Pod instance located at those coordinates
        self.coordinate_to_pods = {}

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
        if len(self.job_queue) > 0:
            current_distance = 1000000
            current_id = -1

            for o in self.moveableObjects():
                if len(self.job_queue) > 0:
                    order: Robot = self.job_queue[0]
                    if o.object_type == "robot" and o.current_state == 'returning_pod':
                        o: Robot = o
                        if not order.has_to_take_pod:
                            dist = calculateDistance(o.pos_x, o.pos_y, order.designated_pod[0], order.designated_pod[1])
                            if dist < current_distance:
                                current_id = o.id
                                current_distance = dist

                            if current_id != -1:
                                self.job_queue.pop(0)

                            for movableObject in self.moveableObjects():
                                if movableObject.id == current_id:
                                    movableObject.setOrderNoPod(order)

                    if o.object_type == "robot" and o.order is None and o.current_state == 'idle' and order.has_to_take_pod == True:
                        dist = calculateDistance(o.pos_x, o.pos_y, order.designated_pod[0], order.designated_pod[1])
                        if dist < current_distance:
                            current_id = o.id
                            current_distance = dist

                        if current_id != -1:
                            self.job_queue.pop(0)

                        for movableObject in self.moveableObjects():
                            if movableObject.id == current_id:
                                movableObject.setOrder(order)

        total_energy = 0
        total_turning = 0
        for o in self.moveableObjects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

        self.total_energy = total_energy
        self.total_turning = total_turning

        print(self.find_new_orders())
        self._tick += self.tick_to_second

    def find_new_orders(self):
        orders_df = pd.read_csv('generated_order.csv')

        current_second = self._tick * self.tick_to_second
        previous_second = (self._tick - 4) * self.tick_to_second

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = orders_df[(orders_df['Order Arrival (in second)'] <= current_second) &
                               (orders_df['Order Arrival (in second)'] > previous_second)]

        for index, row in new_orders.iterrows():
            # Assuming Pod object or similar needs to be passed; placeholder Pod() used
            order = Order(order_id=row['Order Id'], order_arrival_in_seconds=row['Order Arrival (in second)'])
            order.add_sku(row['Item Id'], row['Quantity'])

            self.orders.append(order)

        return new_orders

    def assign_job(self, job: RobotJob):
        current_distance = 1000000
        current_id = -1

        for o in self.moveableObjects():
            if isinstance(o, Robot) and o.job is None and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, job.designated_pod.pos_x, job.designated_pod.pos_y)
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.job_queue.append(job)
            return

        for o in self.moveableObjects():
            if o.id == current_id and isinstance(o, Robot):
                o.assign_job_and_set_move_to_take_pod(job)

    def addStation(self, station):
        self.stations.append(station)

    def add_pod(self, pod: Pod, x, y):
        """Add a pod at a specific coordinate."""
        self.coordinate_to_pods[(x, y)] = pod  # Store pod by its coordinate tuple

    def find_pod(self, x, y):
        """Find and return the pod at the specified x and y coordinates using dictionary lookup."""
        return self.coordinate_to_pods.get((x, y), None)  # Returns None if no pod is found at those coordinates

    def moveableObjects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result
