from engine.landscape import Landscape
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from engine.universe import Universe
from engine.util import *
from .order import Order
from .robot import Robot

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

    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station", "way-direction"]
        self.tick_to_second = 0.25
        self.order_queue = []
        self.landscape = Landscape(self.dimension)

        super().__init__()

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod+1
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
        if len(self.order_queue) > 0:
            current_distance = 1000000
            current_id = -1

            for o in self.moveableObjects():
                if o.object_type == "robot" and o.current_state == 'returning_pod':
                    order = self.order_queue[0]
                    if order.has_to_take_pod == False:
                        dist = calculateDistance(o.pos_x, o.pos_y, order.designated_pod[0], order.designated_pod[1])
                        if dist < current_distance:
                            current_id = o.id
                            current_distance = dist

                        if current_id != -1:
                            self.order_queue.pop(0)

                        for o in self.moveableObjects():
                            if o.id == current_id:
                                o.setOrder2(order)

                if o.object_type == "robot" and o.order is None and o.current_state == 'idle' and o.has_to_take_pod == True:
                    order = self.order_queue[0]
                    dist = calculateDistance(o.pos_x, o.pos_y, order.designated_pod[0], order.designated_pod[1])
                    if dist < current_distance:
                        current_id = o.id
                        current_distance = dist

                    if current_id != -1:
                        self.order_queue.pop(0)

                    for o in self.moveableObjects():
                        if o.id == current_id:
                            o.setOrder(order)

        total_energy = 0
        total_turning = 0
        for o in self.moveableObjects():
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0:
                    self.stop_and_go += 1
        
        self.total_energy = total_energy
        self.total_turning = total_turning
        self._tick += self.tick_to_second

    def addOrder(self, order: Order):
        current_distance = 1000000
        current_id = -1

        for o in self.moveableObjects():
            if o.object_type == "robot" and o.order is None and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, order.designated_pod[0], order.designated_pod[1])
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.order_queue.append(order)
            return
        
        for o in self.moveableObjects():
            if o.id == current_id:
                o.setOrder(order)

    def addStation(self, station):
        self.stations.append(station)

    def moveableObjects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result
