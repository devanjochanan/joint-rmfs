from engine.landscape import Landscape
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from engine.universe import Universe
from engine.util import *
from .order import Order
from .robot import Robot

class Inventory(Universe):
    dimension = 52
    map = []
    landscape = None
    stations = []
    total_energy = 0

    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station"]
        self.tick_to_second = 0.5
        self.order_queue = []
        self.landscape = Landscape(self.dimension)

        super().__init__()

    def addObject(self, object):
        if object.object_type == "pod":
            self.landscape.setObject(NetLogoCoordinate(object.pos_x, object.pos_y), 1)
        
        super().addObject(object)

    def tick(self):
        if len(self.order_queue) > 0:
            current_distance = 1000000
            current_id = -1

            for o in self.moveableObjects():
                if o.object_type == "robot" and o.order is None:
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
        for o in self.moveableObjects():
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
        
        self.total_energy = total_energy
        self._tick += self.tick_to_second

    def addOrder(self, order: Order):
        current_distance = 1000000
        current_id = -1

        for o in self.moveableObjects():
            if o.object_type == "robot" and o.order is None:
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
