from typing import List

from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate


class Station(Object):
    def __init__(self, station_id: int, station_type: str):
        self.station_id = f"{station_type}-{station_id}"
        self.station_type = station_type
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        self.short_path: List[NetLogoCoordinate] = []
        self.long_path: List[NetLogoCoordinate] = []
        self.order_ids: List[int] = []
        self.max_orders = 2
        self.short_path_threshold = 4
        self.robot_ids = {}
        self.is_using_short_route = True
        super().__init__()

    def add_order(self, order_id: int):
        self.order_ids.append(order_id)

    def remove_order(self, order_id: int):
        self.order_ids.remove(order_id)

    def is_picker_station(self) -> bool:
        return self.station_type == "picker"

    def is_replenishment_station(self) -> bool:
        return self.station_type == "replenishment"

    def get_path(self):
        if self.is_using_short_route:
            return self.short_path
        else:
            return self.long_path

    def get_sub_path(self, robot_id, x: int, y: int):
        sub_path = []
        start = False
        for coord in self.get_robot_route(robot_id):
            if (coord.x, coord.y) == (x, y):
                start = True
            if start:
                sub_path.append(NetLogoCoordinate(coord.x, coord.y))
        return sub_path

    def reevaluate_route(self):
        if self.is_using_short_route and len(self.robot_ids) > self.short_path_threshold:
            self.is_using_short_route = False
        elif not self.is_using_short_route and len(self.robot_ids) == 0:
            self.is_using_short_route = True

    def add_robot(self, robot_id):
        self.robot_ids[robot_id] = self.get_path()
        self.reevaluate_route()

    def remove_robot(self, robot_id):
        if robot_id in self.robot_ids:
            del self.robot_ids[robot_id]
        self.reevaluate_route()

    def update_robot_route_type(self, robot_id):
        if robot_id in self.robot_ids:
            self.robot_ids[robot_id] = self.get_path()

    def get_robot_route(self, robot_id):
        return self.robot_ids.get(robot_id, None)

    def has_route_changed(self, robot_id):
        return self.get_path() != self.get_robot_route(robot_id)
