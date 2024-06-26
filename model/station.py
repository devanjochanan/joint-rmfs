from typing import List, Optional, Dict

from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate

from .order_manager import OrderManager
from .order import Order


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
        self.max_orders = 6
        self.short_path_threshold = 4
        self.robot_ids = {}
        self.is_using_short_route = True
        self.skus = {}
        self.incoming_pod: List[int] = []
        super().__init__()

    def add_order(self, order_id: int):
        self.order_ids.append(order_id)

    def remove_order(self, order_id: int):
        if order_id in self.order_ids:
            self.order_ids.remove(order_id)

    def add_pod(self, pod):
        self.incoming_pod.append(pod)
    
    def remove_pod(self, pod):
        self.incoming_pod.remove(pod)

    def is_picker_station(self) -> bool:
        return self.station_type == "picker"

    def is_replenishment_station(self) -> bool:
        return self.station_type == "replenishment"

    def get_path(self):
        if self.is_using_short_route:
            return self.short_path
        else:
            return self.long_path

    def get_sub_path(self, x: int, y: int):
        sub_path = []
        start = False
        for coord in self.get_path():
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
    
    def get_skus_in_station(self, order_manager: OrderManager):
        self._skus_in_station(order_manager)
        return self.skus
    
    def _skus_in_station(self, order_manager: OrderManager):
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            for sku, value in order.get_remaining_skus().items():
                if sku not in self.skus:
                    self.skus[sku] = value
                self.skus[sku] += value
        return
    
    def get_orders_in_station(self, order_manager: OrderManager) -> Optional[List[Order]]: 
        orders = []
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            orders.append(order)
        return orders
    
    def get_skus_in_station_dict(self, order_manager: OrderManager) -> Optional[Dict]:
        orders_sku = {}
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            for sku, value in order.get_remaining_skus().items():
                if sku not in orders_sku:
                    orders_sku[sku] = []
                orders_sku[sku].append(value)

        return self._sort_order_set(orders_sku)

    def _sort_order_set(self,order_set):
        sorted_order_set = {}
        for index, value in order_set.items():
            sorted_order_set[index] = sorted(value, reverse=False)
        return sorted_order_set
