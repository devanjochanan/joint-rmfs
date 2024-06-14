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
        self.order_ids: List[int] = []
        self.max_orders = 2
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
        return self.short_path
