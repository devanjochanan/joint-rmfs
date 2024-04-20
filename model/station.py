from typing import List

from engine.object import Object


class Station(Object):
    def __init__(self, station_id: int):
        self.station_id = station_id
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        self.order_ids: List[int] = []
        self.max_orders = 2
        super().__init__()

    def add_order(self, order_id: int):
        self.order_ids.append(order_id)

    def remove_order(self, order_id: int):
        self.order_ids.remove(order_id)
