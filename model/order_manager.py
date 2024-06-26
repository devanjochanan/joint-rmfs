from typing import List, Dict, Optional

from model.order import Order


class OrderManager:
    def __init__(self):
        self.orders: List[Order] = []
        self.order_id_to_order: Dict[int, Order] = {}
        self.finished_order = []

    def add_order(self, order: Order):
        self.orders.append(order)
        self.order_id_to_order[order.order_id] = order

    def get_order_by_id(self, order_id: int) -> Optional[Order]:
        """Retrieve an order by its ID using the dictionary for quick access."""
        return self.order_id_to_order.get(order_id, None)

    def remove_order(self, order:Order):
        self.orders.remove(order)
