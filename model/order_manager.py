from typing import List, Dict, Optional

from model.order import Order


class OrderManager:
    def __init__(self):
        self.orders: List[Order] = []
        self.order_id_to_order: Dict[int, Order] = {}
        # self.finished_orders: List[Order] = []
        self.unfinished_orders: List[Order] = []
        self.preassign_order_ids: List[int] = []
        # Orders become scientifically "released" when they enter this active
        # manager, not when their rows are merely present in an input CSV.
        self.orders_released_count: int = 0
        self.order_lines_released_count: int = 0

    def add_order(self, order: Order):
        self.orders.append(order)
        self.order_id_to_order[order.order_id] = order
        self.unfinished_orders.append(order)
        self.orders_released_count += 1

    def record_order_line_released(self) -> None:
        """Record one line after it has joined an already-released order."""
        self.order_lines_released_count += 1

    def get_order_by_id(self, order_id) -> Optional[Order]:
        """Retrieve an order by its ID using the dictionary for quick access."""
        return self.order_id_to_order.get(order_id, None)

    def remove_order(self, order:Order):
        self.orders.remove(order)

    def finish_order(self, order_id, tick: int):
        """Move an order from the unfinished_orders list to the finished_orders list."""
        order = self.get_order_by_id(order_id)
        order.complete_order(tick)
        if order and order in self.unfinished_orders:
            self.unfinished_orders.remove(order)
            # self.finished_orders.append(order)
