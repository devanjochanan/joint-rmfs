from engine.object import Object


class Station(Object):
    def __init__(self):
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        self.orders = []
        self.max_orders = 5
        super().__init__()

    def add_order(self, order):
        self.orders.append(order)
