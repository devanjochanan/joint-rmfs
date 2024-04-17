class Order:
    def __init__(self, pod):
        self.designated_pod = pod
        self.coordinate = None
        self.station = None
        self.skus = {}

    def add_sku(self, sku, quantity):
        self.skus[sku] = {
            'quantity': quantity,
            'picked': False
        }

    def mark_sku_picked(self, sku):
        self.skus[sku]['picked'] = True
