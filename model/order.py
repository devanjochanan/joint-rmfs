class Order:
    def __init__(self, pod):
        self.designated_pod = pod
        self.coordinate = None
        self.station = None
        self.skus = {}

    def assign_station(self, station):
        self.station = station

    def add_sku(self, sku, total_quantity):
        self.skus[sku] = {
            'total_quantity': total_quantity,
            'quantity_committed': 0
        }

    def commit_quantity(self, sku, quantity):
        self.skus[sku]['quantity_committed'] += quantity

    def get_remaining_skus(self):
        """Return a dictionary of SKUs with their remaining quantities to be fulfilled."""
        remaining_skus = {
            sku: details['total_quantity'] - details['quantity_committed']
            for sku, details in self.skus.items()
            if details['total_quantity'] > details['quantity_committed']
        }
        return remaining_skus
