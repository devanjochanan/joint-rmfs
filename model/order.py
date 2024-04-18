class Order:
    def __init__(self, order_id, order_arrival):
        self.order_id = order_id
        self.order_arrival = order_arrival
        self.process_start_time = 0
        self.order_complete_time = 0
        self.coordinate = None
        self.station = None
        self.skus = {}

    def assign_station(self, station):
        self.station = station

    def add_sku(self, sku, total_quantity):
        self.skus[sku] = {
            'total_quantity': total_quantity,
            'quantity_committed': 0,
            'quantity_delivered': 0
        }

    def commit_quantity(self, sku, quantity):
        self.skus[sku]['quantity_committed'] += quantity

    def deliver_quantity(self, sku, quantity):
        self.skus[sku]['quantity_committed'] -= quantity
        self.skus[sku]['quantity_delivered'] += quantity

    def get_remaining_skus(self):
        """Return a dictionary of SKUs with their remaining quantities to be fulfilled."""
        remaining_skus = {
            sku: details['total_quantity'] - (details['quantity_delivered'] + details['quantity_committed'])
            for sku, details in self.skus.items()
            if details['total_quantity'] > (details['quantity_delivered'] + details['quantity_committed'])
        }
        return remaining_skus

    def start_processing(self, start_time):
        """Record the start time for order processing."""
        self.process_start_time = start_time

    def complete_order(self, complete_time):
        """Record the time when order processing is completed."""
        self.order_complete_time = complete_time

    def is_order_completed(self):
        """Check if all SKUs in the order have been delivered as per the total quantity."""
        return all(details['total_quantity'] == details['quantity_delivered'] for details in self.skus.values())

    def get_processing_time(self):
        """Calculate and return the total processing time from start to completion, if available."""
        return self.order_complete_time - self.process_start_time
