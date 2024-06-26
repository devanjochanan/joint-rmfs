class Order:
    def __init__(self, order_id, order_arrival):
        self.order_id = order_id
        self.order_arrival = order_arrival
        self.process_start_time = -1
        self.order_complete_time = -1
        self.station_id = None
        self.skus = {}

    def assign_station(self, station_id):
        self.station_id = station_id

    def add_sku(self, sku, total_quantity):
        self.skus[sku] = {
            'total_quantity': total_quantity,
            'quantity_committed': 0,
            'quantity_delivered': 0
        }

    def commit_quantity(self, sku, quantity):
        self.skus[sku]['quantity_committed'] += quantity

    def deliver_quantity(self, sku, quantity):
        self.skus[sku]['quantity_delivered'] += quantity
        self.skus[sku]['quantity_committed'] -= quantity

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

    def get_quantity_left_for_sku(self, sku):
        """Return the total quantity left to be delivered for the specified SKU, including committed quantities."""
        details = self.skus[sku]
        remaining = details['total_quantity'] - (details['quantity_delivered'] + details['quantity_committed'])
        return remaining

    def is_order_completed(self):
        """Check if all SKUs in the order have been delivered as per the total quantity."""
        return all(details['total_quantity'] == details['quantity_delivered'] for details in self.skus.values())

    def get_processing_time(self):
        """Calculate and return the total processing time from start to completion, if available."""
        return self.order_complete_time - self.process_start_time

    def __str__(self):
        return f"Order(order_id={self.order_id}, order_arrival={self.order_arrival}, process_start_time={self.process_start_time}, order_complete_time={self.order_complete_time}, station_id={self.station_id}, skus={self.skus})"

    def __repr__(self):
        return self.__str__()
