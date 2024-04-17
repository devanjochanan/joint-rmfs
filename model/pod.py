from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate


class Pod(Object):
    def __init__(self):
        self.shape = 'full square'
        self.object_type = 'pod'
        self.coordinate = NetLogoCoordinate()
        self.skus = {}
        super().__init__()

    def add_sku(self, sku, limit_qty, current_qty, threshold):
        """Add a new SKU with its limit, current quantity, and threshold."""
        self.skus[sku] = {
            'limit_qty': limit_qty,
            'current_qty': current_qty,
            'threshold': threshold
        }

    def check_replenishment_needed(self):
        """Check if 50% or more SKUs are below their threshold to determine if the pod needs to move to a
        replenishment station."""
        count_below_threshold = 0
        total_skus = len(self.skus)
        for details in self.skus.values():
            if details['current_qty'] <= details['threshold']:
                count_below_threshold += 1

        if count_below_threshold >= total_skus / 2:
            return True
        return False

    def replenish_all_skus(self):
        """Replenish all SKUs by setting each SKU's current quantity to its limit quantity."""
        for sku in self.skus:
            self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']

    def pick_sku(self, sku, qty):
        self.skus[sku]['current_qty'] -= qty

    def get_unassigned_skus(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details['pod'] is None]
        return unassigned_skus
