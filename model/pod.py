from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate


class Pod(Object):
    def __init__(self, pod_id: int):
        self.pod_id = pod_id
        self.shape = 'full square'
        self.object_type = 'pod'
        self.coordinate = NetLogoCoordinate()
        self.skus = {}
        self.is_idle = True
        self.station = None
        self.need_replenishment = False
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Pod):
            return self.pod_id == other.pod_id
        return False

    def __hash__(self):
        return hash(self.pod_id)

    def __repr__(self):
        return f"Pod({self.pod_id})"

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
        alpha = total_skus / 2
        for details in self.skus.values():
            # print(f"crt {details['current_qty']} limit {details['limit_qty']} th {details['threshold']}")
            if float(details['current_qty'])/float(details['limit_qty']) <= float(details['threshold']):
                count_below_threshold += 1

        if count_below_threshold >= alpha:
            return True
        return False

    def replenish_all_skus(self):
        """Replenish all SKUs by setting each SKU's current quantity to its limit quantity."""
        for sku in self.skus:
            self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']

    def pick_sku(self, sku, qty):
        self.skus[sku]['current_qty'] -= qty

    def get_quantity(self, sku):
        return self.skus[sku]['current_qty']

    def get_unassigned_skus(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details['pod'] is None]
        return unassigned_skus

    def set_pod_station(self, station):
        self.station = station
        return
    
    def remove_pod_station(self):
        self.station = None
        return

    def get_skus_in_pod(self):
        return self.skus