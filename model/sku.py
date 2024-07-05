# class SKU:
#     def __init__(self, item_id: int, global_inv_level: float, pod_inv_level: float):
#         self.id = item_id
#         self.global_inv_level = global_inv_level
#         self.pod_inv_level = None
#         self.current_global_qty = None
#         self.max_global_qty = None

#     def update_max_global_qty(self, new_max_global_qty):
#         self.max_global_qty = new_max_global_qty
#         return 1
    
#     def reduce_current_global_qty(self, new_current_global_qty):
#         self.current_global_qty =- new_current_global_qty
#         return 1
    
#     def add_current_global_qty(self, new_current_global_qty):
#         self.current_global_qty =+ new_current_global_qty
#         return 1
