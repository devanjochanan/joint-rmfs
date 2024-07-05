# import os
# import pandas as pd
# import numpy as np
# from sku import SKU
# from pod_manager import PodManager

# script_dir = os.path.dirname(os.path.abspath(__file__))
# pods_csv_path = os.path.join(script_dir, '..', 'pods.csv')

# def start_initial_sku():
#     df = pd.read_csv(pods_csv_path)
#     item_in_pod = df[['item', 'qty', 'max_qty']]

#     total_current_qty_df = item_in_pod.groupby('item')['qty'].sum().reset_index()
#     total_current_qty_df.rename(columns={'qty': 'total_current_qty_in_WH'})

#     total_max_qty_df = item_in_pod.groupby('item')['max_qty'].sum()().reset_index()
#     total_max_qty_df.rename(columns={'qty': 'total_max_qty_in_WH'})
#     merged_df = pd.merge(total_current_qty_df, total_max_qty_df, on='item')

#     sku_objects = []

#     for _, row in merged_df.iterrows():
#         item_id = row['item']
#         total_current_qty = row['total_current_qty']
#         total_max_qty = row['total_max_qty_in_WH']
#         global_inventory_level = total_current_qty / total_max_qty

#         sku = SKU(sku_id=item_id, global_inv_level=global_inventory_level, pod_inv_level=0)
#         sku.current_global_qty = total_current_qty
#         sku.max_global_qty = total_max_qty

#         sku_objects.append(sku)
#     for sku in sku_objects:
#         print(f'SKU ID: {sku.id}, Current Global Qty: {sku.current_global_qty}, Max Global Qty: {sku.max_global_qty}, Global Inventory Level: {sku.global_inv_level}')

#     return sku_objects

# def check_global_inventory_replenish(sku: SKU, minimum_qty):
#     if(sku.global_inv_level < minimum_qty):
#         return True
#     else:
#         return False
    
#     # Iterate all the pod in the Pod Manager
#     # Iterate all the sku inevery pod
#     # Check the global inventory level
#     # Check each level of the item in the pod and measure total of them is below alpha or not, if below == replenish the pod