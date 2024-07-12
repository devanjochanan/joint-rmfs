from typing import List

from model.pod import Pod
from engine import NetLogoCoordinate

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

class PodManager:
    def __init__(self):
        self.pods: List[Pod] = []
        self.id_to_pod = {}
        self.sku_to_pods = {}
        self.coordinate_to_pods = {}
        self.skus_data = {}

    def add_pod(self, pod: Pod):
        self.pods.append(pod)
        self.coordinate_to_pods[(pod.pos_x, pod.pos_y)] = pod
        self.id_to_pod[pod.pod_id] = pod

        for sku in pod.skus:
            if sku not in self.sku_to_pods:
                self.sku_to_pods[sku] = []
            self.sku_to_pods[sku].append(pod)

    def add_sku_to_pod(self, sku: int, pod: Pod):
        if sku not in self.sku_to_pods:
            self.sku_to_pods[sku] = []
        self.sku_to_pods[sku].append(pod)

    def add_sku_data(self,sku,current_qty,max_qty,global_threshold_inv_level):
        sku_id = sku

        if sku_id not in self.skus_data:
            self.skus_data[sku_id] = {
                'current_global_qty': current_qty,
                'max_global_qty': max_qty,
                'global_inv_level': (current_qty / max_qty),
                'global_threshold_inv_level' : global_threshold_inv_level
            }
        else:
            self.skus_data[sku_id]['current_global_qty'] += current_qty
            self.skus_data[sku_id]['max_global_qty'] += max_qty
            self.skus_data[sku_id]['global_inv_level'] = self.skus_data[sku_id]['current_global_qty'] / self.skus_data[sku_id]['max_global_qty']

    def reduce_sku_data(self,sku,quantity):
         if sku in self.skus_data:
            self.skus_data[sku]['current_global_qty'] -= quantity
            self.skus_data[sku]['global_inv_level'] = self.skus_data[sku]['current_global_qty'] / self.skus_data[sku]['max_global_qty']

    def get_all_skus_data(self):
        return self.skus_data
    
    def is_sku_need_replenished(self, sku_id):
        if float(self.skus_data[sku_id]['global_inv_level']) <= float(self.skus_data[sku_id]['global_threshold_inv_level']):
            return sku_id, True
        else:
            return sku_id, False

    def get_pod_need_replenished_by_sku(self, list_of_sku):
        replenished_pod_needed_every_sku = {}

        for sku in list_of_sku:
            check_pod: List[Pod] = self.sku_to_pods[sku]
            replenished_pod_needed_every_sku[sku] = check_pod
        
        return replenished_pod_needed_every_sku
    
    def determine_pod_will_replenished(self, replenished_pod_needed_by_sku):
        stock_out_probability_of_each_pod = {}

        all_pods = sum(replenished_pod_needed_by_sku.values(), [])
        unique_pods = set(all_pods)
        unique_pods_list = list(unique_pods)

        for pod in unique_pods_list: 
            skus_in_pod = pod.skus
            pod_stock_out_probability = 0
            for sku in skus_in_pod:
                sku_current_qty = sku['current_qty']
                # Get the max amount of the SKU that have been ordered
                # Sum the probability from the probability of sku_current_qty until probability of max qty in the sku
            # Put in the result of the sum probability of each pod to the stock_out_probability_of_each_pod
        # Return the pod.pod_id with the highest value of stock_out_probability_of_each_pod
        
    def get_available_pod(self, sku: str):
        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                if pod.is_idle is True:
                    return pod

    def get_available_pod_similarity(self, sku: str, skus_in_station, station_coordinate, robots_coordinate):
        # If SKU is available
        sku_in_station_list = [i for i in skus_in_station]
        pod_available_for_multiple_items = pd.DataFrame(columns=["pod_id", "similarity_score", "distance_to_station", "distance_to_robot"])
        
        station_coordinate = [station_coordinate.x, station_coordinate.y]

        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                similarity_score = 1
                if pod.is_idle is True:
                    pod_skus = [i for i in pod.skus]
                    pod_skus_in_station_skus_mask = np.isin(sku_in_station_list, pod_skus)
                    pod_skus_in_station_skus = np.array(sku_in_station_list)[pod_skus_in_station_skus_mask]
                    
                    if len(pod_skus_in_station_skus) > 0:
                        for skus in pod_skus_in_station_skus:
                            skus_qty_in_pod = pod.get_quantity(skus)
                            if skus_qty_in_pod > 0:
                                similarity_score += 1
                    
                    pod_coordinate = [pod.coordinate.x, pod.coordinate.y]
                    distance = manhattan_distances([pod_coordinate],[station_coordinate])[0][0]
                    distance_to_robot = self._distance_pod_to_robot(pod_coordinate, robots_coordinate)
                    
                    pod_available_for_multiple_items = pd.concat([pod_available_for_multiple_items, 
                                                                pd.DataFrame([[pod.pod_id, similarity_score, distance, distance_to_robot]], 
                                                                                                            columns=["pod_id", 
                                                                                                                    "similarity_score", 
                                                                                                                    "distance_to_station", "distance_to_robot"])], ignore_index=True) 
            pod_available_for_multiple_items["distance_score"] = pod_available_for_multiple_items["distance_to_station"].max() - pod_available_for_multiple_items["distance_to_station"] + pod_available_for_multiple_items["distance_to_robot"].max() - pod_available_for_multiple_items["distance_to_robot"]
            pod_available_for_multiple_items.sort_values(by=["similarity_score", "distance_score"], ascending=[False, False], inplace=True)
            pod_available_for_multiple_items.reset_index(drop=True, inplace=True)
            pod_available_for_multiple_items = pod_available_for_multiple_items[pod_available_for_multiple_items["similarity_score"] > 0]

            assigned_pod = None
            if len(pod_available_for_multiple_items) > 0:
                assigned_pod_id = pod_available_for_multiple_items.loc[0, "pod_id"]
           
                assigned_pod = self.get_pod_by_id(assigned_pod_id)
        
            return assigned_pod
    

    def get_available_pod_inventory(self, sku: str, skus_in_station_dict, station_coordinate, robots_coordinate):
        sku_in_station_list = [i for i in skus_in_station_dict]
        pod_available_for_multiple_items = pd.DataFrame(columns=["pod_id", "similarity_score", "inventory_score","distance_to_station","distance_to_robot"])
        
        station_coordinate = [station_coordinate.x, station_coordinate.y]
        # print("THE SKU ", sku)
        # print(skus_in_station_dict)
        if sku in self.sku_to_pods:
            # a = self.sku_to_pods[sku]
            # print("len of available pod ", len(a))
            for pod in self.sku_to_pods[sku]:
                similarity_score = 0

                if pod.is_idle is True:
                    # Similarity
                    pod_skus = [i for i in pod.skus]
                    pod_skus_in_station_skus_mask = np.isin(sku_in_station_list, pod_skus)
                  
                    pod_skus_in_station_skus = np.array(sku_in_station_list)[pod_skus_in_station_skus_mask]
                    
                    if len(pod_skus_in_station_skus) > 0:
                        for skus in pod_skus_in_station_skus:
                            skus_qty_in_pod = pod.get_quantity(skus)
                            if skus_qty_in_pod > 0:
                                similarity_score += 1
                    
                    pod_coordinate = [pod.coordinate.x, pod.coordinate.y]
                    # D1
                    distance_to_station = manhattan_distances([pod_coordinate],[station_coordinate])[0][0]
                    # D2
                    distance_to_robot = self._distance_pod_to_robot(pod_coordinate, robots_coordinate)
                    # distance_to_robot = 1
                    # Inventory Score
                    # print("sku in dict, gabisa keknya")
                    # print(skus_in_station_dict)
                    inventory_score = self._count_fulfillment(skus_in_station_dict, pod.skus)
                    # inventory_score = 1
                    pod_available_for_multiple_items = pd.concat([pod_available_for_multiple_items, 
                                                                pd.DataFrame([[pod.pod_id, similarity_score,inventory_score, distance_to_station, distance_to_robot]], 
                                                                                                            columns=["pod_id", "similarity_score", "inventory_score","distance_to_station","distance_to_robot"])], ignore_index=True) 
            
            pod_available_for_multiple_items["station_distance_score"] = pod_available_for_multiple_items["distance_to_station"].max() - pod_available_for_multiple_items["distance_to_station"]
            pod_available_for_multiple_items["cost"] = (pod_available_for_multiple_items["station_distance_score"] + pod_available_for_multiple_items["distance_to_robot"]) * pod_available_for_multiple_items["similarity_score"] * (len(sku_in_station_list) / pod_available_for_multiple_items["inventory_score"]) 
            pod_available_for_multiple_items.sort_values(by=["cost"], ascending=[True], inplace=True)
            pod_available_for_multiple_items.reset_index(drop=True, inplace=True)
            pod_available_for_multiple_items = pod_available_for_multiple_items[pod_available_for_multiple_items["similarity_score"] > 0]

            assigned_pod = None
            if len(pod_available_for_multiple_items) > 0:
                assigned_pod_id = pod_available_for_multiple_items.loc[0, "pod_id"]
           
                assigned_pod = self.get_pod_by_id(assigned_pod_id)
        
            return assigned_pod

        return
    
    def _distance_pod_to_robot(self, pod_coordinate, robots_coordinate):
        pod_coordinate = np.array(pod_coordinate).reshape(1, -1)
        distance_to_robot_score = 1000
        robots_coordinate = np.array(robots_coordinate)
        if len(robots_coordinate) == 0:
            return distance_to_robot_score

        distances = manhattan_distances(pod_coordinate, robots_coordinate)
        distance_to_robot_score = np.argmin(distances)
        
        return distance_to_robot_score
    
    def _count_fulfillment(self, skus_in_station_dict, pod_skus):
        total_fulfillment = 1
        pod_skus_copy = pod_skus.copy()
        for sku in skus_in_station_dict:
            for order_qty in skus_in_station_dict[sku]:
                if sku in pod_skus_copy and pod_skus_copy[sku]["current_qty"] >= order_qty:
                    pod_skus_copy[sku]["current_qty"] -= order_qty
                    total_fulfillment += 1
                else: 
                    continue

        return total_fulfillment

    def _count_similarity():
        return
    

    def mark_pod_not_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        pod.is_idle = False

    def mark_pod_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        pod.is_idle = True

    def get_pods_by_sku(self, sku):
        return self.sku_to_pods.get(sku, None)

    def get_pod_by_coordinate(self, x, y):
        return self.coordinate_to_pods.get((x, y), None)

    def get_pod_by_id(self, pod_id):
        return self.id_to_pod.get(pod_id, None)

