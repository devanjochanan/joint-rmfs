from typing import List, Optional, Dict

from model.station import Station
from .pod_manager import PodManager
import pandas as pd
import numpy as np


class StationManager:
    def __init__(self):
        self.stations: List[Station] = []
        self.picking_stations: List[Station] = []
        self.replenishment_stations: List[Station] = []
        self.stations_by_id: Dict[int, Station] = {}

    def find_available_picking_station(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.order_ids) < min_orders:
                    min_orders = len(station.order_ids)
                    available_station = station

        return available_station

    def find_highest_similarity_station(self, skus_in_order, pod_manager: PodManager) -> Optional[Station]:
        available_station_rank = pd.DataFrame(columns=["station_id", "similarity_score"])
        sku_in_order_list = [i for i in skus_in_order]
        available_station = []
        assign_station = None

        # Store all available station
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                available_station.append(station)
        
        # Check if more than one station is available
        if len(available_station) > 1:
            for station in available_station:
                # Check Available Station
                similarity_score = 0
                if len(station.order_ids) < station.max_orders:
                    # Take pod assigned to this particular station
                    station_incoming_pod = station.incoming_pod
                    station_pod_skus_set = set()
                    for pod_id in station_incoming_pod:
                        pod  = pod_manager.get_pod_by_id(pod_id)
                        pod_skus = [item for item, details in pod.skus.items() if details['current_qty'] > 0]
                        station_pod_skus_set.update(pod_skus)

                    station_pod_skus_list = list(station_pod_skus_set)
                    station_pod_skus_in_order_mask = np.isin(sku_in_order_list, station_pod_skus_list)
                    station_pod_skus_in_order = np.array(sku_in_order_list)[station_pod_skus_in_order_mask]
                    similarity_score = len(station_pod_skus_in_order)

                    available_station_rank = pd.concat([available_station_rank , 
                                                pd.DataFrame([[station.station_id, similarity_score]], columns=["station_id", "similarity_score"])], ignore_index=True) 
            
            available_station_rank.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            available_station_rank.reset_index(drop=True, inplace=True)

            if len(available_station_rank) > 0:
                assign_station_id = available_station_rank.loc[0, "station_id"]
                assign_station = self.get_station_by_id(assign_station_id)
        elif len(available_station) == 1:
            assign_station = available_station[0]

        return assign_station

    def add_station(self, station: Station):
        self.stations.append(station)
        self.stations_by_id[station.station_id] = station

        if station.is_picker_station():
            self.picking_stations.append(station)
        elif station.is_replenishment_station():
            self.replenishment_stations.append(station)

    def get_station_by_id(self, station_id):
        return self.stations_by_id[station_id]
