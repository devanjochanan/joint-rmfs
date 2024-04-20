from typing import List, Optional, Dict

from model.station import Station


class StationManager:
    def __init__(self):
        self.stations: List[Station] = []
        self.stations_by_id: Dict[int, Station] = {}

    def find_available_station(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.stations:
            if len(station.order_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.order_ids) < min_orders:
                    min_orders = len(station.order_ids)
                    available_station = station

        if available_station is not None:
            print(available_station.coordinate)
        return available_station

    def add_station(self, station: Station):
        self.stations.append(station)
        self.stations_by_id[station.station_id] = station

    def get_station_by_id(self, station_id: int):
        return self.stations_by_id[station_id]
