from typing import List, Optional

from model.station import Station


class StationManager:
    def __init__(self):
        self.stations: List[Station] = []
        self.max_orders = 5

    def find_available_station(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.stations:
            if len(station.orders) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.orders) < min_orders:
                    min_orders = len(station.orders)
                    available_station = station

        return available_station

    def add_station(self, station: Station):
        self.stations.append(station)
