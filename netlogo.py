import csv
import pickle
import os
import traceback
from typing import List
import random


import networkx as nx
import pandas as pd
from pandas import DataFrame
import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans

from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from model.intersection import Intersection
from model.inventory import Inventory
from model.order import Order
from model.pod import Pod
from model.pod_manager import PodManager
from model.robot import Robot
from model.station import Station
from model.layout import Layout
from model.pod_generator import PodGenerator

from pip._internal import main as pipmain


class DirectedGraph:
    key = ''

    def __init__(self):
        """Initialize an instance with a directed graph."""
        self.graph = nx.DiGraph()

    @staticmethod
    def node_valid(node):
        """Check if a node is valid based on custom logic.

        Args:
            node (str): The node in format 'x,y'.

        Returns:
            bool: True if the node is valid, False otherwise.
        """
        x, y = map(int, node.split(","))
        return x >= 2 and y >= 0

    def add_node(self, node):
        """Add a node to the graph if it's valid.

        Args:
            node (str): The node to add.
        """
        if self.node_valid(node):
            self.graph.add_node(node)

    def add_edge(self, start, end, weight):
        """Add an edge between two nodes with a weight if both nodes are valid.

        Args:
            start (str): The start node.
            end (str): The end node.
            weight (float): The weight of the edge.
        """
        if self.node_valid(start) and self.node_valid(end):
            self.graph.add_edge(start, end, weight=weight)

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 10000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 10000


        for index, zone in enumerate(zone_boundary):
            for row in range(zone[1][0], zone[0][0]):
                for col in range(zone[0][1], zone[1][1]):
                    coordinate_str = f"{row},{col}"
                    for neighbor in list(G.neighbors(coordinate_str)) + list(G.predecessors(coordinate_str)):
                        # Increase the weight based on the zone
                        if G.has_edge(neighbor, coordinate_str):
                            G[neighbor][coordinate_str]['weight'] = penalties[index]
                        if G.has_edge(coordinate_str, neighbor):
                            G[coordinate_str][neighbor]['weight'] = penalties[index]

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None

    def dijkstra(self, start, end, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None


intersections: List[Intersection] = []

stations = [
    [2, 33],
    [2, 27],
    [2, 21],
    [2, 15],
    [2, 9],
    [2, 3],
]


def initPod(universe: Inventory):
    # Access the graphs from the universe object
    graph = universe.graph
    graph_pod = universe.graph_pod

    # Open and read the 'pod.csv' file
    with open('pod.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Initialize a counter for the rows (y-coordinate)
        line_count = 0

        for row in csv_reader:
            # Initialize the x-coordinate for the start of the row
            current_x = 0

            for cell_value in row:
                # Check if the cell indicates a Pod location
                if cell_value == '1':
                    # Create a new Pod object and set its position and coordinates
                    pod = Pod(1)
                    pod.pos_x = current_x - 1
                    pod.pos_y = line_count
                    pod.coordinate = NetLogoCoordinate(pod.pos_x, pod.pos_y)

                    # Add the Pod object to the universe
                    universe.addObject(pod)

                    # Construct the key for the current Pod based on its coordinates
                    obj_key = f"{pod.pos_x},{pod.pos_y}"

                    # Add nodes for the Pod in both graphs
                    graph.add_node(obj_key)
                    graph_pod.add_node(obj_key)

                    # Determine the key for the neighboring node based on the y-coordinate
                    obj_key_neighbor = f"{pod.pos_x},{pod.pos_y - 1}"
                    if (pod.pos_y + 1) % 3 == 0:
                        obj_key_neighbor = f"{pod.pos_x},{pod.pos_y + 1}"

                    # Add the neighboring node and edges between the Pod and its neighbor in both graphs
                    graph_pod.add_node(obj_key_neighbor)
                    graph.add_edge(obj_key, obj_key_neighbor, weight=1)
                    graph_pod.add_edge(obj_key, obj_key_neighbor, weight=1)
                    graph_pod.add_edge(obj_key_neighbor, obj_key, weight=1)

                # Move to the next x-coordinate
                current_x += 1

            # Limit the processing to the first 33 lines
            if line_count > 32:
                break

            # Move to the next row (y-coordinate)
            line_count += 1


def initStation(universe: Inventory):
    # Iterate over each station defined in the 'stations' list
    # Assuming 'stations' is a list of tuples/lists where each item contains the x and y coordinates of a station
    for s in stations:
        # Create a new Station object
        station = Station(1, "picker")

        # Set the x and y positions from the station data
        station.pos_x = s[0]
        station.pos_y = s[1]

        # Set the coordinates for the station using a helper function or class
        # NetLogoCoordinate may be a function or class designed to handle coordinate transformations or representations
        station.coordinate = NetLogoCoordinate(s[0], s[1])

        # Add the station object to the universe's list of objects
        # This could be for general object management within the universe
        universe.addObject(station)

        # Specifically add the station object to the universe's list of stations
        # This could be for easy access to stations or station-specific management
        universe.station_manager.add_station(station)


def initRobots(universe: Inventory):

    num_robot = 20
    
    robots = []
    x_range = (5,43)
    y_range=(0,30)

    # Initialize a set to keep track of used coordinates
    used_coordinates = set()

    # Generate the robots with random unique x and y coordinates
    while len(robots) < num_robot:
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        if (x, y) not in used_coordinates:
            robot = {
                'velocity': 0,
                'heading': 0,
                'x': x,
                'y': y
            }
            robots.append(robot)
            used_coordinates.add((x, y))

    # robots = [
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 9},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 9},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
    #     # {'velocity': 0, 'heading': 270, 'x': 28, 'y': 22},
    #     # {'velocity': 0, 'heading': 180, 'x': 45, 'y': 27},
    #     # {'velocity': 0, 'heading': 0, 'x': 48, 'y': 11},
    #     # {'velocity': 0, 'heading': 0, 'x': 46, 'y': 3},
    # ]

    # Iterate through each robot in the list to initialize and add to the universe
    for r in robots:
        # Create a new Robot instance
        robot = Robot()

        # Set the robot's attributes based on the dictionary values
        robot.velocity = r['velocity']
        robot.heading = r['heading']
        robot.pos_x = r['x']
        robot.pos_y = r['y']

        # Optionally, set the robot's coordinates using a specific coordinate system
        robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)

        # Add the robot to the universe, which likely involves adding it to some internal list or map
        universe.addObject(robot)


def draw_layout(universe):
    # Check if pod.csv exists in the current directory
    if os.path.exists('generated_pod.csv'):
        draw_layout_from_generated_file(universe)
    else:
        layout = Layout()
        layout.generate()
        draw_layout_from_generated_file(universe)


def draw_layout_from_generated_file(universe: Inventory):
    draw_storage_from_generated_file(universe)
    assign_skus_to_pods(universe.pod_manager)
    initRobots(universe)
    assign_backlog_orders(universe)

    pod = list(universe.pod_manager.coordinate_to_pods.values())[0]
    destinations = [
        [pod.pos_x, pod.pos_y, 0]
    ]

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
     
    return intersection / union  

def compute_jaccard_similarity(data):
    similarity_dict = {}
    grouped = data.groupby('order_id')['item_id'].apply(set)
    for order_dum, items in grouped.items():
        similarities = []
        for other_order_dum, other_items in grouped.items():
            if order_dum == other_order_dum:
                similarities.append(1.0)  # similarity with itself is 1
            else:
                similarity = jaccard_similarity(items, other_items)
                similarities.append(similarity)
        similarity_dict[order_dum] = similarities
    return grouped, similarity_dict

def cluster_backlog_orders(jaccard_similarities, total_station, station_capacity_df):
    jaccard_similarities_list = [similarities for similarities in jaccard_similarities.values()]
    # print(jaccard_similarities_list)
    cluster_labels = [-1] * len(jaccard_similarities_list)
    station_remaining_capacity = station_capacity_df['capacity_left'].tolist()

    # K-Means clustering
    kmeans = KMeans(n_clusters=total_station)
    kmeans.fit(jaccard_similarities_list)

    cluster_labels1 = kmeans.labels_

    cluster_distances = []

    # calculate distances for each order
    for i, label in enumerate(cluster_labels1):
        centroid = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(jaccard_similarities_list[i] - centroid)
        cluster_distances.append((i, label, distance))
        
    cluster_distances.sort(key=lambda x: x[2])

    # assign each backlog order to a cluster
    for order_idx, label, distance in cluster_distances:
        station_id = station_capacity_df.iloc[label]['id_station']
        if station_remaining_capacity[label] > 0:
            cluster_labels[order_idx] = station_id
            station_remaining_capacity[label] -= 1
        else:
            cluster_labels[order_idx] = None

    print("cluster label:")
    print(cluster_labels)

    return cluster_labels

def assign_cluster_labels(universe: Inventory, data_backlog_order_df, full_order, cluster_labels, station_capacity_df):
    order_dum_to_cluster = dict(zip(full_order.index, cluster_labels))
    # print("BACKLOG ORDER")
    # print(data_backlog_order_df)
    temp = float('inf')
    # assign cluster labels to the 'station' 
    new_order = None
 
    # print("data backlog ", data_backlog_order_df)
    orders_df = pd.read_csv('generated_order_new.csv')
    
    file_path = 'assign_order.csv'
    if os.path.exists(file_path):
        assign_order_df = pd.read_csv(file_path)
        # pass
    else:
        assign_order_df = orders_df.copy()
        assign_order_df['assigned_station'] = None
        assign_order_df['assigned_pod'] = None
        assign_order_df['status'] = -3
        assign_order_df.to_csv('assign_order.csv', index=False)      
    
    for index, row in data_backlog_order_df.iterrows():
        order_dum = row['order_id']
        if(temp != order_dum or order_dum == -1):
            # if(order_dum == -1):
            #     print("test")
            if(temp != float('inf')):
                
                new_order.station_id = station_id
                print("order: ", new_order.order_id)
                print("station: ", station_id)
                assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'assigned_station'] = station_id
                if(new_order.station_id is not None):
                    assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'status'] = -1
                assign_order_df.to_csv('assign_order.csv', index=False)  
                
                if station_id is not None:
                    station = universe.station_manager.get_station_by_id(station_id)
                    station.add_order(new_order.order_id)

                universe.order_manager.add_order(new_order)

            new_order = Order(order_dum, 0)
            temp = order_dum
 
        new_order.add_sku(row['item_id'], row['item_quantity'])
        station_id = order_dum_to_cluster[order_dum]
    
  
    return station_capacity_df

def assign_backlog_orders(universe: Inventory):
    # order = Order("backlog", 0)
    # order.add_sku(1, 10)
    # universe.order_manager.add_order(order)

    # open file order
    order_path = "generated_order_new.csv"
    data_order_df = pd.read_csv(order_path)
    # filter order_id < 0
    unassigned_backlog_order = data_order_df.loc[(data_order_df['order_id'] < 0)].sort_values(by=['order_id']).reset_index(drop=True)

    columns = ['id_station', 'capacity_left']
    station_id_cap_df = pd.DataFrame(columns=columns)

    for station in universe.station_manager.stations:
        # mask = station.station_id.str.contains(r'^picker-\d+$', regex=True)
        # if(mask):
        id = station.station_id
        cap = station.max_orders - len(station.order_ids)

        station_id_cap_df = station_id_cap_df.append({'id_station': id, 'capacity_left': cap}, ignore_index=True)

    is_picker = station_id_cap_df['id_station'].str.startswith('picker')

    station_id_cap_df = station_id_cap_df[is_picker]
    station_id_cap_df.reset_index(drop=True, inplace=True)

    if len(unassigned_backlog_order) > 0:
        total_station = len(station_id_cap_df)

        full_order, jaccard_similarities = compute_jaccard_similarity(unassigned_backlog_order)

        cluster_labels = cluster_backlog_orders(jaccard_similarities, total_station, station_id_cap_df)

        station_id_cap_df = assign_cluster_labels(universe, unassigned_backlog_order, full_order, cluster_labels, station_id_cap_df)



def draw_storage_from_generated_file(universe: Inventory):
    station_picker_counter = 1
    station_replenish_counter = 1
    pods_horizontal_length = 5
    pods_vertical_length = 2
    pod_counter = 0
    graph = DirectedGraph()
    graph_pod = DirectedGraph()
    graph_pod.key = 'pod'
    universe.graph = graph
    universe.graph_pod = graph_pod
    data = pd.read_csv("generated_pod.csv", header=None)
    total_rows = len(data)
    total_cols = 0
    for y, row in data.iterrows():
        # Invert Y only to draw
        for x, value in row.items():
            obj = Object()
            obj.object_type = 'way-direction'
            obj_key = f"{x},{y}"
            obj.pos_x = x
            obj.pos_y = y

            obj_left_coordinate = f"{x - 1},{y}"
            obj_right_coordinate = f"{x + 1},{y}"
            obj_above_coordinate = f"{x},{y - 1}"
            obj_below_coordinate = f"{x},{y + 1}"

            obj_left_value = data.iloc[y, x - 1] if x > 0 else None
            obj_right_value = data.iloc[y, x + 1] if x < len(row) - 1 else None
            obj_above_value = data.iloc[y - 1, x] if y > 0 else None
            obj_below_value = data.iloc[y + 1, x] if y < total_rows - 1 else None

            weight = 1
            if x <= 7:
                weight = 3

            if value == 0 or value == 1 or value == 2:
                add_all_direction_paths(graph, obj_key, weight)

                if value == 0:
                    obj.shape = 'empty-space'
                elif value == 1:
                    obj = Pod(pod_counter)
                    pod_counter += 1
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.pos_x = x
                    obj.pos_y = y

                    graph_pod.add_node(obj_key)
                    universe.pod_manager.add_pod(obj)
                elif value == 2:
                    obj.shape = 'square 2'

                if obj_left_value != 1:
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                if obj_right_value != 1:
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
                if obj_above_value != 1:
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                if obj_below_value != 1:
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 3:
                obj.shape = 'empty-space'

                if obj.pos_x == 15 and (obj.pos_y == 15 or obj.pos_y == 12):
                    intersection = Intersection(NetLogoCoordinate(x, y))
                    approaching_path_coordinates = []

                    if obj_right_value == 4:
                        for right_x in range(x + 1, (x + pods_horizontal_length) + 1):
                            approaching_path_coordinates.append((right_x, y))
                    if obj_left_value == 5:
                        for left_x in range(x - 1, (x - pods_horizontal_length) - 1, - 1):
                            approaching_path_coordinates.append((left_x, y))
                    if obj_below_value == 6:
                        for below_y in range(y + 1, (y + pods_vertical_length) + 1):
                            approaching_path_coordinates.append((x, below_y))
                    if obj_above_value == 7:
                        for above_y in range(y - 1, (y - pods_vertical_length) - 1, - 1):
                            approaching_path_coordinates.append((x, above_y))

                    for each_approaching_coordinate in approaching_path_coordinates:
                        intersection.approaching_path_coordinates.append(each_approaching_coordinate)

                    universe.intersection_manager.add_intersection(intersection)

                if obj_left_value == 4 or obj_right_value == 4:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
                elif obj_left_value == 5 or obj_right_value == 5:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)

                if obj_above_value == 6 or obj_above_value == 6:
                    graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
                elif obj_below_value == 7 or obj_below_value == 7:
                    graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)

                if obj_left_value == 6 or obj_left_value == 7:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
                elif obj_right_value == 6 or obj_right_value == 7:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 4:
                obj.shape = 'arrow-left'
                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 5:
                obj.shape = 'arrow-right'
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 6:
                obj.shape = 'arrow-up'
                graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 7:
                obj.shape = 'arrow-down'
                graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 11 or value == 21:
                obj.shape = 'person-red'
            elif value == 12 or value == 23:
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 13 or value == 22:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 14 or value == 24:
                if obj_left_value == 11:
                    obj = Station(station_picker_counter, "picker")
                    station_picker_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.short_path = construct_station_path(data, x, y)
                    obj.long_path = construct_station_path(data, x, y, short_path=False)
                    universe.station_manager.add_station(obj)
                elif obj_right_value == 21:
                    obj = Station(station_replenish_counter, "replenishment")
                    station_replenish_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    # obj.path = construct_station_path(data, x, y)
                    universe.station_manager.add_station(obj)

                obj.shape = 'rail-triangle'
                if value == 14:
                    obj.heading = 270
                elif value == 24:
                    obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 16:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 17:
                obj.shape = 'rail-corner'
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 18:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 19:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 26:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 27:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)
            elif value == 28:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 29:
                obj.shape = 'rail-corner'
                obj.heading = 0
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 99:
                obj.shape = 'empty-space'
            else:
                continue

            if obj_left_coordinate == 13:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

            obj.pos_x = x
            obj.pos_y = y
            total_cols += 1
            universe.addObject(obj)

    universe.set_warehouse_size([total_rows, total_cols])

def construct_station_path(data: DataFrame, start_x, start_y, short_path=True):
    station_path: List[NetLogoCoordinate] = [NetLogoCoordinate(start_x, start_y)]

    if not short_path:
        station_path.insert(0, NetLogoCoordinate(start_x + 1, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2, start_y + 1))
        station_path.insert(0, NetLogoCoordinate(start_x + 1, start_y + 1))

    # go to bottom
    y, x = start_y + 1, start_x
    while data.iloc[y, x] in (14, 17):
        station_path.insert(0, NetLogoCoordinate(x, y))

        if data.iloc[y, x] == 17:
            x += 1
            while data.iloc[y, x] == 13:
                station_path.insert(0, NetLogoCoordinate(x, y))
                x += 1

        y += 1

    return station_path


def add_all_direction_paths(graph, obj_key, weight):
    x, y = map(int, obj_key.split(','))
    directions = {
        'left': (x - 1, y),
        'right': (x + 1, y),
        'up': (x, y + 1),
        'down': (x, y - 1)
    }

    for dir_key, (nx, ny) in directions.items():
        neighbor_key = f"{nx},{ny}"
        graph.add_edge(obj_key, neighbor_key, weight=weight)


def assign_skus_to_pods(pod_manager):
    # Check if pod.csv exists in the current directory
    if os.path.exists('pods.csv'):
        assign_skus_to_pods_from_file(pod_manager)
    else:
        PodGenerator(pod_manager).generate()
        assign_skus_to_pods_from_file(pod_manager)


def assign_skus_to_pods_from_file(pod_manager: PodManager):
    with open('pods.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pod_id = int(row['pod_id'])
            sku = int(row['item'])
            limit_qty = int(row['max_qty'])
            current_qty = int(row['qty'])
            # threshold = int(row['threshold'])
            threshold = 5

            # Find the pod by id
            pod: Pod = pod_manager.get_pod_by_id(pod_id)
            pod.add_sku(sku, limit_qty=limit_qty, current_qty=current_qty, threshold=threshold)
            pod_manager.add_sku_to_pod(sku, pod)


def setup():
    try:
        # Initialize the simulation universe
        universe = Inventory()

        # Populate the universe with objects and connections
        draw_layout(universe)

        # Set simulation parameters
        universe.tick_to_second = 0.15
        # print(universe.intersection_manager.intersections[0].intersection_coordinate)

        # Generate initial results
        next_result = universe.generateResult()

        # Save the universe state for future ticks
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(universe, config_dictionary_file)

        return next_result

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def tick():
    try:
        # print("========tick========")

        # Load the simulation state
        with open('netlogo.state', 'rb') as file:
            universe: Inventory = pickle.load(file)

        print("before tick", universe._tick)

        # Update each object with the current universe context
        for _n in universe._objects:
            _n.setUniverse(universe)

        # Perform a simulation tick
        universe.tick()

        # Generate results after the tick
        next_result = universe.generateResult()
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(universe, config_dictionary_file)
        return [next_result, universe.total_energy, len(universe.job_queue), universe.stop_and_go,
                universe.total_turning]
    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def setup_py():
    def install_package(package_name):
        """Install a Python package using pip."""
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)
# setup()
# x = setup()
# print(x)
# x = tick()
# print(x)
