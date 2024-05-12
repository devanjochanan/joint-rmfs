import csv
import pickle
import os
import traceback
from typing import List

import networkx as nx
import pandas as pd
from pandas import DataFrame

from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
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


intersections = []

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
    robots = [
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 9},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 9},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        {'velocity': 0, 'heading': 0, 'x': 42, 'y': 5},
        # {'velocity': 0, 'heading': 270, 'x': 28, 'y': 22},
        # {'velocity': 0, 'heading': 180, 'x': 45, 'y': 27},
        # {'velocity': 0, 'heading': 0, 'x': 48, 'y': 11},
        # {'velocity': 0, 'heading': 0, 'x': 46, 'y': 3},
    ]

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


def assign_backlog_orders(universe: Inventory):
    order = Order("backlog", 0)
    order.add_sku(1, 10)
    universe.order_manager.add_order(order)


def draw_storage_from_generated_file(universe: Inventory):
    station_picker_counter = 1
    station_replenish_counter = 1
    pod_counter = 1
    graph = DirectedGraph()
    graph_pod = DirectedGraph()
    graph_pod.key = 'pod'
    universe.graph = graph
    universe.graph_pod = graph_pod
    data = pd.read_csv("generated_pod.csv", header=None)
    total_rows = len(data)
    for y, row in data.iterrows():
        # Invert Y only to draw
        for x, value in row.items():
            obj = Object()
            obj.object_type = 'way-direction'
            obj_key = f"{x},{y}"

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

            if value == 0 or value == 1:
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
                intersections.append([obj.pos_x, obj.pos_y])

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
                    obj.path = construct_station_path(data, x, y)
                    universe.station_manager.add_station(obj)
                elif obj_right_value == 21:
                    obj = Station(station_replenish_counter, "replenishment")
                    station_replenish_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    # obj.path = construct_station_path(data, x, y)
                    universe.station_manager.add_station(obj)

                obj.shape = 'rail'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 16:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 17:
                obj.shape = 'rail-corner'
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 26:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 27:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)
            elif value == 99:
                obj.shape = 'empty-space'
            else:
                continue

            if obj_left_coordinate == 13:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

            obj.pos_x = x
            obj.pos_y = y
            universe.addObject(obj)


def construct_station_path(data: DataFrame, start_x, start_y):
    station_path: List[NetLogoCoordinate] = [NetLogoCoordinate(start_x, start_y)]
    vertical_path = 14
    horizontal_entrance_path = 13
    horizontal_exit_path = 12
    entrance_corner = 17
    exit_corner = 16

    # go to top
    y, x = start_y - 1, start_x
    while data.iloc[y, x] == vertical_path or data.iloc[y, x] == exit_corner:
        station_path.append(NetLogoCoordinate(x, y))

        if data.iloc[y, x] == exit_corner:
            x += 1
            while data.iloc[y, x] == horizontal_exit_path:
                station_path.append(NetLogoCoordinate(x, y))
                x += 1

        y -= 1

        # go to bottom
        y, x = start_y + 1, start_x
        while data.iloc[y, x] == vertical_path or data.iloc[y, x] == entrance_corner:
            station_path.insert(0, NetLogoCoordinate(x, y))

            if data.iloc[y, x] == entrance_corner:
                x += 1
                while data.iloc[y, x] == horizontal_entrance_path:
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
    if os.path.exists('pod_sku.csv'):
        assign_skus_to_pods_from_file(pod_manager)
    else:
        PodGenerator(pod_manager).generate()
        assign_skus_to_pods_from_file(pod_manager)


def assign_skus_to_pods_from_file(pod_manager: PodManager):
    with open('pod_sku.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pod_id = int(row['pod_id'])
            sku = int(row['sku'])
            limit_qty = int(row['limit_qty'])
            current_qty = int(row['current_qty'])
            threshold = int(row['threshold'])

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
        universe.intersections = intersections  # Ensure 'intersections' is defined earlier

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
        print("========tick========")

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
