from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.robot import Robot
from model.station import Station
from model.order import Order
from engine.object import Object
from model.pod import Pod
import pickle
import csv

import networkx as nx
import matplotlib.pyplot as plt

class DirectedGraph:
    key = ''

    def __init__(self):
        self.graph = nx.DiGraph()

    def node_valid(self, node):
        l = node.split(",")
        x = int(l[0])
        y = int(l[1])
        return x>=2 and y>=0
    
    def add_node(self, node):
        if self.node_valid(node):
            self.graph.add_node(node)

    def add_edge(self, start, end, weight):
        if self.node_valid(start) and self.node_valid(end):
            self.graph.add_edge(start, end, weight=weight)

    def dijkstra(self, start, end):
        try:
            path = nx.shortest_path(self.graph, source=start, target=end, weight='weight', method='bellman-ford')
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
    graph = universe.graph
    graph_pod = universe.graph_pod
    with open('pod.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            current_x = 0
            for j in row:
                if(j == '1'):
                    pObj = Pod()
                    pObj.pos_x = current_x-1
                    pObj.pos_y = line_count
                    pObj.coor = NetLogoCoordinate(pObj.pos_x, pObj.pos_y)
                    universe.addObject(pObj)
                    obj_key = str(pObj.pos_x) + "," + str(pObj.pos_y)
                    graph.add_node(obj_key)
                    graph_pod.add_node(obj_key)

                    obj_key_neighboor = str(pObj.pos_x) + "," + str(pObj.pos_y-1)
                    if (pObj.pos_y + 1) % 3 == 0:
                        obj_key_neighboor = str(pObj.pos_x) + "," + str(pObj.pos_y+1)
                    graph_pod.add_node(obj_key_neighboor)
                    graph.add_edge(obj_key, obj_key_neighboor, weight=1)
                    graph_pod.add_edge(obj_key, obj_key_neighboor, weight=1)
                    graph_pod.add_edge(obj_key_neighboor, obj_key, weight=1)
                current_x += 1
            if line_count > 32:
                break
            line_count += 1

def initStation(universe: Inventory):
    for s in stations:
        sObj = Station()
        sObj.pos_x = s[0]
        sObj.pos_y = s[1]
        sObj.coor = NetLogoCoordinate(s[0], s[1])
        universe.addObject(sObj)
        universe.addStation(sObj)
    
def initOrders(universe: Inventory):
    dest = [
        [12, 13, 2],
        [11, 13, 2],
        [30, 14, 2],
        [12, 25, 2],
        [25, 31, 2],
        [26, 11, 2],
        [15, 25, 2],
        [42, 28, 2],
        [43, 11, 2],
        [30, 16, 2],
        [26, 22, 2],
        [26, 11, 2],
        [15, 25, 0],
        # [42, 28, 1],
        # [44, 31, 2],
        # [26, 11, 1],
        # [14, 25, 0],
        # [25, 28, 4],
        # [24, 31, 2],
        # [26, 11, 1],
        # [14, 25, 0],
        # [42, 28, 2],
        # [43, 11, 2],
        # [30, 16, 4],
        # [24, 22, 4],
        # [26, 11, 3],
        # [14, 25, 1],
        # [42, 28, 0],
        # [44, 31, 2],
        # [26, 11, 3],
        # [14, 25, 3],

        # [27, 28, 1],
        
        # [27, 28, 4],
        # [27, 28, 0],
        # [27, 28, 1],
        # [27, 28, 2],
        # [27, 28, 3],
        # [27, 28, 4],
        # [27, 28, 0],
        # [27, 28, 1],
        # [27, 28, 2],
        # [27, 28, 3],
        # [27, 28, 4],
        # [27, 28, 0],
        # [27, 28, 1],
        # [27, 28, 2],
        # [27, 28, 3],
        # [27, 28, 4],
    ]

    for d in dest:
        order = Order([d[0], d[1]])
        order.coor = NetLogoCoordinate(order.designated_pod[0], order.designated_pod[1])
        order.station_number = d[2]
    
        universe.addOrder(order)
        obj = Object()
        obj.pos_x = d[0]
        obj.pos_y = d[1]
        obj.shape = 'box'
        obj.object_type = 'order'
        universe.addObject(obj)
        
    
def initRobots(universe: Inventory):
    robots = [
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 11},
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 12},
        {'velocity': 0, 'heading': 0, 'x': 14, 'y': 10},
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 5},
        {'velocity': 0, 'heading': 270, 'x': 28, 'y': 21},
        {'velocity': 0, 'heading': 180, 'x': 45, 'y': 26},
        {'velocity': 0, 'heading': 0, 'x': 48, 'y': 10},
        {'velocity': 0, 'heading': 0, 'x': 46, 'y': 2},
        # {'velocity': 0, 'heading': 180, 'x': 9, 'y': 14},
        # {'velocity': 0, 'heading': 180, 'x': 7, 'y': 14},
        # {'velocity': 0, 'heading': 180, 'x': 7, 'y': 6},
        # {'velocity': 0, 'heading': 270, 'x': 28, 'y': 22},
        # {'velocity': 0, 'heading': 180, 'x': 45, 'y': 27},
        # {'velocity': 0, 'heading': 0, 'x': 48, 'y': 11},
        # {'velocity': 0, 'heading': 0, 'x': 46, 'y': 3},
    ]
    
    for r in robots:
        robot = Robot()
        robot.velocity = r['velocity']
        robot.heading = r['heading']
        robot.pos_x = r['x']
        robot.pos_y = r['y']
        robot.coor = NetLogoCoordinate(robot.pos_x, robot.pos_y)
        universe.addObject(robot)


def initWays(universe):
    # Example Usage
    graph = DirectedGraph()
    graph_pod = DirectedGraph()
    graph_pod.key = 'pod'
    universe.graph = graph
    universe.graph_pod = graph_pod

    for i in range(universe.dimension+1):
        if i > 33:
            break
        for j in range(universe.dimension+1):
            obj = Object()
            obj.object_type = 'way-direction'
            obj.pos_x = j
            obj.pos_y = i
            obj_key = str(j) + "," + str(i)
            graph.add_node(obj_key)
            obj.shape = 'empty-space'
            shape_modification = 0

            weight = 1
            obj_key_neighboor = str(j-1) + "," + str(i)
            if i % 2 == 0:
                obj_key_neighboor = str(j+1) + "," + str(i)
            if j < 5 and i % 3 != 0:
                weight = 20
            graph.add_edge(obj_key, obj_key_neighboor, weight=weight)

            obj_key_neighboor_2 = str(j) + "," + str(i-1)
            if j % 2 == 0:
                obj_key_neighboor_2 = str(j) + "," + str(i+1)
                if j == 2:
                    obj_key_neighboor_2 = str(j) + "," + str(i-1)
            graph.add_edge(obj_key, obj_key_neighboor_2, weight=1)
            

            if i % 3 == 0:
                obj.shape = 'arrow-left'
                shape_modification += 1
                if i % 6 == 0:
                    obj.shape = 'arrow-right'
                    shape_modification += 1

            if (j-9) % 5 == 0 and j > 9:
                obj.shape = 'arrow-up'
                shape_modification += 1
            if (j-9) % 10 == 0 and j > 9:
                obj.shape = 'arrow-down'
                shape_modification += 1
            
            # draw hallway
            if j < 10:
                if j % 2 == 1:
                    obj.shape = 'arrow-down'
                    shape_modification += 1
                else:
                    obj.shape = 'arrow-up'
                    shape_modification += 1
                    
            if j < 5:
                obj.shape = 'empty-space'
            if j == 9:
                obj.shape = 'arrow-down'
            
            if shape_modification:
                if j > 9 and (((j-9) % 6  == 0) == False):
                    shape_modification = 1
                if j > 9 and ((i%3 != 0)):
                    shape_modification = 1
            if shape_modification > 1 or (j < 10 and i%3 == 0) or (j == 5):
                intersections.append([obj.pos_x, obj.pos_y])
            
            if j < 5 and j >= 2 and i < 34:
                if i % 3 == 0:
                    obj.shape = 'rail'
                    if j == 2:
                        if i % 6 == 0:
                            obj.heading = 270
                        obj.shape = 'rail-corner'
                else:
                    if j == 2 and i not in [4, 5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35]:
                        obj.shape = 'rail'
                        obj.heading = 90
                    else:
                        obj.shape = 'empty-space'

            if j == 1:
                if i % 3 == 0:
                    if i % 6 != 0:
                        obj.pos_y -= 2
                        obj.shape = 'person-red'

            if i == 35:
                obj.shape = "empty-space"
            
            if j > 44:
                if j % 2 == 0:
                    obj.shape = 'arrow-up'
                else:
                    obj.shape = 'arrow-down'
            
            if j > 49:
                obj.shape = 'empty-space'

            if i % 3 == 0:
                obj_key_neighboor = str(j-1) + "," + str(i)
                if i % 6 == 0:
                    obj_key_neighboor = str(j+1) + "," + str(i)
                graph_pod.add_edge(obj_key, obj_key_neighboor, weight=1)

            weight = 1
            if j < 10:
                weight = 3

            if obj.shape == 'arrow-up':
                obj_key_neighboor = str(j) + "," + str(i+1)
                graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)
            elif obj.shape == 'arrow-down':
                obj_key_neighboor = str(j) + "," + str(i-1)
                graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)
            
            if j < 5:
                if (i-3) % 6 == 0:
                    obj_key_neighboor = str(j-1) + "," + str(i)
                    graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)
                if (i) % 6 == 0:
                    obj_key_neighboor = str(j+1) + "," + str(i)
                    graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)
                if j == 2:
                    obj_key_neighboor = str(j) + "," + str(i-1)
                    graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)
                    obj_key_neighboor = str(j) + "," + str(i-2)
                    graph_pod.add_edge(obj_key, obj_key_neighboor, weight=weight)

            universe.addObject(obj)
    # Visualization (optional)
    # pos = nx.spring_layout(graph.graph)
    # nx.draw(graph.graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", arrows=True, connectionstyle="arc3,rad=0.1")
    # labels = nx.get_edge_attributes(graph.graph, 'weight')
    # nx.draw_networkx_edge_labels(graph.graph, pos, edge_labels=labels)
    # plt.show()

def setup():
    universe = Inventory()
    initWays(universe)
    initStation(universe)
    initRobots(universe)
    initPod(universe)
    initOrders(universe)

    universe.tick_to_second = 0.15
    universe.intersections = intersections
    
    next = universe.generateResult()

    with open('netlogo.state', 'wb') as config_dictionary_file:
        pickle.dump(universe, config_dictionary_file)

    return next

def tick():
    print("========tick========")
    universe = None
    
    # open a file, where you stored the pickled data
    file = open('netlogo.state', 'rb')

    # dump information to that file
    universe = pickle.load(file)

    print("before tick", universe._tick)

    for _n in universe._objects:
        _n.setUniverse(universe)

    # close the file
    file.close()

    universe.tick()

    next = universe.generateResult()
    with open('netlogo.state', 'wb') as config_dictionary_file:
        pickle.dump(universe, config_dictionary_file)
    return [next, universe.total_energy, len(universe.order_queue), universe.stop_and_go, universe.total_turning]

def setup_py():
    from pip._internal import main as pipmain

    def install_package(package_name):
        pipmain(["install", package_name])

    # Example: Install the "requests" package
    packages = [
        "networkx",
        "matplotlib",
    ]
    for p in packages:
        install_package(p)
# setup()
# x = setup()
# print(x)
# x = tick()
# print(x)