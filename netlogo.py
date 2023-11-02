from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.robot import Robot
from model.station import Station
from model.order import Order
from model.pod import Pod
import pickle
import csv

universe = Inventory()

def createPod(p1, p2):
    res = []
    current_x = p1['x']
    for i in range(5):
        res.append([current_x, p1['y']])
        current_x += 1

    current_x = p1['x']
    for i in range(5):
        res.append([current_x, p2['y']])
        current_x += 1
    return res

stations = [
    [5, 50],
    [5, 40],
    [5, 30],
    [5, 20],
    [5, 10],
]

def initPod(universe: Inventory):
    with open('pod.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            current_x = 0
            for j in row:
                if(j == '1'):
                    pObj = Pod()
                    pObj.pos_x = current_x
                    pObj.pos_y = line_count
                    pObj.coor = NetLogoCoordinate(current_x, line_count)
                    universe.addObject(pObj)
                current_x += 1
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
        [23, 11, 1],
        [25, 47, 0],
        [27, 28, 2],
        [16, 15, 1],
        [16, 17, 4],
        [40, 11, 1],
        [25, 47, 4],
        [23, 11, 1],
        [25, 47, 0],
        [27, 28, 2],
        [16, 15, 1],
        [16, 17, 4],
        [40, 11, 1],
        [25, 47, 4],
    ]

    for d in dest:
        order = Order([d[0], d[1]])
        order.coor = NetLogoCoordinate(order.designated_pod[0], order.designated_pod[1])
        order.station_number = d[2]
    
        universe.addOrder(order)
        
    
def initRobots(universe: Inventory):
    robots = [
        {'velocity': 0, 'heading': 0, 'x': 12, 'y': 13},
        {'velocity': 0, 'heading': 180, 'x': 24, 'y': 48},
        {'velocity': 0, 'heading': 180, 'x': 36, 'y': 21},
        # {'velocity': 1, 'heading': 180, 'x': 50, 'y': 50},
        # {'velocity': 1, 'heading': 180, 'x': 51, 'y': 2},
    ]
    
    for r in robots:
        robot = Robot()
        robot.velocity = r['velocity']
        robot.heading = r['heading']
        robot.pos_x = r['x']
        robot.pos_y = r['y']
        robot.coor = NetLogoCoordinate(robot.pos_x, robot.pos_y)
        universe.addObject(robot)

def setup():
    initStation(universe)
    initRobots(universe)
    initPod(universe)
    initOrders(universe)

    universe.tick_to_second = 0.25
    
    next = universe.generateResult()

    with open('netlogo.state', 'wb') as config_dictionary_file:
        pickle.dump(universe, config_dictionary_file)

    return next

def tick():
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
    return [next, universe.total_energy, len(universe.order_queue)]

# setup()
# x = setup()
# print(x)
# x = tick()
# print(x)