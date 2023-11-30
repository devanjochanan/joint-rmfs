from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.robot import Robot
from model.station import Station
from model.order import Order
from engine.object import Object
from model.pod import Pod
import pickle
import csv

intersections = []

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
    [2, 33],
    [2, 27],
    [2, 21],
    [2, 15],
    [2, 9],
    [2, 3],
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
                    pObj.pos_x = current_x - 3
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
        [25, 28, 4],
        [24, 31, 2],
        [26, 11, 3],
        [14, 25, 0],
        [42, 28, 1],
        [43, 11, 2],
        [30, 16, 4],
        [24, 22, 1],
        [26, 11, 3],
        [14, 25, 0],
        [42, 28, 1],
        [44, 31, 2],
        [26, 11, 1],
        [14, 25, 0],
        [25, 28, 4],
        [24, 31, 2],
        [26, 11, 1],
        [14, 25, 0],
        [42, 28, 2],
        [43, 11, 2],
        [30, 16, 4],
        [24, 22, 4],
        [26, 11, 3],
        [14, 25, 1],
        [42, 28, 0],
        [44, 31, 2],
        [26, 11, 3],
        [14, 25, 3],

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
        {'velocity': 0, 'heading': 0, 'x': 15, 'y': 11},
        {'velocity': 0, 'heading': 90, 'x': 14, 'y': 12},
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 5},
        {'velocity': 0, 'heading': 270, 'x': 28, 'y': 21},
        {'velocity': 0, 'heading': 180, 'x': 45, 'y': 26},
        {'velocity': 0, 'heading': 0, 'x': 48, 'y': 10},
        {'velocity': 0, 'heading': 0, 'x': 46, 'y': 2},
        {'velocity': 0, 'heading': 180, 'x': 9, 'y': 14},
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 14},
        {'velocity': 0, 'heading': 180, 'x': 7, 'y': 6},
        {'velocity': 0, 'heading': 270, 'x': 28, 'y': 22},
        {'velocity': 0, 'heading': 180, 'x': 45, 'y': 27},
        {'velocity': 0, 'heading': 0, 'x': 48, 'y': 11},
        {'velocity': 0, 'heading': 0, 'x': 46, 'y': 3},
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
    for i in range(universe.dimension+1):
        for j in range(universe.dimension+1):
            obj = Object()
            obj.object_type = 'way-direction'
            obj.pos_x = j
            obj.pos_y = i
            obj.shape = 'empty-space'
            shape_modification = 0
            if i%3== 0:
                obj.shape = 'arrow-left'
                shape_modification += 2
                if i % 6 == 0:
                    obj.shape = 'arrow-right'
                    shape_modification += 1

            if (j-9) % 6 == 0 and j > 9:
                obj.shape = 'arrow-up'
                shape_modification += 1
            if (j-9) % 12 == 0 and j > 9:
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
                    shape_modification =1
            if shape_modification > 1 or (j < 10 and i%3 == 0) or (j == 5):
                # obj.shape = 'intersection'
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
                        obj.pos_y -= 1
                        obj.shape = 'person-red'

            if i == 35:
                obj.shape = "empty-space"
                # if j % 5 == 0:
                #     obj.shape = 'wall'
            
            if j > 44:
                if j % 2 == 0:
                    obj.shape = 'arrow-up'
                else:
                    obj.shape = 'arrow-down'
            
            if j > 49:
                obj.shape = 'empty-space'
            universe.addObject(obj)

def setup():
    universe = Inventory()
    initWays(universe)
    initStation(universe)
    initRobots(universe)
    initPod(universe)
    initOrders(universe)

    universe.tick_to_second = 0.10
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

# setup()
# x = setup()
# print(x)
# x = tick()
# print(x)