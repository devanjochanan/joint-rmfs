from engine.heading import Heading
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from engine.universe import Universe
from engine.util import calculateDistance
from engine.movement import Movement

import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def clean_astar_result(r):
    s = []
    for p in r:
        s.append([p[0], 53-p[1]])
    return s

def astar(grid, start, end):
    def heuristic(node, end):
        return abs(node.x - end.x) + abs(node.y - end.y)

    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    open_set.append(start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return clean_astar_result(path[::-1])

        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_node.x + dx, current_node.y + dy

            if (
                0 <= x < len(grid)
                and 0 <= y < len(grid[0])
                and grid[x][y] == 0
                and (x, y) not in closed_set
            ):
                neighbor = Node(x, y, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = heuristic(neighbor, end_node)

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)

    return None

class Robot(Object):
    destination = None
    order = None
    routes = []
    movement = []
    mass = 1
    load_mass = 0
    maximum_speed = 2
    shape = 'turtle'
    object_type = 'robot'
    latest_rotation = ''
    _gravity = 10
    _friction = 0.3
    _inertia = 0.4
    energy_consumption = 0

    def __init__(self):
        super().__init__()
            
    def checkMovementDirection(self, p1, p2):
        if p1.y == p2.y:
            # horizontal
            if p1.x < p2.x:
                return 90
            if p1.x > p2.x:
                return 270
            
        if p1.x == p2.x:
            # vertical
            if p1.y < p2.y:
                return 0
            if p1.y > p2.y:
                return 180
        
        return None
    
    def calculateEnergy(self, velocity, acceleration):
        tick_unit = self.universe.tick_to_second
        if acceleration != 0:
            average_speed = 2*velocity + (acceleration * tick_unit)
            return (self.mass + self.load_mass) * ((self._gravity * self._friction) + (acceleration * self._inertia)) * average_speed * tick_unit/7200
        elif velocity != 0:
            return (self.mass + self.load_mass) * self._gravity * self._friction * velocity * tick_unit/3600
        return 0

            
    def moveNew(self):
        if len(self.movement) < 1 and self.order != None:
            raise Exception("weird")
            return

        initial_velocity = self.velocity
        initial_acceleration = self.acceleration

        self.energy_consumption += self.calculateEnergy(initial_velocity, initial_acceleration)
        
        if(self.velocity != 0):
            if(self.heading == 0):
                self.pos_y += self.velocity * self.universe.tick_to_second
            elif(self.heading == 180):
                self.pos_y -= self.velocity * self.universe.tick_to_second
            elif(self.heading == 90):
                self.pos_x += self.velocity * self.universe.tick_to_second
            elif(self.heading == 270):
                self.pos_x -= self.velocity * self.universe.tick_to_second
        self.coor = NetLogoCoordinate(int(self.pos_x), int(self.pos_y))

        if self.acceleration != 0:
            self.velocity = self.velocity + (self.acceleration * self.universe.tick_to_second)
            if self.velocity > self.maximum_speed:
                self.velocity = self.maximum_speed
        
        # print("masuk next-tick: {} tick {} heading {} x {} y {} velocity {} acceleration {}".format(self.movement[0], self.universe._tick, self.heading, self.pos_x, self.pos_y, self.velocity, self.acceleration))
        while len(self.movement) > 0 and (isinstance(self.movement[0], Heading) or self.movement[0].tick <= self.universe._tick):
            next = self.movement.pop(0)
            if isinstance(next, Heading):
                print("Update heading")
                self.heading = next.getHeading()
                self.energy_consumption += 1
            elif isinstance(next, Movement):
                print("Update speed")
                self.acceleration = next.acceleration
        
        # if len(self.movement) > 0:
        #     print("out next-tick: {} tick {} heading {} x {} y {} velocity {} acceleration {}".format(self.movement[0].tick, self.universe._tick, self.heading, self.pos_x, self.pos_y, self.velocity, self.acceleration))
        
        if self.order != None:
            station_number = self.order.station_number
            station = self.universe.stations[station_number]
            if (self.pos_x == station.pos_x and self.pos_y == station.pos_y):
                self.order = None
                self.routes = []
                self.movement = []

    def movePatches(self):
        if len(self.patches) < 1:
            return
        
        next = self.patches.pop(0)
        if isinstance(next, Heading):
            return
        
        self.pos_x = next.x
        self.pos_y = next.y
        self.coor = NetLogoCoordinate(self.pos_x, self.pos_y)

        if self.order != None:
            station_number = self.order.station_number
            station = self.universe.stations[station_number]
            if self.pos_x == station.pos_x and self.pos_y == station.pos_y:
                raise Exception("im here")
                self.order = None
                self.routes = []
                self.patches = []

    def move(self):
        # return self.movePatches()
        return self.moveNew()
        self.universe.total_energy += 1
        if len(self.routes) < 1:
            return
        
        next = self.routes.pop(0)
        if isinstance(next, Heading):
            if self.heading == next.getHeading():
                self.universe.total_energy -= 1
                self.move()
                return
            self.heading = next.getHeading()
        else:
            if(self.velocity != 0 or self.acceleration != 0):
                if self.acceleration != 0:
                    self.velocity = self.velocity + (self.acceleration*self.universe.tick_to_second)
                    
                if(self.heading == 0):
                    self.pos_y += self.velocity
                elif(self.heading == 180):
                    self.pos_y -= self.velocity
                elif(self.heading == 90):
                    self.pos_x += self.velocity
                elif(self.heading == 270):
                    self.pos_x -= self.velocity
            if self.pos_y == next.y and self.pos_x == next.x:
                self.universe.total_energy -= 1
                self.move()
                return
            self.pos_x = next.x
            self.pos_y = next.y
            if isinstance(next.x, int) and isinstance(next.y, int):
                self.coor = NetLogoCoordinate(int(next.x), int(next.y))

        if self.order != None:
            station_number = self.order.station_number
            station = self.universe.stations[station_number]
            if self.pos_x == station.pos_x and self.pos_y == station.pos_y:
                self.order = None
                self.routes = []

    def setOrder(self, order):
        self.order = order

        get_path = self.aStar(order)
        if len(self.routes) > 0:
            raise Exception("Check code")
        
        movement_direction = self.checkMovementDirection(self.coor, get_path[1])
        movement = []
        if movement_direction != self.heading and movement_direction != None:
            get_path.insert(0, Heading(movement_direction))
            movement.append(Heading(movement_direction))

        self.patches = get_path
        print("get_path", movement_direction, get_path)
        self.movement = self.routeToMovement(get_path)
        # if isinstance(get_path[0], Heading):
        #     self.movement.insert(0, get_path[0])
        # print("movement2", self.movement)
        self.routes = get_path
    
    def calculateEnergyConstant(self, v):
        return v*100

    def calculateEnergyOnAcceleration(self, d):
        return
    
    def routeToMovement(self, paths):
        paths_simplified = []
        current_direction = self.heading
        # movement_direction = self.checkMovementDirection(self.coor, paths[1])
        # if current_direction != movement_direction:

        moved_units = 0
        print("convert", len(paths), paths)
        for p in range(len(paths)-2):
            p1 = paths[p]
            p2 = paths[p+1]
            if isinstance(p1, Heading) or isinstance(p2, Heading):
                if isinstance(p1, Heading):
                    if moved_units > 0:
                        paths_simplified.append({
                            'type': current_direction,
                            'units': moved_units,
                        })
                    moved_units = 0
                    current_direction = p1.getHeading()
                    paths_simplified.append(p1)
                continue
            moved_units += 1
        if moved_units > 0:
            paths_simplified.append({
                'type': current_direction,
                'units': moved_units+1,
            })
            # if p1.x == p2.x and p2.y == p1.y:
            #     continue
            # moved_units += 1
            

        current_second = self.universe._tick

        simulated_velocity = self.velocity
        simulated_acceleration = self.acceleration
        movement = []
        print(paths_simplified)
        for p in paths_simplified:
            if isinstance(p, Heading):
                current_second += 1
                movement.append(p)
                continue
            # accel then decel, no constant
            movements = self.calculateTimeForUAEShort(0, p['units'], current_second)
            current_second = movements[-1].tick
            for x in movements:
                movement.append(x)
            
        print("movement", movement)
        return movement

    # calculate movement for movement less than 8 units
    def calculateTimeForUAEShort(self, initial_speed, distance, current_tick):
        answer = []
        if distance == 1:
            movement_1 = Movement()
            movement_1.acceleration = 0.25
            movement_1.tick = current_tick
            
            movement_1_stop = Movement()
            movement_1_stop.acceleration = 0
            movement_1_stop.tick = current_tick + (2)

            start_decelerate_time = Movement()
            start_decelerate_time.acceleration = -0.25
            start_decelerate_time.tick = movement_1_stop.tick
            
            stop_decelerate_time = Movement()
            stop_decelerate_time.acceleration = 0
            stop_decelerate_time.tick = start_decelerate_time.tick + (2)

            answer.append(movement_1)
            answer.append(movement_1_stop)
            answer.append(start_decelerate_time)
            answer.append(stop_decelerate_time)
        elif distance <= 8:
            constant_distance = distance-2
            constant_time = constant_distance/1
            if constant_time < 0:
                constant_time = 0

            movement_1 = Movement()
            movement_1.acceleration = 0.5
            movement_1.tick = current_tick
            
            movement_1_stop = Movement()
            movement_1_stop.acceleration = 0
            movement_1_stop.tick = current_tick + (2)

            start_decelerate_time = Movement()
            start_decelerate_time.acceleration = -0.5
            start_decelerate_time.tick = movement_1_stop.tick + (constant_time)
            
            stop_decelerate_time = Movement()
            stop_decelerate_time.acceleration = 0
            stop_decelerate_time.tick = start_decelerate_time.tick + (2)

            answer.append(movement_1)
            answer.append(movement_1_stop)
            answer.append(start_decelerate_time)
            answer.append(stop_decelerate_time)
        else:
            constant_distance = distance-8
            constant_time = constant_distance/2
            if constant_time < 0:
                constant_time = 0
            movement_1 = Movement()
            movement_1.acceleration = 0.5
            movement_1.tick = current_tick
            
            movement_1_stop = Movement()
            movement_1_stop.acceleration = 0
            movement_1_stop.tick = current_tick + (4)

            start_decelerate_time = Movement()
            start_decelerate_time.acceleration = -0.5
            start_decelerate_time.tick = movement_1_stop.tick + (constant_time)
            
            stop_decelerate_time = Movement()
            stop_decelerate_time.acceleration = 0
            stop_decelerate_time.tick = start_decelerate_time.tick + (4)

            answer.append(movement_1)
            answer.append(movement_1_stop)
            answer.append(start_decelerate_time)
            answer.append(stop_decelerate_time)
        
        return answer


    def aStar(self, order):
        _from = self.coor
        _to = order.coor

        station_number = order.station_number
        station = self.universe.stations[station_number]
        _station = NetLogoCoordinate(station.pos_x, station.pos_y)

        landscape = self.universe.landscape
        print("coordinates", _from, _to, _station)
        path = landscape.getRoute(_from, _to)

        path2 = landscape.getRoute(_to, _station)
        
        latest_heading = None
        for p in path:
            if isinstance(p, Heading):
                latest_heading = p.getHeading()

        direction = landscape.checkMovementDirection(path2[0], path2[1])
        if latest_heading != direction:
            new_heading = Heading()
            new_heading.heading = direction
            path2.insert(0, new_heading)

        return path + path2

