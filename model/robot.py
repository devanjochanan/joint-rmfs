import math
from engine.heading import Heading
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from engine.universe import Universe
from engine.util import calculateDistance
from engine.movement import Movement

class Robot(Object):
    order = None
    destination = None
    
    # netlogo related
    shape = 'turtle-2'
    object_type = 'robot'

    # movement related
    coor = None
    maximum_speed = 2
    current_state = 'idle'
    energy_consumption = 0
    turning = 0
    heading = 0

    # routing related
    latest_rotation = ''
    x_offset = ''
    y_offset = ''

    # energy consumption related
    mass = 1
    load_mass = 0
    _gravity = 10
    _friction = 0.3
    _inertia = 0.4

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

    def move(self):
        current_state = self.current_state

        if self.destination is not None:
            if current_state == 'aligning_x':
                if self.close_enough(self.pos_x, self.destination.x, 1):
                    self.velocity = 0
                    self.acceleration = 0
                    self.current_state = 'aligning_y'
                    self.pos_x = int(self.destination.x)
                    if self.pos_y < self.destination.y:
                        self.heading = 0
                    else:
                        self.heading = 180
                else:
                    if self.heading == 0 or self.heading == 180:
                        self.velocity = 1
                        if self.x_offset == 'right':
                            if self.close_enough(self.pos_y, int(self.pos_y)):
                                if (int(self.pos_y) % 2 == 0):
                                    self.velocity = 0
                                    self.heading = 90
                                    self.pos_y = int(self.pos_y)
                        else:
                            if (round(self.pos_y) % 2 == 1):
                                self.velocity = 0
                                self.heading = 270
                    else:
                        self.acceleration = 1
            if current_state == 'aligning_y':
                self.acceleration = 1
                if self.close_enough(self.pos_y, self.destination.y, 1):
                    self.velocity = 0
                    self.acceleration = 0
                    self.current_state = 'delivery_on_pod'
                    self.pos_y = int(self.destination.y)
                    if (self.pos_y + 1) % 3 == 0:
                        self.pos_y += 1
                    else:
                        self.pos_y -= 1

                    # set offset
                    self.destination = NetLogoCoordinate(5, 27-(self.order.station_number)*6)
                    if self.pos_y < self.destination.y:
                        self.y_offset = 'up'
                    else:
                        self.y_offset = 'down'
            if current_state == 'delivery_on_pod':
                self.velocity = 0
                self.acceleration = 1
                self.current_state = 'delivery_aligning_y'
                if self.pos_y % 6 == 0:
                    self.heading = 90
                else:
                    self.heading = 270
            if current_state == 'delivery_aligning_y':
                up_directed = [6, 8, 15, 27, 39]
                down_directed = [9, 21, 33]
                if self.heading == 90 or self.heading == 270:
                    if self.pos_x > 9:
                        if self.y_offset == 'up' and (int(self.pos_x)-3 ) % 12 == 0:
                            self.heading = 0
                            self.pos_x = int(self.pos_x)
                        elif self.y_offset == 'down' and (int(self.pos_x)+3 ) % 12 == 0:
                            self.heading = 180
                            self.pos_x = int(self.pos_x)
                
                if self.close_enough(self.pos_y, self.destination.y, 0.5):
                    self.velocity = 0
                    self.acceleration = 1
                    self.heading = 270
                    self.current_state = 'delivery_aligning_x'
                    self.pos_y = int(self.destination.y)
            if current_state == 'delivery_aligning_x':
                if self.close_enough(self.pos_x, self.destination.x, 0.5):
                    self.velocity = 0
                    self.acceleration = 0
                    self.order = None
                    self.destination = None
                    self.pos_y -= 3
                    self.heading = 90
        return self.moveNew()

    def moveNew(self):
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

    def setOrder(self, order):
        self.order = order
        self.destination = order.coor
        self.current_state = 'aligning_x'
        if self.pos_x < order.coor.x:
            self.x_offset = 'right'
        else:
            self.x_offset = 'left'

        if self.pos_y < order.coor.y:
            self.y_offset = 'up'
        else:
            self.y_offset = 'down'

    # utility functions
    def calculateTwoPoint(self, p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def close_enough(self, first_number, second_number, precision=0.1):
        return abs(first_number - second_number) < precision
    
    def ensure_coor(self, number):
        if isinstance(number, int):
            print(f"{number} is an integer.")
        elif isinstance(number, float) and number.is_integer():
            print(f"{number} is a float with 0 precision.")
        else:
            print(f"{number} is not a valid integer or float with 0 precision.")

    def get_decimal(self, number):
        subtractor = int(number)
        return number - subtractor