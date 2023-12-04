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
    _id = 0

    # movement related
    coor = None
    maximum_speed = 2
    current_state = 'idle'
    energy_consumption = 0
    turning = 0
    heading = 0
    suspend_movement = 0

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
        self.shape = 'turtle-2'
        current_state = self.current_state

        collision_radius = 4
        collide_distance = 1.25
        neighboors = self.universe.landscape.getNeighboorObject(int(self.pos_x), int(self.pos_y), collision_radius)
        traffic_policy = False
        if self.robotName() == 'robot-1':
            print("Robot:", self.heading, "neighboors", len(neighboors))
        if len(neighboors) > 0:
            will_collide = False
            selected_label = ""
            object_heading = 0
            self_next_blocks = self.calculateNextBlocks(int(self.pos_x), int(self.pos_y), self.heading, 5)
            movement = "vertical"
            if self.heading == 90 or self.heading == 270:
                movement = "horizontal"
            for o in neighboors:
                if o['label'] in self.universe.getTrafficPolicyHistory(self.robotName()):
                    print("Ignore traffic policy from", self.robotName(), "to", o['label'])
                    continue
                if movement != o['movement']:
                    object_next_blocks = self.calculateNextBlocks(int(o['x']), int(o['y']), o['heading'], 5)
                    collision_block = None
                    for p in object_next_blocks:
                        if p in self_next_blocks:
                            collision_block = p
                            break

                    if collision_block is not None:
                        self_distance = self.calculateTwoPoint(NetLogoCoordinate(self.pos_x, self.pos_y), NetLogoCoordinate(collision_block[0], collision_block[1]))
                        object_distance = self.calculateTwoPoint(NetLogoCoordinate(o['x'], o['y']), NetLogoCoordinate(collision_block[0], collision_block[1]))
                        if self_distance < object_distance:
                            continue

                if self.heading == 270:
                    # care robot at left
                    if o['x'] < self.pos_x:
                        if (o['heading'] == 0 and o['y'] <= self.pos_y and o['velocity'] != 100) or (o['heading'] == 180 and o['y'] >= self.pos_y and o['velocity'] != 100) or (o['heading'] == 270 and o['y'] == self.pos_y):
                            # calculate distance with hypotenuse
                            distance = math.sqrt((o['x'] - self.pos_x)**2 + (o['y'] - self.pos_y)**2)
                            if distance < collide_distance:
                                will_collide = True
                                selected_label = o['label']
                                object_heading = o['heading']
                                break

                elif self.heading == 90:
                    # care robot at right
                    if o['x'] > self.pos_x:
                        if (o['heading'] == 0 and o['y']-1 <= self.pos_y and o['velocity'] != 100) or (o['heading'] == 180 and o['y'] >= self.pos_y and o['velocity'] != 100) or (o['heading'] == 90 and o['y'] == self.pos_y):
                            # calculate distance with hypotenuse
                            distance = math.sqrt((o['x'] - self.pos_x)**2 + (o['y'] - self.pos_y)**2)
                            if distance < collide_distance:
                                will_collide = True
                                selected_label = o['label']
                                object_heading = o['heading']
                                break
                    
                elif self.heading == 0:
                    # care robot at top
                    if o['y'] > self.pos_y:
                        if (o['heading'] == 90 and o['x'] <= self.pos_x+1 and o['velocity'] != 100) or (o['heading'] == 270 and o['x'] >= self.pos_x-1 and o['velocity'] != 100) or (o['heading'] == 0 and o['x'] == self.pos_x):
                            distance = math.sqrt((o['x'] - self.pos_x)**2 + (o['y'] - self.pos_y)**2)
                            if distance < collide_distance:
                                will_collide = True
                                selected_label = o['label']
                                object_heading = o['heading']
                                break

                elif self.heading == 180:
                    # care robot at bottom
                    if o['y'] < self.pos_y:
                        if (o['heading'] == 90 and o['x'] <= self.pos_x and o['velocity'] != 100) or (o['heading'] == 270 and o['x'] >= self.pos_x and o['velocity'] != 100) or (o['heading'] == 180 and o['x'] == self.pos_x):
                            distance = math.sqrt((o['x'] - self.pos_x)**2 + (o['y'] - self.pos_y)**2)
                            if distance < collide_distance:
                                will_collide = True
                                selected_label = o['label']
                                object_heading = o['heading']
                                break

            if will_collide:
                self.velocity = 0
                self.acceleration = 0
                self.suspend_movement = 8
                traffic_policy = True

                if object_heading != self.heading:
                    print("Applied traffic policy from", self.robotName(), "to", selected_label)
                    self.universe.addTrafficPolicyHistory(self.robotName(), selected_label)

            if self.robotName() == 'robot-1':
                print("Robot:", self.heading, "will collide", will_collide)
        if self.suspend_movement > 0 and traffic_policy == False:
            self.suspend_movement = 0
            if self.suspend_movement == 0:
                self.acceleration = 1

        if self.destination is not None and traffic_policy == False:
            if current_state == 'aligning_x':
                if self.close_enough(self.pos_x, self.destination.x, 0.5):
                    self.velocity = 0
                    self.acceleration = 0
                    self.current_state = 'aligning_y'
                    self.pos_x = int(self.destination.x)
                    if self.pos_y < self.destination.y:
                        self.heading = 0
                        self.turning += 1
                    else:
                        self.heading = 180
                        self.turning += 1
                else:
                    if self.heading == 0 or self.heading == 180:
                        self.velocity = 1
                        if self.x_offset == 'right':
                            if self.close_enough(self.pos_y, int(self.pos_y)):
                                if (int(self.pos_y) % 2 == 0):
                                    self.velocity = 0
                                    self.heading = 90
                                    self.turning += 1
                                    self.pos_y = int(self.pos_y)
                        else:
                            if self.close_enough(self.pos_y, int(self.pos_y)):
                                if (int(self.pos_y) % 2 == 1):
                                    self.velocity = 0
                                    self.heading = 270
                                    self.turning += 1
                                    self.pos_y = int(self.pos_y)
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
                    self.destination = NetLogoCoordinate(2, 27-(self.order.station_number)*6)
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
                    self.turning += 1
                else:
                    self.heading = 270
                    self.turning += 1
            if current_state == 'delivery_aligning_y':
                if self.heading == 90 or self.heading == 270:
                    if self.pos_x > 9:
                        if self.y_offset == 'up' and ((self.pos_x >= 45 and int(self.pos_x) % 2 == 0) or (int(self.pos_x)-3 ) % 12 == 0):
                            self.heading = 0
                            self.turning += 1
                            self.pos_x = int(self.pos_x)
                        elif self.y_offset == 'down' and ((self.pos_x >= 45 and int(self.pos_x) % 2 != 0) or (int(self.pos_x)+3 ) % 12 == 0):
                            self.heading = 180
                            self.turning += 1
                            self.pos_x = int(self.pos_x)
                if self.close_enough(self.pos_y, self.destination.y, 0.5):
                    self.velocity = 0
                    self.acceleration = 1
                    self.heading = 270
                    self.turning += 1
                    self.current_state = 'delivery_aligning_x'
                    self.pos_y = int(self.destination.y)
            if current_state == 'delivery_aligning_x':
                if self.close_enough(self.pos_x, self.destination.x, 0.5):
                    self.pos_x = int(self.destination.x)
                    self.acceleration = 0
                    self.heading = 180
                    self.turning += 1
                    self.velocity = 1
                    self.destination = NetLogoCoordinate(self.destination.x, self.destination.y-2)
                    self.current_state = 'delivery_on_station'
                    self.idle_tick = 32
            if current_state == 'delivery_on_station':
                if self.close_enough(self.pos_y, self.destination.y, 0.5):
                    self.velocity = 0
                    if self.idle_tick == 0:
                        self.current_state = 'delivery_on_exit_station'
                        self.velocity = 1
                        self.destination = NetLogoCoordinate(self.destination.x, self.destination.y-1)
                    self.idle_tick -= 1
            if current_state == 'delivery_on_exit_station':
                if self.close_enough(self.pos_y, self.destination.y, 0.5):
                    self.velocity = 0
                    self.heading = 270
                    self.turning += 1
                    self.current_state = 'idle'
                    self.heading = 90
                    self.pos_x = int(self.destination.x)
                    self.pos_y = int(self.destination.y)
                    self.order = None
                    self.destination = None
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

        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration, self.heading)

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
    
    def robotName(self):
        return f"robot-{self._id}"
    
    def robotID(self, robotName):
        return int(robotName.split('-')[1])
    
    def calculateNextBlocks(self, x, y, heading, block_count = 5):
        x_difference = 0
        y_difference = 0

        if heading == 0:
            y_difference = 1
        if heading == 90:
            x_difference = 1
        if heading == 270:
            x_difference = -1
        if heading == 180:
            y_difference = -1
        
        result = []
        for i in range(block_count):
            result.append([x, y])

            x += x_difference
            y += y_difference
        
        return result