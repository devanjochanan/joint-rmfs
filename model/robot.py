import math
from typing import Optional

from engine.heading import Heading
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from .robot_job import RobotJob
from .traffic_policy import TrafficPolicy


class Robot(Object):
    # netlogo related
    shape = 'turtle-2'
    object_type = 'robot'
    _id = 0

    # movement related
    coordinate = None
    maximum_speed = 2
    current_state = 'idle'
    energy_consumption = 0
    turning = 0
    heading = 0
    suspend_movement = 0
    route_stop_points = []
    pick_pod_item_delay = 0

    # routing related
    latest_rotation = ''

    # traffic policy related
    traffic_policy = []
    latest_tick = 0

    # energy consumption related
    mass = 1
    load_mass = 0
    _gravity = 10
    _friction = 0.3
    _inertia = 0.4

    def __init__(self):
        self.id = None
        self.traffic_policy = []
        self.job: Optional[RobotJob] = None
        super().__init__()

    @staticmethod
    def _checkMovementDirection(p1, p2):
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
        if acceleration != 0 and velocity != 0:
            average_speed = 2 * velocity + (acceleration * tick_unit)
            return (self.mass + self.load_mass) * ((self._gravity * self._friction) + (
                    acceleration * self._inertia)) * average_speed * tick_unit / 7200
        elif velocity != 0:
            return (self.mass + self.load_mass) * self._gravity * self._friction * velocity * tick_unit / 3600
        return 0

    def setMovementPlanToStation(self):
        start = self.coordinate_to_string_key(self.pos_x, self.pos_y)

        station_pos_x, station_pos_y = self.get_station_position()
        end = self.coordinate_to_string_key(station_pos_x, station_pos_y)

        node_routes = self.universe.graph_pod.dijkstra(start, end)
        self.setPath(self._transformRouteToList(node_routes))

    def get_station_position(self):
        pos_x = self.job.station_coordinate.x
        pos_y = self.job.station_coordinate.y
        return pos_x, pos_y

    def setPath(self, path):
        current_heading = self.heading
        route_stop_points = []

        # convert path list to route_stop_points (list of NetLogoCoordinate where the robot should stop and Heading to turn the robot)
        for i in range(1, len(path), 1):
            p1 = NetLogoCoordinate(path[i - 1][0], path[i - 1][1])
            p2 = NetLogoCoordinate(path[i][0], path[i][1])
            heading = self.getHeading(p1, p2)
            if current_heading != heading:
                current_heading = heading
                route_stop_points.append(NetLogoCoordinate(path[i - 1][0], path[i - 1][1]))
                route_stop_points.append(Heading(heading))
        if len(route_stop_points) > 0 and isinstance(route_stop_points[len(route_stop_points) - 1],
                                                     NetLogoCoordinate) == False:
            now = path[len(path) - 1]
            route_stop_points.append(NetLogoCoordinate(now[0], now[1]))
        self.route_stop_points = route_stop_points

    def changeColorByState(self):
        if self.current_state == "taking_pod":
            self.color = 57  # green
        elif self.current_state == "delivering_pod":
            self.color = 15  # red
        elif self.current_state == "returning_pod":
            self.color = 46  # yellow

    def advanceState(self):
        if self.current_state == "taking_pod":
            self.current_state = "delivering_pod"
        elif self.current_state == "delivering_pod":
            self.current_state = "picking"
            self.pick_pod_item_delay = 64
        elif self.current_state == "returning_pod":
            self.current_state = "idle"

    def decideCollision(self, collision_block, o, collide_distance):
        will_collide = False
        selected_label = ""
        object_heading = 0
        distance = math.sqrt((o['x'] - self.pos_x) ** 2 + (o['y'] - self.pos_y) ** 2)
        if collision_block is not None:
            self_distance = self._calculateTwoPoint(NetLogoCoordinate(self.pos_x, self.pos_y),
                                                    NetLogoCoordinate(collision_block[0], collision_block[1]))
            if (self.velocity ** 2) / 2 >= self_distance or distance < collide_distance:
                will_collide = True
                selected_label = o['label']
                object_heading = o['heading']
        elif distance < collide_distance:
            will_collide = True
            selected_label = o['label']
            object_heading = o['heading']
        return will_collide, selected_label, object_heading

    def getCollisionCandidates(self, exclude=None):
        if exclude is None:
            exclude = []
        collision_radius = 4

        # this variable then will be replaced with correct closest neighboor that might collide
        closest_candidates = []

        neighbors = self.universe.landscape.getNeighborObject(int(self.pos_x), int(self.pos_y), collision_radius)
        if len(neighbors) > 0:
            self_next_blocks = self._calculateNextBlocks(int(self.pos_x), int(self.pos_y), self.heading, 10)

            for o in neighbors:
                if o['label'] in exclude:
                    continue
                object_next_blocks = self._calculateNextBlocks(int(o['x']), int(o['y']), o['heading'], 10)

                collision_block = self._getIntersectionBlock(self_next_blocks, object_next_blocks)
                if collision_block is None:
                    continue

                if self.heading == 270:
                    # care robot at left
                    if o['x'] < self.pos_x:
                        if (o['heading'] == 0 and o['y'] <= self.pos_y) or (
                                o['heading'] == 180 and o['y'] >= self.pos_y) or (
                                o['heading'] == 270 and int(o['y']) == int(self.pos_y)):
                            closest_candidates.append(self._getRobot(o['label']))

                elif self.heading == 90:
                    # care robot at right
                    if o['x'] > self.pos_x:
                        if (o['heading'] == 0 and o['y'] <= self.pos_y) or (
                                o['heading'] == 180 and o['y'] >= self.pos_y) or (
                                o['heading'] == 90 and int(o['y']) == int(self.pos_y)):
                            closest_candidates.append(self._getRobot(o['label']))

                elif self.heading == 0:
                    # care robot at top
                    if o['y'] > self.pos_y:
                        if (o['heading'] == 90 and o['x'] <= self.pos_x) or (
                                o['heading'] == 270 and o['x'] >= self.pos_x) or (
                                o['heading'] == 0 and int(o['x']) == int(self.pos_x)):
                            closest_candidates.append(self._getRobot(o['label']))

                elif self.heading == 180:
                    # care robot at bottom
                    if o['y'] < self.pos_y:
                        if (o['heading'] == 90 and o['x'] <= self.pos_x) or (
                                o['heading'] == 270 and o['x'] >= self.pos_x) or (
                                o['heading'] == 180 and int(o['x']) == int(self.pos_x)):
                            closest_candidates.append(self._getRobot(o['label']))

        return closest_candidates

    def _getRobot(self, robot_name):
        for o in self.universe._objects:
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o

    def appliedTrafficPolicyKeys(self):
        result = []
        for tp in self.traffic_policy:
            result.append(tp.prioritized_robot)
        return result

    def hasTrafficPolicyFor(self, robot_name):
        for robot in self.appliedTrafficPolicyKeys():
            if robot == robot_name:
                return True
        return False

    def addTrafficPolicy(self, traffic_policy: TrafficPolicy):
        if not self.hasTrafficPolicyFor(traffic_policy.prioritized_robot):
            self.traffic_policy.append(traffic_policy)

    def deadlockPrevention(self):
        self_next_blocks = self._calculateNextBlocks(int(self.pos_x), int(self.pos_y), self.heading, 5)
        for p in self_next_blocks:
            if self.universe.deadlock_prevention_manager.coordinateBooked(NetLogoCoordinate(p[0], p[1])):
                self.acceleration = -1
                return True
        return False

    def trafficPolicy(self):
        """
        This trafficPolicy function returns True if robot has applied traffic policy
        otherwise, it returns False
        traffic policy is applied when future collision is detected with specific rules
        traffic policy is removed when the collision is no longer detected based on specific rules
        """

        # if robot has traffic policy, then check if the traffic policy is still valid
        if len(self.traffic_policy) > 0:
            index = 0

            # iterate all traffic policies and check if the collision is still detected
            while index < len(self.traffic_policy):
                tp: TrafficPolicy = self.traffic_policy[index]
                prioritized_robot: Robot = self._getRobot(tp.prioritized_robot)

                prioritized_robot_next_blocks = self._calculateNextBlocks(int(prioritized_robot.pos_x),
                                                                          int(prioritized_robot.pos_y),
                                                                          prioritized_robot.heading, 5)
                self_next_blocks = self._calculateNextBlocks(tp.collision_block.x, tp.collision_block.y, self.heading,
                                                             5)

                # if the two objects are heading to the same direction, then as long as the prioritized robot is in front of this robot, then it is safe to remove the traffic policy
                # if the two objects are in different heading direction, resolve traffic policy after the two objects do not have collision block
                if self.heading == prioritized_robot.heading:
                    if self.velocity < prioritized_robot.velocity:
                        del self.traffic_policy[index]
                else:
                    collision_block = self._getIntersectionBlock(self_next_blocks, prioritized_robot_next_blocks)
                    if collision_block is None:
                        del self.traffic_policy[index]

                index += 1

            if len(self.traffic_policy) == 0:
                self.acceleration = 1
                return False
            return True
        else:
            for c in self.getCollisionCandidates(exclude=self.appliedTrafficPolicyKeys()):
                collision_candidate: Robot = c

                # check if collision candidate has traffic policy for this robot
                if not collision_candidate.hasTrafficPolicyFor(self.robotName()):
                    self_next_blocks = self._calculateNextBlocks(int(self.pos_x), int(self.pos_y), self.heading, 5)
                    object_next_blocks = self._calculateNextBlocks(int(collision_candidate.pos_x),
                                                                   int(collision_candidate.pos_y),
                                                                   collision_candidate.heading, 5)

                    collision_block: tuple[int, int] = self._getIntersectionBlock(self_next_blocks, object_next_blocks)
                    if collision_block is not None:
                        object_coor = NetLogoCoordinate(collision_candidate.pos_x, collision_candidate.pos_y)
                        self_coor = NetLogoCoordinate(self.pos_x, self.pos_y)

                        self_distance_to_collision_block = self._calculateTwoPoint(self_coor,
                                                                                   NetLogoCoordinate(collision_block[0],
                                                                                                     collision_block[
                                                                                                         1]))
                        other_object_distance_to_collision_block = self._calculateTwoPoint(object_coor,
                                                                                           NetLogoCoordinate(
                                                                                               collision_block[0],
                                                                                               collision_block[1]))

                        each_other_distance = self._calculateTwoPoint(self_coor, object_coor)
                        self_should_decelerate = (self.velocity ** 2) / 2 >= self_distance_to_collision_block - 1.5

                        # prepare traffic policy object
                        tp = TrafficPolicy()
                        tp.collision_block = NetLogoCoordinate(collision_block[0], collision_block[1])
                        tp.involved_robots = [self.robotName(), collision_candidate.robotName()]
                        tp.prioritized_robot = collision_candidate.robotName()

                        if self.heading != collision_candidate.heading:
                            # if the two objects are heading to different direction and this object is further to collision block and it is time to decelerate (by equation/formula), then decelerate
                            if self_distance_to_collision_block >= other_object_distance_to_collision_block and self_should_decelerate:
                                self.acceleration = -1
                                self.addTrafficPolicy(tp)
                        else:
                            # if the two objects are heading to the same direction, just make sure object that is behind keeps appropriate distance gap
                            if (self.velocity ** 2) / 2 >= each_other_distance - 1:
                                self.acceleration = -1
                                self.addTrafficPolicy(tp)

        return len(self.traffic_policy) > 0

    @staticmethod
    def _transformRouteToList(path):
        path_int = []
        for p in path:
            l = p.split(',')
            path_int.append([int(l[0]), int(l[1])])
        return path_int

    def neutralizeRobotState(self):
        self.job.is_active = False
        self.route_stop_points = []

    def update_current_position(self):
        self.coordinate = NetLogoCoordinate(int(self.pos_x), int(self.pos_y))
        self.pos_x = int(self.pos_x)
        self.pos_y = int(self.pos_y)

    def picking_item_in_pod(self):
        if self.job is None or not self.job.is_active:
            return False
        else:
            if self.is_in_station():
                self.job.picking_delay -= 1

                if self.job.picking_delay == 0:
                    self.current_state = "returning_pod"
                    self.set_move(self.job.pod_coordinate, self.universe.graph_pod, need_neutralize_robot=True)

                return True

    def is_in_station(self):
        return self.pos_x is self.job.station_coordinate.x and self.pos_y is self.job.station_coordinate.y

    def shouldMoveToDestination(self):
        tp = self.trafficPolicy()
        return len(self.route_stop_points) > 0 and tp == False

    def movementPlan(self):
        if self.picking_item_in_pod() is True:
            return
        if self.shouldMoveToDestination():
            next_destination_coordinate = self.route_stop_points[0]
            if isinstance(next_destination_coordinate, Heading):
                self.heading = next_destination_coordinate.getHeading()
                self.turning += 1
                self.route_stop_points.pop(0)
                return

            if isinstance(next_destination_coordinate, NetLogoCoordinate):
                now_coor = NetLogoCoordinate(self.pos_x, self.pos_y)

                if self.close_enough(next_destination_coordinate, 0.25):
                    # stop robot and advance to next stop points
                    self.pos_x = int(next_destination_coordinate.x)
                    self.pos_y = int(next_destination_coordinate.y)
                    self.coordinate = NetLogoCoordinate(self.pos_x, self.pos_y)
                    self.velocity = 0
                    self.acceleration = 0
                    self.route_stop_points.pop(0)

                    # check if robot is done doing movement plan
                    if len(self.route_stop_points) == 0:
                        # set robot to correct position and route to destination if any
                        if self.current_state == "taking_pod":
                            self.update_current_position()
                            self.set_move_to_station()
                        elif self.current_state == "returning_pod":
                            self.update_current_position()

                        # advance robot to next state
                        self.advanceState()
                else:
                    self.acceleration = 1

                    # if robot is close enough to next destination, then decelerate
                    if (self.velocity ** 2) / 2 >= self._calculateTwoPoint(now_coor, next_destination_coordinate):
                        self.acceleration = -1

        self.drawNextPosition()

    def move(self):
        self.changeColorByState()

        self.movementPlan()

        self.latest_tick += 1

    def drawNextPosition(self):
        initial_velocity = self.velocity
        initial_acceleration = self.acceleration

        self.energy_consumption += self.calculateEnergy(initial_velocity, initial_acceleration)

        if self.velocity != 0:
            distance_delta = self.velocity * self.universe.tick_to_second
            if self.heading == 0:
                self.pos_y += distance_delta
            elif self.heading == 180:
                self.pos_y -= distance_delta
            elif self.heading == 90:
                self.pos_x += distance_delta
            elif self.heading == 270:
                self.pos_x -= distance_delta
        self.coordinate = NetLogoCoordinate(int(self.pos_x), int(self.pos_y))

        if self.acceleration != 0:
            self.velocity += (self.acceleration * self.universe.tick_to_second)
            self.velocity = max(0, min(self.maximum_speed, self.velocity))

        # for traffic policy purposes, report states to the manager
        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading)

    def assign_job_and_set_move_to_take_pod(self, job: RobotJob):
        self.job = job

        self.set_move_to_take_pod()

    def set_move_to_take_pod(self):
        self.set_move(self.job.pod_coordinate, graph=self.universe.graph, need_neutralize_robot=False)
        self.current_state = "taking_pod"

    def set_move_to_station(self):
        self.set_move(self.job.station_coordinate, graph=self.universe.graph_pod, need_neutralize_robot=False)

    def set_move(self, dest: NetLogoCoordinate, graph, need_neutralize_robot: bool):
        start = self.coordinate_to_string_key(self.pos_x, self.pos_y)
        end = self.coordinate_to_string_key(dest.x, dest.y)

        if need_neutralize_robot:
            self.neutralizeRobotState()

        node_routes = graph.dijkstra(start, end)
        self.setPath(self._transformRouteToList(node_routes))

    # utility functions
    @staticmethod
    def getHeading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        if p1.x == p2.x:
            if p1.y > p2.y:
                return 180
            else:
                return 0
        elif p1.y == p2.y:
            if p1.x > p2.x:
                return 270
            else:
                return 90

    @staticmethod
    def _calculateTwoPoint(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def close_enough(self, p: NetLogoCoordinate, precision=0.25):
        self_coor = NetLogoCoordinate(self.pos_x, self.pos_y)
        return self._calculateTwoPoint(self_coor, p) < precision

    @staticmethod
    def ensure_coordinate(number):
        if isinstance(number, int):
            print(f"{number} is an integer.")
        elif isinstance(number, float) and number.is_integer():
            print(f"{number} is a float with 0 precision.")
        else:
            print(f"{number} is not a valid integer or float with 0 precision.")

    @staticmethod
    def get_decimal(number):
        subtractor = int(number)
        return number - subtractor

    def robotName(self):
        return f"robot-{self._id}"

    @staticmethod
    def robotID(robot_name):
        return int(robot_name.split('-')[1])

    @staticmethod
    def _calculateNextBlocks(x, y, heading, block_count=5):
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

    @staticmethod
    def coordinate_to_string_key(x: int, y: int):
        return "{},{}".format(x, y)

    @staticmethod
    def _getIntersectionBlock(blocks_1, blocks_2):
        for p in blocks_1:
            if p in blocks_2:
                return p
