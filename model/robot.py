import math
from typing import Optional, List

from engine.heading import Heading
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from .intersection import Intersection
from .robot_job import RobotJob
from .station import Station
from .traffic_policy import TrafficPolicy
from .zone import Zone


class Robot(Object):
    # netlogo related
    shape = 'turtle-2'
    object_type = 'robot'
    _id = 0

    # movement related
    coordinate = None
    maximum_speed = 1.5
    current_state = 'idle'
    energy_consumption = 0
    turning = 0
    heading = 0
    suspend_movement = 0
    route_stop_points = []

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
        self.turning_delay = 0
        self.taking_pod_delay = 0
        self.delay_per_task = 10
        self.idle_time = 0
        self.current_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_start_time = None
        self.current_intersection_finish_time = None
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

    def setPath(self, path):
        current_heading = self.heading
        route_stop_points = []

        # convert path list to route_stop_points (list of NetLogoCoordinate where the robot should stop and Heading to turn the robot)
        for i in range(1, len(path), 1):
            p1 = NetLogoCoordinate(path[i - 1][0], path[i - 1][1])
            p2 = NetLogoCoordinate(path[i][0], path[i][1])
            heading = self.getHeading(p1, p2) # ini dapet turning or not keknya
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
        elif self.current_state == "station_processing":
            self.color = 94  # brown
        elif self.current_state == "idle":
            self.color = 0  # black

    def advance_state(self):
        if self.current_state == "taking_pod":
            self.taking_pod_delay += self.delay_per_task
            self.current_state = "delivering_pod"
        elif self.current_state == "delivering_pod":
            self.current_state = "station_processing"
        elif self.current_state == "station_processing":
            self.current_state = "returning_pod"
        elif self.current_state == "returning_pod":
            self.taking_pod_delay += self.delay_per_task
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

    def get_nearest_robot_conflict_candidate(self, next_step_coords, search_area):
        neighbor_candidates = []

        neighbors = self.universe.landscape.getNeighborObject(round(self.pos_x), round(self.pos_y), search_area)
        if neighbors:
            for neighbor in neighbors:
                neighbor_robot = self.get_robot_by_name(neighbor['label'])
                if neighbor_robot == self:
                    continue

                # if neighbor['velocity'] == 0:
                #     continue

                neighbor_next_step_coords = self._calculate_next_blocks(round(neighbor['x']), round(neighbor['y']),
                                                                        neighbor['heading'], search_area,
                                                                        include_self=True)
                meeting_coordinate = self._getIntersectionBlock(next_step_coords, neighbor_next_step_coords)
                if meeting_coordinate and self.is_collision_candidate(neighbor):
                    neighbor_candidates.append((neighbor, meeting_coordinate))

        return neighbor_candidates

    def get_priority_diff(self, object):
        state_priority = {
            'station_processing': 3,
            'delivering_pod': 3,
            'returning_pod': 2,
            'taking_pod': 1,
            'idle': 0
        }

        self_priority = state_priority[self.current_state]
        other_priority = state_priority[object['state']]

        return self_priority - other_priority

    def is_collision_candidate(self, obj):
        # Check for collision candidate based on relative positions and headings
        relative_x = obj['x'] - self.pos_x
        relative_y = obj['y'] - self.pos_y

        if self.heading == 0:
            return relative_y >= 0
        if self.heading == 180:
            return relative_y <= 0
        if self.heading == 90:
            return relative_x >= 0
        if self.heading == 270:
            return relative_x <= 0

    def get_robot_by_name(self, robot_name):
        for o in self.universe._objects:
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o

    def get_robot_by_coord(self, x, y):
        for o in self.universe._objects:
            if o.object_type == "robot" and o.pos_x == x and o.pos_y == y:
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
        self_next_blocks = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y), self.heading, 5)
        for p in self_next_blocks:
            if self.universe.deadlock_prevention_manager.coordinateBooked(NetLogoCoordinate(p[0], p[1])):
                self.acceleration = -1
                return True
        return False

    def should_remove_policy(self, prioritized_robot, self_blocks, other_blocks):
        if self.heading == prioritized_robot.heading:
            return self.velocity < prioritized_robot.velocity
        else:
            return self._getIntersectionBlock(self_blocks, other_blocks) is None

    @staticmethod
    def _transformRouteToList(path):
        path_int = []
        for p in path:
            l = p.split(',')
            path_int.append([int(l[0]), int(l[1])])
        return path_int

    @staticmethod
    def transform_coords_to_list(coords: List[NetLogoCoordinate]):
        path_int = []
        for coord in coords:
            path_int.append([coord.x, coord.y])

        return path_int

    def neutralizeRobotState(self):
        self.route_stop_points = []

    def update_current_position(self):
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))
        self.pos_x = round(self.pos_x)
        self.pos_y = round(self.pos_y)

    def picking_item_in_pod(self):
        if self.job is not None and self.is_being_process_on_station():
            self.job.picking_delay -= 1
            return True

    def is_in_station_path(self):
        if self.job is not None:
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
            for coord in station.get_path():
                if round(self.pos_x) == coord.x and round(self.pos_y) == coord.y:
                    return True

    def is_being_process_on_station(self):
        station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
        return self.job.picking_delay > 0 and self.close_enough(station.coordinate, 0.1)

    def movementPlan(self):
        if self.picking_item_in_pod():
            return

        if not self.route_stop_points:
            self.advance_state_if_needed()
            return

        if self.eligible_to_reroute():
            if self.current_state == "taking_pod":
                self.set_move(self.route_stop_points[-1], self.universe.graph, avoid_front=True)
            elif self.current_state == "delivering_pod" or self.current_state == "returning_pod":
                self.set_move(self.route_stop_points[-1], self.universe.graph_pod, avoid_front=True)
            elif self.current_state == "station_processing":
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                path = station.get_sub_path(round(self.pos_x), round(self.pos_y))
                self.setPath(self.transform_coords_to_list(path))
                station.update_robot_route_type(self.robotName())

            self.idle_time = 0

        testing = self.route_stop_points
        next_destination_coordinate = []
        if len(testing) > 0:
            next_destination_coordinate = self.route_stop_points[0]

        if isinstance(next_destination_coordinate, Heading):
            self.handle_directional(next_destination_coordinate)
            return

        if self.not_able_to_move(next_destination_coordinate):
            self.update_idle_state()
            return

        candidate_conflict_coordinate = self.handle_conflicts(next_destination_coordinate)

        self.execute_move(candidate_conflict_coordinate, next_destination_coordinate)

    def update_idle_state(self):
        self.idle_time += 1
        self.velocity = 0
        self.acceleration = 0
        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity,
                                          self.acceleration, self.heading, self.current_state)
        self.universe.landscape.objects.values()

    def handle_conflicts(self, next_destination_coordinate):
        candidate_conflict_coordinate = None
        if isinstance(next_destination_coordinate, NetLogoCoordinate) and not self.is_in_station_path():
            self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
            search_area = self.calculate_search_area(next_destination_coordinate)

            next_step_coordinates = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y),
                                                                self.heading, search_area)
            nearest_conflict_candidates = self.get_nearest_robot_conflict_candidate(next_step_coordinates, search_area)
            if nearest_conflict_candidates is not None:
                for candidate, meeting_coordinate in nearest_conflict_candidates:
                    if (candidate['state'] == "station_processing" or candidate['state'] == 'idle'
                            or candidate['velocity'] == 0):
                        continue

                    neighbor_coord = NetLogoCoordinate(candidate['x'], candidate['y'])
                    meeting_coordinate = NetLogoCoordinate(meeting_coordinate[0], meeting_coordinate[1])
                    self_distance_to_meeting_block = self._calculateTwoPoint(self_coord, meeting_coordinate)
                    neighbor_distance_to_meeting_block = self._calculateTwoPoint(neighbor_coord, meeting_coordinate)

                    priority_diff = self.get_priority_diff(candidate)
                    if candidate['heading'] == self.heading:
                        for x, y in next_step_coordinates:
                            if round(candidate['x']) == x and round(candidate['y']) == y:
                                candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(
                                    meeting_coordinate, next_destination_coordinate)
                                break

                    elif priority_diff < 0:
                        candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(meeting_coordinate,
                                                                                                   next_destination_coordinate)

                    elif priority_diff == 0:
                        if self_distance_to_meeting_block > neighbor_distance_to_meeting_block:
                            candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(
                                meeting_coordinate, next_destination_coordinate)

        return candidate_conflict_coordinate

    def execute_move(self, candidate_conflict_coordinate, next_destination_coordinate):
        self.idle_time = 0
        if candidate_conflict_coordinate and candidate_conflict_coordinate != next_destination_coordinate:
            self.handle_next_movement(candidate_conflict_coordinate, is_next_route_stop=False)
        else:
            self.handle_next_movement(next_destination_coordinate, is_next_route_stop=True)

        self.drawNextPosition()

    def eligible_to_reroute(self):
        if self.idle_time <= 50 or self.current_state == "delivering_pod":
            return False

        if self.is_in_station_path():
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)

            if self.current_state == "station_processing" and station.has_route_changed(self.robotName()):
                return True
            else:
                return False

        # Calculate next step coordinates
        next_step_coordinates = self._calculate_next_blocks(
            round(self.pos_x), round(self.pos_y), self.heading, 1, include_self=False)
        robot_front = self.universe.landscape.get_neighbor_object(*next_step_coordinates[0])

        # Check if there is no robot in front
        if not robot_front:
            return False

        # if self.idle_time > 50 and robot_front['velocity'] == 0 and self.current_state == "returning_pod":
        #     return True 

        # Check if the robot in front is idle
        if robot_front['state'] == "idle":
            return True

        # Check if the robot in front is heading in the same direction
        if robot_front['heading'] == self.heading:
            return False

        # Compare priorities
        priority_diff = self.get_priority_diff(robot_front)
        if priority_diff > 0:
            return False
        elif priority_diff < 0:
            return True

        # Resolve ties by ID
        return self.robotID(self.robotName()) < self.robotID(robot_front['label'])

    def calculate_next_movement_from_conflict(self, conflict_coordinate: NetLogoCoordinate,
                                              next_destination_coordinate: NetLogoCoordinate):
        potential_next = None
        if self.heading == 0:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y - 1)
        elif self.heading == 180:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y + 1)
        elif self.heading == 90:
            potential_next = NetLogoCoordinate(conflict_coordinate.x - 1, conflict_coordinate.y)
        elif self.heading == 270:
            potential_next = NetLogoCoordinate(conflict_coordinate.x + 1, conflict_coordinate.y)

        if self.heading in [0, 180]:
            if abs(potential_next.y - next_destination_coordinate.y) < abs(
                    conflict_coordinate.y - next_destination_coordinate.y):
                return next_destination_coordinate
        else:
            if abs(potential_next.x - next_destination_coordinate.x) < abs(
                    conflict_coordinate.x - next_destination_coordinate.x):
                return next_destination_coordinate

        return potential_next

    def calculate_search_area(self, next_destination_coordinate: NetLogoCoordinate):
        if self.is_in_station_path():
            return 1

        return 3

    def not_able_to_move(self, next_destination_coordinate: NetLogoCoordinate):
        if self.turning_delay > 0:
            self.turning_delay -= 1
            return True

        if self.taking_pod_delay > 0:
            self.taking_pod_delay -= 1
            return True

        if next_destination_coordinate.x == round(self.pos_x) and next_destination_coordinate.y == round(self.pos_y):
            return False

        return self.path_blocked()

    def path_blocked(self):
        next_step_coordinates = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y),
                                                            self.heading, 1, include_self=False)

        if not self.is_aligned_with_heading(next_step_coordinates):
            return False

        return (self.path_blocked_by_intersection(next_step_coordinates)
                or self.path_blocked_by_robot(next_step_coordinates))

    def path_blocked_by_intersection(self, next_step_coordinates):
        for next_x, next_y in next_step_coordinates:
            intersection = self.universe.intersection_manager.get_intersection_by_coordinate(next_x, next_y)
            if (intersection and self.close_enough(intersection.intersection_coordinate, 1)
                    and not intersection.is_allowed_to_move(self.heading)):
                return True
        return False

    def path_blocked_by_robot(self, next_step_coordinates):
        # if round(self.pos_y) == 58:
        #     print(self.robotName())

        neighbors = self.universe.landscape.getNeighborObject(round(self.pos_x), round(self.pos_y), 2)
        for neighbor in neighbors:
            if self.get_robot_by_name(neighbor['label']) == self:
                continue

            near_robot_coord = NetLogoCoordinate(neighbor['x'], neighbor['y'])

            for next_x, next_y in next_step_coordinates:
                x_difference = abs(neighbor['x'] - next_x)
                y_difference = abs(neighbor['y'] - next_y)
                if x_difference < 1 and y_difference < 1 and self.close_enough(near_robot_coord, 1):
                    self_distance_to_conflict = abs(self.pos_x - next_x) + abs(self.pos_y - next_y)
                    neighbor_robot_distance_to_conflict = abs(neighbor['x'] - next_x) + abs(neighbor['y'] - next_y)

                    if neighbor_robot_distance_to_conflict < self_distance_to_conflict:
                        return True
                    else:
                        continue

        return False

    def is_in_path(self, destination, steps):
        # Assuming steps is a list of tuples (x, y) coordinates
        current_x, current_y = round(self.pos_x), round(self.pos_y)
        dest_x, dest_y = destination.x, destination.y

        # Check if destination is on the path between current and next step
        for step_x, step_y in steps:
            if self.heading == 0:  # Moving up along the y-axis
                if current_y <= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y <= step_y <= dest_y
            elif self.heading == 180:  # Moving down along the y-axis
                if current_y >= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y >= step_y >= dest_y
            elif self.heading == 90:  # Moving right along the x-axis
                if current_x <= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x <= step_x <= dest_x
            elif self.heading == 270:  # Moving left along the x-axis
                if current_x >= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x >= step_x >= dest_x
        return False

    def is_aligned_with_heading(self, steps):
        for step_x, step_y in steps:
            if self.heading in (0, 180):  # Vertical movement
                return round(self.pos_x) == step_x
            elif self.heading in (90, 270):  # Horizontal movement
                return round(self.pos_y) == step_y

    def get_robots_by_coords(self, coords):
        robots = []
        for coord in coords:
            robot = self.get_robot_by_coord(coord[0], coord[1])
            if robot:
                robots.append(robot)
        return robots

    def handle_directional(self, heading):
        angular_change = min(abs(heading.getHeading() - self.heading),
                             360 - abs(heading.getHeading() - self.heading)) // 90
        self.turning_delay += self.delay_per_task * angular_change

        self.heading = heading.getHeading()
        self.turning += 1
        self.route_stop_points.pop(0)

    def handle_next_movement(self, next_destination_coordinate, is_next_route_stop=True):
        current_coord = NetLogoCoordinate(self.pos_x, self.pos_y)

        if self.close_enough(next_destination_coordinate, 0.3):
            self.update_position(next_destination_coordinate)
            if is_next_route_stop:
                self.route_stop_points.pop(0)
        else:
            self.update_motion_parameters(current_coord, next_destination_coordinate)

    def update_position(self, coordinate):
        # Update robot's position and movement parameters to match the intersection_coordinate
        self.pos_x = round(coordinate.x)
        self.pos_y = round(coordinate.y)
        self.coordinate = NetLogoCoordinate(self.pos_x, self.pos_y)
        self.velocity = 0
        self.acceleration = 0

    def advance_state_if_needed(self):
        # Check if all route points are done and handle state transitions
        if not self.route_stop_points:
            self.advance_state()
            self.update_current_position()
            if self.current_state == "delivering_pod":
                self.set_move_to_station_gate()
            elif self.current_state == "returning_pod":
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                station.remove_robot(self.robotName())
                self.set_move(self.job.pod_coordinate, self.universe.graph_pod, need_neutralize_robot=True)
            elif self.current_state == "station_processing":
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                station.add_robot(self.robotName())
                self.setPath(self.transform_coords_to_list(station.get_path()))

        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state)

    def update_motion_parameters(self, current_coord, next_destination_coordinate):
        # Adjust robot's acceleration based on proximity to the next intersection_coordinate
        self.acceleration = 1
        deceleration_buffer = 0.5
        distance_to_stop = self._calculateTwoPoint(current_coord, next_destination_coordinate)
        if (self.velocity ** 2) / (2 * deceleration_buffer) >= distance_to_stop:
            self.acceleration = -1

    def move(self):
        self.changeColorByState()

        self.movementPlan()

        self.latest_tick += 1

    def drawNextPosition(self):
        initial_velocity = self.velocity
        initial_acceleration = self.acceleration

        energy = self.calculateEnergy(initial_velocity, initial_acceleration)
        if self.robotName() == 'robot-1':
            print(energy)
        self.energy_consumption += energy

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
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))

        if self.acceleration != 0:
            self.velocity += (self.acceleration * self.universe.tick_to_second)
            self.velocity = max(0, min(self.maximum_speed, self.velocity))

        # for traffic policy purposes, report states to the manager
        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state)
        self.update_intersection_information(energy)

    def update_intersection_information(self, energy):
        intersection_id = self.universe.intersection_manager.find_intersection_by_path_coordinate(round(self.pos_x),
                                                                                                  round(self.pos_y))
        if intersection_id:
            if self.current_intersection_id == intersection_id:
                self.update_current_intersection(energy)
            else:
                self.finalize_current_intersection()

                self.start_new_intersection(intersection_id)
        else:
            self.finalize_current_intersection()

            self.reset_intersection_tracking()

    def update_current_intersection(self, energy):
        self.current_intersection_energy_consumption += energy

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)
        intersection.update_robot(self)

    def finalize_current_intersection(self):
        if self.current_intersection_id is None:
            return

        # Mark the finish time for the current intersection
        self.current_intersection_finish_time = self.universe._tick
        # Log the intersection information to CSV

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)

        self.insert_robot_intersection_information_to_csv(intersection)

        intersection.remove_robot(self)

    def start_new_intersection(self, intersection_id):
        # Set the new intersection ID and reset the energy consumption
        self.current_intersection_id = intersection_id
        self.current_intersection_energy_consumption = 0
        self.current_intersection_start_time = self.universe._tick

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)
        intersection.add_robot(self)

    def reset_intersection_tracking(self):
        # Reset all intersection-related data
        self.current_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_start_time = 0
        self.current_intersection_finish_time = 0

    def insert_robot_intersection_information_to_csv(self, intersection: Intersection):
        header = ["robot_name", "robot_state", "robot_destination", "intersection_start_time",
                  "intersection_finish_time", "intersection_id",
                  "energy_consumption_intersection", "queueing_robot"]
        data = [self.robotName(), self.current_state, self.route_stop_points[-1], self.current_intersection_start_time,
                self.current_intersection_finish_time, self.current_intersection_id,
                self.current_intersection_energy_consumption,
                intersection.robot_count()]

        self.universe.write_to_csv("intersection-energy-consumption.csv", header, data)

    def assign_job_and_set_move_to_take_pod(self, job: RobotJob):
        self.job = job

        self.set_move_to_take_pod()

    def set_move_to_take_pod(self):
        self.set_move(self.job.pod_coordinate, graph=self.universe.graph, need_neutralize_robot=False)
        self.current_state = "taking_pod"

    def set_move_to_station_gate(self):
        station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
        self.set_move(station.get_path()[0], graph=self.universe.graph_pod, need_neutralize_robot=False)

    def set_move(self, dest: NetLogoCoordinate, graph, need_neutralize_robot: bool = False, avoid_front: bool = False):
        start = self.coordinate_to_string_key(round(self.pos_x), round(self.pos_y))
        end = self.coordinate_to_string_key(dest.x, dest.y)

        if need_neutralize_robot:
            self.neutralizeRobotState()

        nodes_to_avoid = []
        if avoid_front:
            avoid_coord = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y),
                                                      self.heading, 1, include_self=False)
            nodes_to_avoid.append(self.coordinate_to_string_key(*avoid_coord[0]))

        robot_objects = self.universe.landscape.get_robot_object()
        robots_location = [[info['x'], info['y']] for info in robot_objects.values() if info['state'] != 'station_processing']

        robots_idle_time = []
        robot_list = []
        if len(robots_location) > 0:
            robot_list = self.get_robots_by_coords(robots_location)

        for robot in robot_list:
            robots_idle_time.append(robot.idle_time)

        zones = Zone(robots_location, self.universe.get_warehouse_size(), methods="kmeans")
        penalties = zones.calculate_penalty(robots_location, robots_idle_time, self.universe.get_warehouse_size(), threshold=5)
        zone_boundary = zones.get_boundary()
        print("zone boundary:", zone_boundary)
        print("penalties:", penalties)
        # node_routes = graph.dijkstra(start, end, nodes_to_avoid)
        node_routes = graph.dijkstra_modified(start,end, penalties, zone_boundary, nodes_to_avoid)
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
        self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
        return self._calculateTwoPoint(self_coord, p) < precision

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
    def _calculate_next_blocks(x, y, heading, block_count=5, include_self=False):
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

        if include_self:
            result.append([x, y])
        for i in range(block_count):
            x += x_difference
            y += y_difference

            result.append([x, y])

        return result

    @staticmethod
    def coordinate_to_string_key(x: int, y: int):
        return "{},{}".format(x, y)

    @staticmethod
    def _getIntersectionBlock(blocks_1, blocks_2):
        for p in blocks_1:
            if p in blocks_2:
                return p
