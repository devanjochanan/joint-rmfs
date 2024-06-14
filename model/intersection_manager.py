from typing import List, Optional

from engine.deep_q_network import DeepQNetwork
from model.intersection import Intersection


class IntersectionManager:
    def __init__(self):
        self.intersections: List[Intersection] = []
        self.coordinate_to_intersection = {}
        self.intersection_id_to_intersection = {}
        self.q_models = {}
        self.previous_state = None
        self.previous_action = None

    def add_intersection(self, intersection: Intersection):
        self.intersections.append(intersection)
        coordinate = intersection.intersection_coordinate
        self.coordinate_to_intersection[(coordinate.x, coordinate.y)] = intersection
        self.intersection_id_to_intersection[intersection.intersection_id] = intersection

    def get_intersection_by_coordinate(self, x, y):
        return self.coordinate_to_intersection.get((x, y), None)

    def get_connected_intersections(self, current_intersection: Intersection):
        x, y = current_intersection.intersection_coordinate.x, current_intersection.intersection_coordinate.y
        connected_intersections = []

        # Check the intersection to the right
        if (x + 6, y) in self.coordinate_to_intersection:
            connected_intersections.append(self.coordinate_to_intersection[(x + 6, y)])

        # Check the intersection to the left
        if (x - 6, y) in self.coordinate_to_intersection:
            connected_intersections.append(self.coordinate_to_intersection[(x - 6, y)])

        # Check the intersection above
        if (x, y + 3) in self.coordinate_to_intersection:
            connected_intersections.append(self.coordinate_to_intersection[(x, y + 3)])

        # Check the intersection below
        if (x, y - 3) in self.coordinate_to_intersection:
            connected_intersections.append(self.coordinate_to_intersection[(x, y - 3)])

        return connected_intersections

    def get_state(self, current_intersection: Intersection, tick):
        state = []
        connected_intersections = [current_intersection] + self.get_connected_intersections(current_intersection)
        for intersection in connected_intersections:
            total_energy_horizontal, total_energy_vertical = intersection.calculate_total_energy_per_direction()
            average_wait_horizontal, average_wait_vertical = intersection.calculate_average_waiting_time_per_direction(
                tick)
            state.extend([
                intersection.get_allowed_direction_code(),
                intersection.last_changed_tick,
                intersection.duration_since_last_change(tick),
                intersection.robot_count(),
                len(intersection.horizontal_robots),
                len(intersection.vertical_robots),
                average_wait_horizontal,
                average_wait_vertical,
                total_energy_horizontal,
                total_energy_vertical,
            ])
        return state

    def handle_model(self, intersection: Intersection, tick):
        state = self.get_state(intersection, tick)
        self.previous_state = state
        if intersection.intersection_id not in self.q_models:
            self.q_models[intersection.intersection_id] = self.create_new_model(intersection)
        model = self.q_models[intersection.intersection_id]
        action = model.act(state)
        self.previous_action = action
        new_direction = intersection.get_allowed_direction_by_code(action)
        self.update_allowed_direction(intersection.intersection_id, new_direction, tick)

    def create_new_model(self, intersection: Intersection):
        connected_intersections = self.get_connected_intersections(intersection)
        return DeepQNetwork(state_size=len(connected_intersections) * 10 + 10,
                            action_size=3,
                            model_name=intersection.intersection_id)

    def update_allowed_direction_using_q_model(self, tick):
        for intersection in self.intersections:
            if intersection.robot_count() > 0:
                self.handle_model(intersection, tick)

    def update_model_after_execution(self, tick):
        for intersection in self.intersections:
            self.remember_and_replay(intersection, self.calculate_reward(intersection),
                                     self.is_episode_done(intersection, tick), tick)

    def remember_and_replay(self, intersection: Intersection, reward, done, tick):
        if intersection.intersection_id not in self.q_models:
            return

        model = self.q_models[intersection.intersection_id]
        if self.previous_state is not None and self.previous_action is not None:
            next_state = self.get_state(intersection, tick)
            model.remember(self.previous_state, self.previous_action, reward, next_state, done)
            if done:
                model.replay(64)

            self.reset_previous_state_and_action()

        if tick % 1000 == 0 and tick != 0:
            model.save_model(intersection.intersection_id, tick)

    def reset_previous_state_and_action(self):
        self.previous_state = None
        self.previous_action = None

    @staticmethod
    def is_episode_done(intersection: Intersection, tick):
        if intersection.robot_count() == 0:
            return True
        elif int(tick) % 1000 == 0:
            return True
        else:
            return False

    @staticmethod
    def calculate_reward(intersection: Intersection):
        total_energy_horizontal, total_energy_vertical = intersection.calculate_total_energy_per_direction()
        total_energy = total_energy_horizontal + total_energy_vertical
        # Negative reward as we want to minimize energy consumption
        reward = -total_energy
        return reward

    def update_allowed_direction(self, intersection_id, direction, tick):
        intersection: Intersection = self.find_intersection_by_id(intersection_id)
        intersection.change_traffic_light(direction, tick)

    def find_intersection_by_path_coordinate(self, x: int, y: int) -> Optional[str]:
        for intersection in self.intersections:
            if (x, y) in intersection.approaching_path_coordinates:
                return intersection.intersection_id
        return None

    def find_intersection_by_id(self, intersection_id):
        return self.intersection_id_to_intersection.get(intersection_id, None)

    def print_info(self, x, y):
        intersection = self.coordinate_to_intersection.get((x, y))
        if intersection is not None:
            intersection.print_info()
