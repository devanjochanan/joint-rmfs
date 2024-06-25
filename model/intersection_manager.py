from typing import List, Optional

from engine.deep_q_network import DeepQNetwork
from model.intersection import Intersection


class IntersectionManager:
    def __init__(self):
        self.intersections: List[Intersection] = []
        self.coordinate_to_intersection = {}
        self.intersection_id_to_intersection = {}
        self.q_models = {}
        self.previous_state = {}
        self.previous_action = {}

    def add_intersection(self, intersection: Intersection):
        self.intersections.append(intersection)
        coordinate = intersection.intersection_coordinate
        self.coordinate_to_intersection[(coordinate.x, coordinate.y)] = intersection
        self.intersection_id_to_intersection[intersection.intersection_id] = intersection

    def get_intersection_by_coordinate(self, x, y):
        return self.coordinate_to_intersection.get((x, y), None)

    def get_connected_intersections(self, current_intersection: Intersection):
        connected_intersections = []
        connected_intersection_ids = current_intersection.connected_intersection_ids

        for intersection_id in connected_intersection_ids:
            intersection = self.find_intersection_by_id(intersection_id)
            if intersection is not None:
                connected_intersections.append(intersection)

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
        self.previous_state[intersection.RL_model_name] = state
        if intersection.RL_model_name not in self.q_models:
            self.q_models[intersection.RL_model_name] = self.create_new_model(intersection)
        model = self.q_models[intersection.RL_model_name]
        action = model.act(state)
        self.previous_action[intersection.RL_model_name] = action
        new_direction = intersection.get_allowed_direction_by_code(action)
        self.update_allowed_direction(intersection.intersection_id, new_direction, tick)

    def create_new_model(self, intersection: Intersection):
        connected_intersections = self.get_connected_intersections(intersection)
        state_size = len(connected_intersections) * 10 + 10
        return DeepQNetwork(state_size=state_size,
                            action_size=3,
                            model_name=intersection.RL_model_name)

    def update_allowed_direction_using_q_model(self, tick):
        for intersection in self.intersections:
            if intersection.use_reinforcement_learning and intersection.robot_count() > 0:
                self.handle_model(intersection, tick)

    def update_model_after_execution(self, tick):
        for intersection in self.intersections:
            if intersection.use_reinforcement_learning and intersection.RL_model_name in self.q_models:
                self.remember_and_replay(intersection, self.calculate_reward(intersection, tick),
                                         self.is_episode_done(intersection, tick), tick)

    def remember_and_replay(self, intersection: Intersection, reward, done, tick):
        model = self.q_models[intersection.RL_model_name]
        if intersection.RL_model_name in self.previous_state and intersection.RL_model_name in self.previous_action:
            next_state = self.get_state(intersection, tick)
            model.remember(self.previous_state[intersection.RL_model_name],
                           self.previous_action[intersection.RL_model_name], reward, next_state, done)
            if done:
                model.replay(64)

            self.reset_previous_state_and_action(intersection)

        if tick % 1000 == 0 and tick != 0:
            print("SAVING_MODEL")
            model.save_model(intersection.RL_model_name, tick)

    def reset_previous_state_and_action(self, intersection: Intersection):
        if intersection.RL_model_name in self.previous_state:
            del self.previous_state[intersection.RL_model_name]
        if intersection.RL_model_name in self.previous_action:
            del self.previous_action[intersection.RL_model_name]

    @staticmethod
    def is_episode_done(intersection: Intersection, tick):
        if intersection.robot_count() == 0:
            return True
        elif int(tick) % 1000 == 0:
            return True
        else:
            return False

    @staticmethod
    def calculate_reward(intersection: Intersection, tick):
        total_energy_horizontal, total_energy_vertical = intersection.calculate_total_energy_per_direction()
        total_energy = total_energy_horizontal + total_energy_vertical
        average_wait_horizontal, average_wait_vertical = intersection.calculate_average_waiting_time_per_direction(tick)
        # Negative reward as we want to minimize energy consumption and waiting time
        reward = -total_energy - average_wait_horizontal - average_wait_vertical
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
