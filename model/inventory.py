import csv
import os
from typing import Optional, List

import pandas as pd

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from .intersection_manager import IntersectionManager
from .order import Order
from .order_manager import OrderManager
from .pod import Pod
from .pod_manager import PodManager
from .robot import Robot
from .robot_job import RobotJob
from .station_manager import StationManager


class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stop_and_go = 0
    total_energy = 0
    total_pod = 0
    total_turning = 0
    movement_channel = {}
    graph = None
    graph_pod = None

    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station", "way-direction"]
        self.tick_to_second = 0.25
        self.job_queue = []
        self.landscape = Landscape(self.dimension)
        self.pod_manager = PodManager()
        self.station_manager = StationManager()
        self.order_manager = OrderManager()
        self.next_process_tick = 0
        self.intersection_manager = IntersectionManager()
        self.update_intersection_using_RL = False

        super().__init__()

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod + 1
            self.total_pod += 1

        super().addObject(object)

    def addTrafficPolicyHistory(self, sender, target):
        if target not in self.movement_channel:
            self.movement_channel[target] = []

        self.movement_channel[target].append(sender)

    def getTrafficPolicyHistory(self, target):
        if target not in self.movement_channel:
            return []

        return self.movement_channel[target]

    def tick(self):
        self.movement_channel = {}
        if int(self._tick) == self.next_process_tick:
            self.find_new_orders()
            self.process_orders()
            if self.update_intersection_using_RL:
                self.intersection_manager.update_allowed_direction_using_q_model(int(self._tick))
        if len(self.job_queue) > 0:
            current_distance = 1000000
            nearest_robot: Optional[Robot] = None

            for o in self.get_movable_objects():
                if len(self.job_queue) > 0:
                    job: RobotJob = self.job_queue[0]

                    if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculateDistance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist

            if nearest_robot is not None:
                job: RobotJob = self.job_queue.pop(0)
                nearest_robot.assign_job_and_set_move_to_take_pod(job)

        total_energy = 0
        total_turning = 0
        for o in self.get_movable_objects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    self.finish_orders_in_job(o.job)

                if o.current_state == 'idle' and o.job is not None:
                    self.pod_manager.mark_pod_available(o.job.pod_coordinate)
                    o.job = None

        self.total_energy = total_energy
        self.total_turning = total_turning

        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.update_model_after_execution(self._tick)

        self._tick += self.tick_to_second

    # def finish_orders_in_job(self, job: RobotJob):
    #     for order_id, sku, quantity in job.orders:
    #         order: Order = self.order_manager.get_order_by_id(order_id)
    #         order.deliver_quantity(sku, quantity)

    #         station = self.station_manager.get_station_by_id(order.station_id)

    #         # Assign to csv
    #         assign_order_df = pd.read_csv('assign_order.csv')
    #         assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
    #         assign_order_df.to_csv('assign_order.csv', index=False)

    #         if order.is_order_completed():
    #             # order.complete_order(int(self._tick))
    #             self.order_manager.finish_order(order_id, int(self._tick))
    #             station = self.station_manager.get_station_by_id(order.station_id)
    #             station.remove_order(order_id)
    #             self.insert_finished_order_to_csv(order)


    #         orders_in_station: List[Order] = station.get_orders_in_station()
    #         pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)

    #         for order_in_station in orders_in_station:
    #             if order_in_station.has_sku(sku) and order_in_station.order_id != order_id:
    #                 quantity_to_take_other = order_in_station.get_quantity_left_for_sku(sku)
                    
    #                 if pod.get_quantity(sku) > quantity_to_take_other:
    #                     order_in_station.deliver_quantity(sku, quantity_to_take_other)
    #                     # pod.pick_sku(sku,quantity_to_take_other)
                        
    #                     if order_in_station.is_order_completed():
    #                         # order_in_station.complete_order(int(self._tick))
    #                         self.order_manager.finish_order(order_id, int(self._tick))
    #                         station.remove_order(order_id)
    #                         self.insert_finished_order_to_csv(order)

    #         station.remove_pod(pod.pod_id)
    #         job.is_finished = True

    def finish_orders_in_job(self, job: RobotJob):
        pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        sku_need_replenished = {}
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.get_order_by_id(order_id)
            order.deliver_quantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)
            # station = self.station_manager.get_station_by_id(order.station_id)  
            # # Suba stract the sku from the skus_in_station
            # station.subtract_sku_in_station(sku, quantity)

            pod.pick_sku(sku, quantity)

            # Check for SKU Replenishment
            self.pod_manager.reduce_sku_data(sku, quantity)
            sku, replenished_status = self.pod_manager.is_sku_need_replenished(sku, 0.8)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished[sku] = True

            # Check for pod Replenishment
            replenished_pod = self.pod_manager.get_pod_need_replenished_by_sku(sku)

            assign_order_df = pd.read_csv('assign_order.csv')
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
            assign_order_df.to_csv('assign_order.csv', index=False)
            
            if order.is_order_completed():
                self.order_manager.finish_order(order_id, int(self._tick))
                station = self.station_manager.get_station_by_id(order.station_id)
                station.remove_order(order_id,order)
                self.insert_finished_order_to_csv(order)

        #trigger Check replenishment
        job.is_finished = True

    def insert_finished_order_to_csv(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.order_id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        self.write_to_csv("order-finished.csv", header, data)

    def find_new_orders(self):
        orders_df = pd.read_csv('generated_order.csv')

        file_path = 'assign_order.csv'
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
            # pass
        else:
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv('assign_order.csv', index=False)      

        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = orders_df[(orders_df['order_arrival']<= current_second) & 
                               (orders_df['order_arrival'] > previous_second) &
                               (orders_df['order_arrival'] != 0)]

        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = Order(order_id=order_id, order_arrival=current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.add_sku(item['item_id'], item['item_quantity'])

            self.order_manager.add_order(order)

        return new_orders

    def assign_job_to_available_robot(self, job: RobotJob):
        current_distance = 1000000
        current_id = -1

        for o in self.get_movable_objects():
            if isinstance(o, Robot) and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.job_queue.append(job)
            return

        for o in self.get_movable_objects():
            if o.id == current_id and isinstance(o, Robot):
                o.assign_job_and_set_move_to_take_pod(job)

    def get_movable_objects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def process_orders(self):
        robots_location = []
        for o in self.get_movable_objects():
            if len(self.job_queue) > 0:
                job: RobotJob = self.job_queue[0]

                if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                    robots_location.append([o.pos_x, o.pos_y])

        for order in self.order_manager.unfinished_orders:
            assign_order_df = pd.read_csv('assign_order.csv')
            if order.station_id is None:
                # available_station = self.station_manager.find_available_picking_station()
                available_station = self.station_manager.find_highest_similarity_station(order.skus, self.pod_manager)
                if available_station is not None:
                    order.assign_station(available_station.station_id)
                    available_station.add_order(order.order_id, order)

                    assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'assigned_station'] = available_station.station_id
                    assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'status'] = -1
                else:
                    break

            if order.process_start_time <= 0:
                order.start_processing(int(self._tick))

            assign_order_df.to_csv('assign_order.csv', index=False)

            
            # Get the station assigned to this order and orders in that station
            order_station = self.station_manager.get_station_by_id(order.station_id)
            orders_in_station = order_station.get_orders_in_station()

            # For Emily {A:10, B:5, C:12}
            skus_in_station = order_station.get_skus_in_station()

            # For Jhen {A:[5,5], B:[5], C:[3,4,5]}
            skus_in_station_dict = order_station.get_skus_in_station_dict()
            
            station_coordinate = order_station.coordinate
            for sku in order.get_remaining_skus():
                # available_pod: Pod = self.pod_manager.get_available_pod(sku)
                # available_pod: Pod = self.pod_manager.get_available_pod_similarity(sku, skus_in_station, station_coordinate)
                available_pod: Pod = self.pod_manager.get_available_pod_inventory(sku, skus_in_station_dict, station_coordinate, robots_location)
                if available_pod is None:
                    continue
                quantity_to_take = order.get_quantity_left_for_sku(sku)
                order.commit_quantity(sku, quantity_to_take)

                # Commiting every order that has the sku in the pod chosen
               
                
                 # Append pod to station
                order_station.add_pod(available_pod.pod_id)
                available_pod.station = order_station

                assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'assigned_pod'] = int(available_pod.pod_id)
                
                assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 0

                assign_order_df.to_csv('assign_order.csv', index=False)

                job = RobotJob(available_pod.coordinate, station_id=order.station_id)
                self.pod_manager.mark_pod_not_available(available_pod.coordinate)
                # print(f"sku {sku} quantity {quantity_to_take}")

                job.add_picking_task(order.order_id, sku, quantity_to_take) # Simple kan disini ya beb
                for order_ in orders_in_station:
                    if order_ != order and order_.has_sku(sku):
                            quantity_to_take_other = order.get_quantity_left_for_sku(sku)
                            if available_pod.get_quantity(sku) > quantity_to_take_other and quantity_to_take > 0:
                                # print(f"sku {sku} quantity {quantity_to_take}")
                                job.add_picking_task(order_.order_id, sku,quantity_to_take_other)

                self.job_queue.append(job)

    def write_to_csv(self, filename, header, data):
        folder_path = os.path.join("result", self.landscape.current_date_string)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, filename)
        file_exists = os.path.exists(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)

            writer.writerow(data)
