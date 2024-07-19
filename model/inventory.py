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
    total_robot_idle = 0
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
        self.intersection_manager = IntersectionManager(self.landscape.current_date_string)
        self.update_intersection_using_RL = False
        self.zoning = False
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
        total_idle = 0
        for o in self.get_movable_objects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                total_idle += (o.total_idle * 0.15)
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finish_task_in_job(o.job)
                    if need_replenish_pod:
                        pod: Pod = self.pod_manager.get_pod_by_coordinate(o.job.pod_coordinate.x, o.job.pod_coordinate.y)
                        station_replenish = self.station_manager.find_available_replenish_station()
                        if station_replenish is not None:
                            station_replenish.add_pod(pod.pod_id)
                            new_job = RobotJob(pod.coordinate, station_id=station_replenish.station_id, pod=pod)
                            new_job.add_replenishment_task(pod)
                            # station_replenish.add_robot()
                            o.assign_job_and_set_move_to_station(new_job)
                            

                if o.current_state == 'idle' and o.job is not None:
                    self.pod_manager.mark_pod_available(o.job.pod_coordinate)
                    o.job = None
        self.total_robot_idle = total_idle
        self.total_energy = total_energy
        self.total_turning = total_turning

        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.update_model_after_execution(self._tick)

        self._tick += self.tick_to_second

    def finish_task_in_job(self, job: RobotJob):
        job_station = self.station_manager.get_station_by_id(job.station_id)
        if job_station.is_picker_station():
            return self.finish_picking_task(job)
        elif job_station.is_replenishment_station():
            return self.finish_replenishment_task(job)
    
    def finish_picking_task(self, job: RobotJob):
        pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod_info_df = pd.read_csv('pod_info.csv')
        sku_need_replenished = []
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.get_order_by_id(order_id)
            order.deliver_quantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)

            # Check for SKU Replenishment
            # sku is sku_id (String)
            self.pod_manager.reduce_sku_data(sku, quantity)
            sku, replenished_status = self.pod_manager.is_sku_need_replenished(sku)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished.append(sku)
    
            assign_order_df = pd.read_csv('assign_order.csv')
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'order_finished'] = int(self._tick)
            assign_order_df.to_csv('assign_order.csv', index=False)
            new_row = {
                "pod_id": pod.pod_id,
                "item_id": sku,
                "qty": quantity,
                "order_id": order_id,
                "processed_time": int(self._tick),
                "task_type": 1
            }
            
            new_row_df = pd.DataFrame([new_row])
            pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
            
            if order.is_order_completed():
                self.order_manager.finish_order(order_id, int(self._tick))
                station = self.station_manager.get_station_by_id(order.station_id)
                station.remove_order(order_id,order)
                self.insert_finished_order_to_csv(order)
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        
        pod_info_df.to_csv('pod_info.csv', index=False)
        # Replenishment baseline
        job.is_finished = True
        if len(sku_need_replenished) > 0:
            return True
        need_replenish_pod = pod.check_replenishment_needed()
        print(f"reple ga yaaa {need_replenish_pod}")
        return need_replenish_pod
    
    def finish_replenishment_task(self, job: RobotJob):
        pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod.replenish_all_skus()
        pod_info_df = pd.read_csv('pod_info.csv')
        new_row = {
                "pod_id": pod.pod_id,
                "item_id": -1,
                "qty": -1,
                "order_id": -999,
                "processed_time": int(self._tick),
                "task_type": 2
            }
            
        new_row_df = pd.DataFrame([new_row])
        pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
        pod_info_df.to_csv('pod_info.csv', index= False)
        job.is_finished = True
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        return False

    def insert_finished_order_to_csv(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.order_id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        self.write_to_csv("order-finished.csv", header, data)

    def find_new_orders(self):
        file_path = 'assign_order.csv'
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
            # pass
        else:
            orders_df = pd.read_csv('generated_order.csv')
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv('assign_order.csv', index=False)
        new_file_df = pd.read_csv(file_path)
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = new_file_df[(new_file_df['order_arrival']<= current_second) & 
                               (new_file_df['order_arrival'] > previous_second) &
                               (new_file_df['status'] == -3)]
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
            # print(f"order in station {orders_in_station}")
            station_coordinate = order_station.coordinate
            for sku in order.get_remaining_skus():
                if assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)) , 'status'].values[0] != 0:
                    # This is the baseline
                    # available_pod: Optional[Pod] = self.pod_manager.get_available_pod(sku) 
                    
                    # This is Emily's pod picking
                    # available_pod: Optional[Pod] = self.pod_manager.get_available_pod_similarity(sku, skus_in_station, station_coordinate, robots_location) 
                    
                    # This is Jhen's pod picking
                    available_pod: Optional[Pod] = self.pod_manager.get_available_pod_inventory(sku, order_station.skus_in_station, station_coordinate, robots_location) 
                    if available_pod is None:
                        continue
                    quantity_to_take = order.get_quantity_left_for_sku(sku)
                    # print(f"in process {order.order_id} sku {sku} qty {quantity_to_take} qty_pod {available_pod.get_quantity(sku)} pod_id {available_pod.pod_id}")
                    
                    # print(f"pod skus {[i for i in available_pod.skus]}")
                    job = RobotJob(available_pod.coordinate, station_id=order.station_id, pod=available_pod)
                    if available_pod.get_quantity(sku) > 0:
                        if available_pod.get_quantity(sku) < quantity_to_take and available_pod.get_quantity(sku) > 0:
                            quantity_to_take = available_pod.get_quantity(sku)
                        
                        order.commit_quantity(sku, quantity_to_take)

                        # Commiting every order that has the sku in the pod chosen
                        available_pod.pick_sku(sku, quantity_to_take)
                        print(f"{available_pod.get_quantity(sku)} qty pod after picked in process" )
                        
                        # Append pod to station
                        order_station.add_pod(available_pod.pod_id)
                        available_pod.station = order_station

                        assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'assigned_pod'] = int(available_pod.pod_id)
                        
                        assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 0
                        
                        assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'order_processed'] = int(self._tick)
                        assign_order_df.to_csv('assign_order.csv', index=False)

                        
                        self.pod_manager.mark_pod_not_available(available_pod.coordinate)
                        order_station.reduce_sku_from_station(sku, quantity_to_take)
                        
                        job.add_picking_task(order.order_id, sku, quantity_to_take) 
                    pod_skus = [i for i in available_pod.skus]
                    
                   
                    
                    # Turn this off for baseline 
                    for skus_pod in pod_skus:
                        for order_ in orders_in_station:
                            if order_ != order and order_.has_sku(skus_pod):
                                    quantity_to_take_other = order_.get_quantity_left_for_sku(skus_pod)
                                    # print(f"sku{skus_pod} quantity other {quantity_to_take_other} pod {available_pod.get_quantity(skus_pod)}")
                                    if available_pod.get_quantity(skus_pod) > 0 and quantity_to_take_other > 0:
                                        if quantity_to_take_other > available_pod.get_quantity(skus_pod):
                                            quantity_to_take_other = available_pod.get_quantity(skus_pod)
                                        order_.commit_quantity(skus_pod, quantity_to_take_other)
                                        # available_pod.pick_sku(sku, quantity_to_take_other)
                                        job.add_picking_task(order_.order_id, skus_pod,quantity_to_take_other)
                                        
                                        assign_order_df.loc[((assign_order_df['order_id'] == order_.order_id) & (assign_order_df['item_id'] == skus_pod)), 'assigned_pod'] = int(available_pod.pod_id)
                    
                                        assign_order_df.loc[((assign_order_df['order_id'] == order_.order_id) & (assign_order_df['item_id'] == skus_pod)), 'status'] = 0
                                        assign_order_df.loc[((assign_order_df['order_id'] == order_.order_id) & (assign_order_df['item_id'] == skus_pod)), 'order_processed'] = int(self._tick)
                                        order_station.reduce_sku_from_station(skus_pod, quantity_to_take_other)
                                        # print(f"for other order {order_.order_id} sku {skus_pod} qty {quantity_to_take_other} qty_pod {available_pod.get_quantity(skus_pod)} pod_id {available_pod.pod_id}")
                                        available_pod.pick_sku(skus_pod, quantity_to_take_other)
                                        # print(f"{available_pod.get_quantity(skus_pod)} qty pod after picked in other" )
                                        
                                        assign_order_df.to_csv('assign_order.csv', index=False)
                                        
                                        
                    if len(job.orders) > 0:
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
