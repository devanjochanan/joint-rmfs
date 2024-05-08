from typing import List

from model.pod import Pod
from engine import NetLogoCoordinate


class PodManager:
    def __init__(self):
        self.pods: List[Pod] = []
        self.id_to_pod = {}
        self.sku_to_pods = {}
        self.coordinate_to_pods = {}

    def add_pod(self, pod: Pod):
        self.pods.append(pod)
        self.coordinate_to_pods[(pod.pos_x, pod.pos_y)] = pod
        self.id_to_pod[pod.pod_id] = pod

        for sku in pod.skus:
            if sku not in self.sku_to_pods:
                self.sku_to_pods[sku] = []
            self.sku_to_pods[sku].append(pod)

    def add_sku_to_pod(self, sku: int, pod: Pod):
        if sku not in self.sku_to_pods:
            self.sku_to_pods[sku] = []
        self.sku_to_pods[sku].append(pod)

    def get_available_pod(self, sku: str):
        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                if pod.is_idle is True:
                    return pod

    def mark_pod_not_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        pod.is_idle = False

    def mark_pod_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        pod.is_idle = True

    def get_pods_by_sku(self, sku):
        return self.sku_to_pods.get(sku, None)

    def get_pod_by_coordinate(self, x, y):
        return self.coordinate_to_pods.get((x, y), None)

    def get_pod_by_id(self, pod_id):
        return self.id_to_pod.get(pod_id, None)
