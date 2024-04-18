from typing import List

from model.pod import Pod


class PodManager:
    def __init__(self):
        self.pods: List[Pod] = []
        self.sku_to_pods = {}
        self.coordinate_to_pods = {}

    def add_pod(self, pod: Pod):
        self.pods.append(pod)
        self.coordinate_to_pods[(pod.pos_x, pod.pos_y)] = pod

        for sku in pod.skus:
            if sku not in self.sku_to_pods:
                self.sku_to_pods[sku] = []
            self.sku_to_pods[sku].append(pod)

    def add_sku_to_pod(self, sku: str, pod: Pod):
        if sku not in self.sku_to_pods:
            self.sku_to_pods[sku] = []
        self.sku_to_pods[sku].append(pod)

    def get_available_pod(self, sku: str):
        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                if pod.is_idle is True:
                    return pod

    def get_pods_by_sku(self, sku):
        return self.sku_to_pods.get(sku, None)

    def get_pod_by_coordinate(self, x, y):
        return self.coordinate_to_pods.get((x, y), None)
