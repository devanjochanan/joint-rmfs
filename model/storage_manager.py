from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
from model.storage import Storage
from model.pod import Pod
from engine.netlogo_coordinate import NetLogoCoordinate
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

if TYPE_CHECKING:
    from model.inventory import Inventory

class StorageManager:
    def __init__(self, warehouse: "Inventory"):
        self.warehouse = warehouse
        self.storages: List[Storage] = []
        self.storage_counter = 0
        self.pods_to_storage: Dict[Pod, Storage] = {}
        self.coordinate_to_storages: Dict[tuple[int, int], Storage] = {}
        self.empty_storages: List[Storage] = []

    def initStorageManager(self):
        for storage in self.storages:
            storage.setStorageManager(self)

    def getAllStorages(self) -> List[Storage]:
        return self.storages

    def getStorageByPod(self, pod: Pod) -> Optional[Storage]:
        return self.pods_to_storage.get(pod, None)

    def get_owned_storage_for_pod(self, pod: Pod) -> Optional[Storage]:
        """Return the storage currently owned by (assigned to) ``pod``.

        A storage is owned by the pod when it is mapped to the pod AND the
        storage's ``assigned_pod`` back-reference points to the same pod. Used to
        pin a proactive replenishment pod's exact origin so it is never handed to
        another pod while the pod is away.
        """
        storage = self.pods_to_storage.get(pod)
        if storage is None:
            return None
        if getattr(storage, "assigned_pod", None) is not pod:
            return None
        return storage

    def storage_owned_by_pod(self, storage: Storage | None, pod: Pod) -> bool:
        if storage is None or pod is None:
            return False
        return self.pods_to_storage.get(pod) is storage and getattr(storage, "assigned_pod", None) is pod

    def getStorageByCoordinate(self, x: int, y: int) -> Optional[Storage]:
        return self.coordinate_to_storages.get((x, y), None)

    def getEmptyStorage(self) -> Optional[Storage]:
        return self.empty_storages[0] if self.empty_storages else None

    def setStorageNotAvailable(self, coordinate: NetLogoCoordinate):
        storage = self.getStorageByCoordinate(coordinate.x, coordinate.y)
        if storage and storage in self.empty_storages:
            self.empty_storages.remove(storage)
        if storage:
            storage.is_empty = False

    def setStorageAvailable(self, coordinate: NetLogoCoordinate):
        storage = self.getStorageByCoordinate(coordinate.x, coordinate.y)
        if storage and storage not in self.empty_storages:
            self.empty_storages.append(storage)
        if storage:
            storage.is_empty = True

    def createStorage(self, x: int, y: int) -> Storage:
        storage = Storage(self.storage_counter, x, y)
        storage.is_empty = True
        self.storage_counter += 1
        self.storages.append(storage)
        self.coordinate_to_storages[(x, y)] = storage
        self.empty_storages.append(storage)
        return storage

    def addPodToStorage(self, pod: Pod, storage: Storage):
        self.pods_to_storage[pod] = storage
        storage.assigned_pod = pod
        if storage in self.empty_storages:
            self.empty_storages.remove(storage)
        storage.is_empty = False

    def reserveStorageForPod(self, pod: Pod, storage: Storage):
        if pod is None:
            raise ValueError("cannot reserve storage for missing pod")
        if storage is None:
            raise ValueError("cannot reserve missing storage")
        if storage not in self.storages:
            raise ValueError(f"storage {storage!r} is not managed by this warehouse")
        if not getattr(storage, "is_empty", False) or getattr(storage, "assigned_pod", None) is not None:
            raise ValueError(f"storage {storage!r} is not available for reservation")
        existing_storage = self.pods_to_storage.get(pod)
        if existing_storage is not None and existing_storage is not storage:
            raise ValueError(f"pod {pod!r} is already assigned to storage {existing_storage!r}")
        self.addPodToStorage(pod, storage)

    def releaseStorageReservation(self, pod: Pod, storage: Storage | None = None):
        target_storage = storage or self.pods_to_storage.get(pod)
        if target_storage is None:
            return False
        if self.pods_to_storage.get(pod) is target_storage:
            del self.pods_to_storage[pod]
        if getattr(target_storage, "assigned_pod", None) is pod:
            target_storage.removeStoragePod()
        elif getattr(target_storage, "assigned_pod", None) is None:
            target_storage.is_empty = True
        if target_storage not in self.empty_storages:
            self.empty_storages.append(target_storage)
        return True

    def getPodByNumber(self, pod_number: int) -> Optional[Pod]:
        return next((p for p in self.pods_to_storage if p.pod_number == pod_number), None)

    def getNearestEmptyStorage(self, station_coordinate: NetLogoCoordinate, robots_coordinate: List[List[int]]) -> Optional[Storage]:
        station_pos = np.array([[station_coordinate.x, station_coordinate.y]])
        best_storage = None
        best_score = -1

        candidate_data = []

        for storage in self.empty_storages:
            storage_pos = np.array([[storage.pos_x, storage.pos_y]])
            distance_to_station = manhattan_distances(storage_pos, station_pos)[0][0]
            distance_to_robot = self._distanceStorageToRobot(storage_pos[0], robots_coordinate)

            score = -distance_to_station - distance_to_robot  # Lower is better
            candidate_data.append((storage, score))

        if candidate_data:
            best_storage = sorted(candidate_data, key=lambda x: x[1])[0][0]

        return best_storage

    def _distanceStorageToRobot(self, storage_coord, robots_coordinate: List[List[int]]) -> float:
        if not robots_coordinate:
            return 1000.0  # Arbitrary high distance if no robot is present
        distances = manhattan_distances([storage_coord], robots_coordinate)
        return distances.min()
    

    def getNearestEmptyStorageToLocation(self, location_coordinate: NetLogoCoordinate, robot_coordinate: NetLogoCoordinate):
        location_pos = np.array([[location_coordinate.x, location_coordinate.y]])
        robot_coords_np = np.array([[robot_coordinate.x, robot_coordinate.y]])

        available_storages = [
            s for s in self.storages
            if s.is_empty and s.assigned_pod is None
        ]

        if not available_storages:
            return None

        storage_positions = np.array([[s.pos_x, s.pos_y] for s in available_storages])
        # print("[DEBUG] getNearestEmptyStorageToLocation")
        # print(f"[DEBUG] storage_position {storage_positions}")
        distances = manhattan_distances(storage_positions, location_pos).flatten()

        min_idx = distances.argmin()
        return available_storages[min_idx]
