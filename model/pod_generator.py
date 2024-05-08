import csv
import random

from model.pod_manager import PodManager


class PodGenerator:
    def __init__(self, pod_manager: PodManager):
        self.total_skus = 1000
        self.pod_manager = pod_manager

    def generate(self):
        skus = list(range(self.total_skus))  # List of all SKUs
        random.shuffle(skus)  # Shuffle the list of SKUs for random distribution

        data_to_save = []  # Initialize an empty list to store data for the CSV file

        # Assign 5 SKUs to each pod
        sku_index = 0
        for pod in self.pod_manager.pods:
            for _ in range(5):
                if sku_index >= self.total_skus:
                    sku_index = 0  # Reset index if you reach the end of the SKU list
                    # Collecting data for each SKU added to a pod
                data_to_save.append({
                    "pod_id": pod.pod_id,
                    "sku": skus[sku_index],
                    "limit_qty": 999,
                    "current_qty": 999,
                    "threshold": 5
                })
                sku_index += 1

        # Save the collected data to a CSV file
        self.save_to_csv(data_to_save)

    @staticmethod
    def save_to_csv(data):
        with open('pod_sku.csv', 'w', newline='') as file:
            fieldnames = ['pod_id', 'sku', 'limit_qty', 'current_qty', 'threshold']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for item in data:
                writer.writerow(item)

