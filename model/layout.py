import csv


class Layout(object):
    def __init__(self):
        self.pod_batch_horizontal = 5
        self.pod_batch_vertical = 2
        self.pod_batch_horizontal_count = 0
        self.pod_batch_horizontal_max = 5
        self.pod_batch_vertical_count = 0
        self.pod_batch_vertical_max = 10
        self.reserved_column_start = 10
        self.reserved_column_end = 10

    def draw(self):
        with open('generated_pod.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = 0
            while self.pod_batch_vertical_count < self.pod_batch_vertical_max:
                if row % (self.pod_batch_vertical + 1) == 0:
                    # Fill the entire row with 0s, increment vertical batch count after the first row
                    current_row = [0] * self.total_cols()
                    if row != 0:
                        self.pod_batch_vertical_count += 1
                else:
                    current_row = []
                    self.pod_batch_horizontal_count = 0
                    for col in range(self.total_cols()):
                        # Starting from the reserved column, alternate between 1s and 0s based on horizontal batch logic
                        if col >= self.reserved_column_start - 1:
                            pod_index = (col - self.reserved_column_start) % (self.pod_batch_horizontal + 1)
                            if pod_index < self.pod_batch_horizontal and self.pod_batch_horizontal_count <= self. pod_batch_horizontal_max:
                                current_row.append(1)
                            else:
                                self.pod_batch_horizontal_count += 1
                                current_row.append(0)
                        else:
                            current_row.append(0)

                writer.writerow(current_row)
                row += 1

    def total_cols(self):
        # Total columns include reserved columns, pods, and spaces between pods
        total_pod_space = self.pod_batch_horizontal * self.pod_batch_horizontal_max + self.pod_batch_horizontal_max - 1
        return self.reserved_column_start + total_pod_space + self.reserved_column_end
