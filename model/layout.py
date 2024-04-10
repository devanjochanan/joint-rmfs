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
        self.order_picker_total = 3

    def draw(self):
        order_picker_positions = self.get_order_picker_indexes(self.get_order_picker_positions())
        # print(order_picker_positions)
        with open('generated_pod.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = 0
            while self.pod_batch_vertical_count < self.pod_batch_vertical_max:
                if row % (self.pod_batch_vertical + 1) == 0:
                    # Horizontal straight line
                    current_row = [0] * self.total_cols()
                    if row != 0:
                        self.pod_batch_vertical_count += 1
                    
                    # Overwrite for intersection
                    for col in range(self.reserved_column_start - 1, len(current_row) - (self.reserved_column_end - 1), self.pod_batch_horizontal_max + 1):
                        current_row[col] = 3
                else:
                    current_row = []
                    self.pod_batch_horizontal_count = 0
                    for col in range(self.total_cols()):
                        # Write pods
                        if col >= self.reserved_column_start - 1:
                            pod_index = (col - self.reserved_column_start) % (self.pod_batch_horizontal + 1)
                            if pod_index < self.pod_batch_horizontal and self.pod_batch_horizontal_count <= self.pod_batch_horizontal_max:
                                current_row.append(1)
                            else:
                                self.pod_batch_horizontal_count += 1
                                current_row.append(0)
                        else:
                            current_row.append(0)

                # Overwrite for order picker
                if any(lower <= row <= upper for lower, upper in order_picker_positions):
                    for col in range(5):
                        current_row[col] = 2

                writer.writerow(current_row)
                row += 1

    def total_cols(self):
        # Total columns include reserved columns, pods, and spaces between pods
        total_pod_space = (self.pod_batch_horizontal * self.pod_batch_horizontal_max) + self.pod_batch_horizontal_max - 1
        return self.reserved_column_start + total_pod_space + self.reserved_column_end
    
    def determine_order_picker_limits(self):
        return int((self.pod_batch_vertical_max + 1) / 2)

    def get_order_picker_positions(self):
        total_numbers = self.determine_order_picker_limits()
        numbers = list(range(1, total_numbers + 1))
        selected_sequence = []
        middle_index = len(numbers) // 2

        for i in range(self.order_picker_total):
            if i % 2 == 0:
                offset = (i // 2)
            else:
                offset = -(i // 2) - 1
            
            selected_sequence.append(numbers[middle_index + offset])

        print(selected_sequence)
        print(total_numbers)
        return sorted(selected_sequence)
    
    def get_order_picker_indexes(self, order_picker_positions):
        result = []
        for i in order_picker_positions:
            result.append(self.get_order_picker_index(i))

        return result
    
    def get_order_picker_index(self, order_picker_position):
        start_index = (order_picker_position - 1) * (2 * self.pod_batch_vertical + 2)
        return (start_index, start_index + self.pod_batch_vertical + 1)