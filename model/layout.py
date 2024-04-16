import csv


class Layout(object):
    def __init__(self):
        self.pod_batch_horizontal = 5
        self.pod_batch_vertical = 2
        self.pod_batch_horizontal_max = 5
        self.pod_batch_vertical_count = 0
        self.pod_batch_vertical_max = 10
        self.reserved_column_start = 10
        self.reserved_column_end = 10
        self.order_picker_total = 3
        self.horizontal_direction_switch = False
        self.vertical_direction_switch = False

    def generate(self):
        order_picker_positions = self.get_order_picker_indexes(self.get_order_picker_positions())
        print(order_picker_positions)
        with open('generated_pod.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.total_rows()):
                current_row = []
                self.vertical_direction_switch = False
                for col in range(self.total_cols()):
                    if col < 5:
                        current_row.append(self.get_value_for_order_picking(row, col, order_picker_positions))
                    else:
                        if self.reserved_column_start <= col < (self.total_cols() - self.reserved_column_end):
                            if row % (self.pod_batch_vertical + 1) == 0:
                                if (col - self.reserved_column_start) % (self.pod_batch_horizontal + 1) == 0:
                                    current_row.append(3)
                                    self.vertical_direction_switch = not self.vertical_direction_switch
                                else:
                                    current_row.append(4 if self.horizontal_direction_switch else 5)
                            else:
                                pod_index = (col - self.reserved_column_start - 1) % (self.pod_batch_horizontal + 1)
                                if pod_index < self.pod_batch_horizontal:
                                    current_row.append(1)
                                else:
                                    current_row.append(6 if self.vertical_direction_switch else 7)
                                    self.vertical_direction_switch = not self.vertical_direction_switch
                        else:
                            current_row.append(6 if self.vertical_direction_switch else 7)
                            self.vertical_direction_switch = not self.vertical_direction_switch

                writer.writerow(current_row)
                self.horizontal_direction_switch = not self.horizontal_direction_switch

    def total_rows(self):
        return (self.pod_batch_vertical_max * self.pod_batch_vertical) + self.pod_batch_vertical_max + 1

    def total_cols(self):
        # Total columns include reserved columns, pods, and spaces between pods
        total_pod_space = (
                                  self.pod_batch_horizontal * self.pod_batch_horizontal_max) + self.pod_batch_horizontal_max - 1
        return (self.reserved_column_start + 1) + total_pod_space + (self.reserved_column_end + 1)

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
        return start_index, start_index + self.pod_batch_vertical + 1

    @staticmethod
    def get_value_for_order_picking(row, col, ranges):
        for start, end in ranges:
            if row == start:
                # First row
                if col == 2:
                    return 16
                elif col == 3 or col == 4:
                    return 12
                else:
                    return 99
            elif row == end:
                # Last row
                if col == 2:
                    return 17
                elif col == 3 or col == 4:
                    return 13
                else:
                    return 99
            elif start < row < end:
                # Middle row
                if row == end - 1:
                    if col == 1:
                        return 11
                    if col == 2:
                        return 15
                    else:
                        return 99
                if col == 2:
                    return 14
                else:
                    return 99
        return 99
