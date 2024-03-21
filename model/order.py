class Order:
    has_to_take_pod = True

    def __init__(self, pod):
        self.designated_pod = pod
        self.coordinate = None
