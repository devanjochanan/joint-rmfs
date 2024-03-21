from engine.object import Object


class Station(Object):
    def __init__(self):
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        super().__init__()
