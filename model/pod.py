from engine.object import Object


class Pod(Object):
    def __init__(self):
        self.shape = 'full square'
        self.object_type = 'pod'
        self.mass = 4
        self.coordinate = None
        super().__init__()
