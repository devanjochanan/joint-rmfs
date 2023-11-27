from engine.object import Object
from engine.universe import Universe

class Pod(Object):
    def __init__(self):
        self.shape = 'full square'
        # self.shape = 'empty-space'
        self.object_type = 'pod'
        self.mass = 4
        super().__init__()
