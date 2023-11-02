from engine.object import Object
from engine.universe import Universe

class Station(Object):
    def __init__(self):
        self.shape = 'arrow'
        self.object_type = 'station'
        self.mass = 1
        super().__init__()
