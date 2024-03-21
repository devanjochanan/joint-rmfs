class Universe:
    _tick = 0
    id = 0
    tick_to_second = 0.5
    _objects = []
    landscape = None
    graph_pod = None

    def addObject(self, object):
        object.id = len(self._objects)
        object.setUniverse(self)
        self._objects.append(object)

    def tick(self):
        for o in self._objects:
            o.move()

    def moveableObjects(self):
        return self._objects
    
    def generateResult(self):
        result = []
        for o in self.moveableObjects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })

        return result
