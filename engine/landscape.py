from .netlogo_coordinate import NetLogoCoordinate
from .coordinate import Coordinate
from .heading import Heading
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def astar(grid, start, end):
    def heuristic(node, end):
        return abs(node.x - end.x) + abs(node.y - end.y)

    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    open_set.append(start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_node.x + dx, current_node.y + dy

            if (
                0 <= x < len(grid)
                and 0 <= y < len(grid[0])
                and grid[x][y] == 0
                and (x, y) not in closed_set
            ):
                neighbor = Node(x, y, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = heuristic(neighbor, end_node)

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
    
    return None

class Landscape:
    dimension = 0
    _map = []

    def __init__(self, dimension):
        self.dimension = dimension
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append(0)
            self._map.append(one_row)
    
    def cloneMap(self):
        return self._map.copy()
    
    def printMap(self, to_print=None):
        if to_print is None:
            to_print = self._map
            
        idx = 0
        for i in to_print:
            print(str(idx) + " ", end="")
            print(i)
            idx += 1
    
    def setObject(self, coor: NetLogoCoordinate, obj):
        p = Coordinate.fromNetLogoCoordinate(self, coor)
        self._map[p.y][p.x] = obj
    
    def getRoute(self, _from: NetLogoCoordinate, _to: NetLogoCoordinate):
        to = Coordinate.fromNetLogoCoordinate(self, _to)
        origin = Coordinate.fromNetLogoCoordinate(self, _from)

        # build list based map for a star
        aStarMap = self.cloneMap()

        # set destination and origin
        aStarMap[to.y][to.x] = 0
        aStarMap[origin.y][origin.x] = 0

        start = (origin.y, origin.x)
        end = (to.y, to.x)

        path = astar(aStarMap, start, end)
        res = []
        real_res = []
        for p in path:
            q = Coordinate(p[1], p[0], self)
            res.append(q.toNetLogoCoordinate())
        real_res.append(res[0])

        curr_movement_type = None
        for i in range(len(res)-1):
            points, movement_type = self.splitTwoPoints(res[i], res[i+1])
            if curr_movement_type != movement_type and curr_movement_type is not None:
                direction = self.checkMovementDirection(res[i], res[i+1])
                real_res.append(Heading(direction))
                real_res.append(res[i])
            curr_movement_type = movement_type
            
            # for x in points:
            #     real_res.append(x)
            real_res.append(res[i+1])

        return real_res
    
    def checkMovementDirection(self, p1, p2):
        if p1.y == p2.y:
            # horizontal
            if p1.x < p2.x:
                return 90
            if p1.x > p2.x:
                return 270
            
        if p1.x == p2.x:
            # vertical
            if p1.y < p2.y:
                return 0
            if p1.y > p2.y:
                return 180
            
    def splitTwoPoints(self, p1, p2):
        x_difference = p2.x - p1.x
        y_difference = p2.y - p1.y

        # print(p1, p2, x_difference, y_difference)
        # raise Exception("tests")
        if p1.x == p2.x and p1.y == p2.y:
            return [p1], x_difference == 0
        
        addition_x = x_difference / 4
        addition_y = y_difference / 4
        res = [p1]
        now_x = p1.x
        now_y = p1.y

        for i in range(4):
            now_x = now_x + addition_x
            now_y = now_y + addition_y
            new_p = Coordinate(now_x, now_y)
            res.append(new_p)

        return res, x_difference == 0