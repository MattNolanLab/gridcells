'''Common definitions'''


class Pair2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Position2D(Pair2D):
    '''Specifies positional information with a constant time step.'''
    def __init__(self, x, y, dt):
        self.x = x
        self.y = y
        self.dt = dt


