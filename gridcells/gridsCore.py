'''
.. currentmodule:: gridcells.gridsCore

The :mod::`~gridcells.gridsCore` module provides definitions for the
core/shared components for manipulating gridcells components.
'''

from _gridsCore import *

#from abc import ABCMeta, abstractmethod
#
#class Arena(object):
#    __metaclass__ = ABCMeta
#
#
#    @abstractmethod
#    def getDiscretisation(self):
#        raise NotImplementedError()
#
#    @abstractmethod
#    def getMask(self):
#        raise NotImplementedError()
#
#
#
#
#class RectangularArena(Arena):
#
#    def __init__(self, size, discretisation):
#        self._sz = size
#        self._q = discretisation
#
#    def getDiscretisation(self):
#        precision = arenaDiam/h
#
#        numX = self._sz.x / self._q.x + 1
#        numY = self._sz.y / self._q.y + 1
#        xedges = np.linspace(-self._sz.x/2., self._sz.x/2., numX)
#        yedges = np.linspace(-self._sz.y/2., self._sz.y/2., numY)
#        return xedges, yedges
#
#    def getMask(self):
#        return None
#
#
#
#class SquareArena(RectangularArena):
#    def __init__(self, size, discretisation):
#        tmpSz = Size2D(size, size)
#        super(SquareArena, self).__init__(tmpSz, discretisation)
#    
#
#class CircularArena(SquareArena):
#    def __init__(self, radius, discretisation):
#        super(CircularArena, self).__init__(radius*2., discretisation)
#        self.radius = radius
#
#
#    def getMask(self):
#        xedges, yedges = self.getDiscretisation()
#        X, Y = np.meshgrid(xedges, yedges)
#        return np.sqrt(X**2 + Y**2) > self.radius
