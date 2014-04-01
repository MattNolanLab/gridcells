'''Profiling of the grid field analysis module.'''
import cProfile
import numpy as np

from gridcells import gridsCore, arena, fields

dataDir = "data"
spikeTimes = np.loadtxt("%s/spikeTimes.txt" % dataDir)
pos_x      = np.loadtxt("%s/pos_x.txt" % dataDir)
pos_y      = np.loadtxt("%s/pos_y.txt" % dataDir)
dt         = 20
arenaDiam  = 180.0
h          = 3

ar = arena.CircularArena(arenaDiam / 2., gridsCore.Size2D(h, h))
positions = gridsCore.Position2D(pos_x, pos_y, dt)
cProfile.run('fields.spatialRateMap(spikeTimes, positions, ar, h)',
        sort='tottime')
