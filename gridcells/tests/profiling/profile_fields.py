# cython: profile=True
'''Profiling of the grid field analysis module.'''
import cProfile
import numpy as np

from gridcells import fields

N = 40
field = np.random.rand(40, 40)

dataDir = "data"
spikeTimes = np.loadtxt("%s/spikeTimes.txt" % dataDir)
pos_x      = np.loadtxt("%s/pos_x.txt" % dataDir)
pos_y      = np.loadtxt("%s/pos_y.txt" % dataDir)
dt         = 20
arenaDiam  = 180.0
h          = 3

cProfile.run('fields.SNSpatialRate2D(spikeTimes, pos_x, pos_y, dt, arenaDiam, h)',
        sort='tottime')
