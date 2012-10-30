import sys, unittest
import warnings

import sys
sys.path.append( "./build/bin" )

import numpy as N
from pyml import *

c = MatTestComplexDouble( 2 )
j = N.sqrt( N.complex(-1,0) )

c.set_m( [ [1+0.1*j,2,3], [4,5+0.5*j,6] ] )
print c.get_m()

m = N.array( [ [1+j*0.1], [2+j*0.2] , [3+j*3] ], order='F' )
c.set_m( m )
print c.get_m()
