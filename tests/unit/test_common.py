from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from gridcells.core import (Pair2D, Position2D, divisor_mod,
                            twisted_torus_distance)


@pytest.fixture(scope='module', params=[0, 1, 2, 3, 10] + list(range(20, 110, 10)))
def possize(request):
    return request.param

@pytest.fixture(scope='module')
def dt_fixture():
    return 20

@pytest.fixture(scope='module')
def copymaker():
    def pc(pos):
        return Position2D(np.copy(pos.x), np.copy(pos.y), pos.dt)
    return pc


@pytest.fixture(scope='module')
def position(possize, dt_fixture):
    x = np.arange(possize)
    y = np.arange(possize) + 10
    return Position2D(x, y, dt_fixture)


class TestPosition2D(object):

    def createPosition(self, nx, ny, dt):
        x = np.arange(nx)
        y = np.arange(ny) + 10
        return Position2D(x, y, dt)

    def test_attributes(self):
        p = self.createPosition(10, 10, 0.02)
        assert np.all(np.arange(10) == p.x)
        assert np.all(np.arange(10)+10 == p.y)
        assert .02 == p.dt


    def assert_should_equal(self, p1, p2):
        assert p1 == p2
        assert not(p1 != p2)

    def assert_should_not_equal(self, p1, p2):
        assert not(p1 == p2)
        assert p1 != p2

    def test_equals(self, position, copymaker):
        p1 = position
        p2 = copymaker(p1)
        self.assert_should_equal(p1, p2)

        if len(p1) != 0:
            p2.x[-1] = -100
            self.assert_should_not_equal(p1, p2)

        p2 = copymaker(p1)
        if len(p1) != 0:
            p2.y[-1] = -100
            self.assert_should_not_equal(p1, p2)

        p2 = copymaker(p1)
        if len(p1) != 0:
            p2.dt = -100
            self.assert_should_not_equal(p1, p2)


def test_divisor_mod():
    dividend_range = 11
    divisor_range = 11
    dx = .3
    for dividend in np.arange(-dividend_range, dividend_range, dx):
        for divisor in np.arange(-divisor_range, divisor_range, dx):
            np.testing.assert_allclose(
                    divisor_mod(dividend, divisor),
                    dividend % divisor,
                    rtol=0, atol=1e-10)

    assert(divisor_mod(0, 3) == 0 % 3)
    assert(divisor_mod(1, 3) != 2)
    assert(divisor_mod(1, 3) != 0)


##############################################################################

class TestTTDistance(object):
    def remapTwistedTorus(self, a, others, dim):
        ''' Calculate a distance between ``a`` and ``others`` on a twisted torus.

        Take ``a`` which is a 2D position and others, which is a vector of 2D
        positions and compute the distances between them based on the topology of
        the twisted torus.

        If you just want to remap a function of (X, Y), set a==[[0, 0]].

        Parameters
        ----------

        a : a Position2D instance
            Specifies the initial position. ``a.x`` and ``a.y`` must be convertible
            to floats
        others : Position2D instance
            Positions for which to compute the distance.
        dim : Position2D
            Dimensions of the torus. ``dim.x`` and ``dim.y`` must be convertible to
            floats.

        Returns
        -------
        An array of positions, always of the length of others
        '''
        a_x      = float(a.x)
        a_y      = float(a.y)
        others_x = np.asarray(others.x)
        others_y = np.asarray(others.y)
        x_dim    = float(dim.x)
        y_dim    = float(dim.y)

        # Remap the values modulo torus size.
        a_x = a_x % x_dim
        a_y = a_y % y_dim
        others_x = others_x % x_dim
        others_y = others_y % y_dim

        d = np.empty((7, others_x.size))

        d[0, :] = np.sqrt((a_x - others_x            )**2 + (a_y - others_y        )**2)
        d[1, :] = np.sqrt((a_x - others_x - x_dim    )**2 + (a_y - others_y        )**2)
        d[2, :] = np.sqrt((a_x - others_x + x_dim    )**2 + (a_y - others_y        )**2)
        d[3, :] = np.sqrt((a_x - others_x + 0.5*x_dim)**2 + (a_y - others_y - y_dim)**2)
        d[4, :] = np.sqrt((a_x - others_x - 0.5*x_dim)**2 + (a_y - others_y - y_dim)**2)
        d[5, :] = np.sqrt((a_x - others_x + 0.5*x_dim)**2 + (a_y - others_y + y_dim)**2)
        d[6, :] = np.sqrt((a_x - others_x - 0.5*x_dim)**2 + (a_y - others_y + y_dim)**2)

        return np.min(d, axis=0)

    def test_reference_imp(self):
        # Basic test
        NX, NY = 53, 53
        a = Pair2D(0, 0)
        dim = Pair2D(1., 1.)
        shift = .7
        for x_shift in [0, shift]:
            for y_shift in [0, shift]:
                for nx in np.arange(1, NX+1):
                    for ny in np.arange(1, NY+1):
                        X, Y = np.meshgrid(np.linspace(0, 1, nx),
                                           np.linspace(0, 1, ny))
                        X += x_shift
                        Y += y_shift
                        others = Pair2D(X.flatten(), Y.flatten())
                        np.testing.assert_allclose(
                            self.remapTwistedTorus(a, others, dim),
                            twisted_torus_distance(a, others, dim),
                            rtol=1e-10, atol=0)
