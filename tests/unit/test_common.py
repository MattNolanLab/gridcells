from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from gridcells.core import Position2D


@pytest.fixture(scope='module', params=[0, 1, 2, 3, 10] + range(20, 110, 10))
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

