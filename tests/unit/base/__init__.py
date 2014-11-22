'''Common testing code.'''
from __future__ import absolute_import, print_function, division

import pytest

__all__ = ['notimpl']

notimpl = pytest.mark.skipif(True, reason="Not implemented")
