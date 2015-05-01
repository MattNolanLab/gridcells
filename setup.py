'''Setup script for GridCells.'''
from __future__ import absolute_import, print_function, division

from os.path import join
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

all_packages = [
    'gridcells',
    'gridcells.analysis',
    'gridcells.core',
    'gridcells.plotting',
]

default_swig_opts = [
    '-builtin',
    '-Wall',
    '-O',
    '-ignoremissing',
    '-c++',
    '-Iexternal/armanpy/include',
    '-Isrc/include'
]


ARMANPY_INCLUDE_DIR = 'external/armanpy/include'
ARMANPY_DEPS = [
    join(ARMANPY_INCLUDE_DIR, 'armanpy.hpp'),
    join(ARMANPY_INCLUDE_DIR, 'armanpy.i'),
    join(ARMANPY_INCLUDE_DIR, 'armanpy_1d.i'),
    join(ARMANPY_INCLUDE_DIR, 'armanpy_2d.i'),
    join(ARMANPY_INCLUDE_DIR, 'armanpy_3d.i'),
    join(ARMANPY_INCLUDE_DIR, 'numpy.i'),
]


class DelayedExtension(Extension, object):
    """
    A distutils Extension subclass where some of its members
    may have delayed computation until reaching the build phase.

    This is so we can, for example, get the Numpy include dirs
    after pip has installed Numpy for us if it wasn't already
    on the system.

    This class has been adapted from the matplotlib package.
    """
    def __init__(self, *args, **kwargs):
        super(DelayedExtension, self).__init__(*args, **kwargs)
        self._hooks = {"include_dirs": self.get_include_dirs}

    @staticmethod
    def get_include_dirs():
        import numpy
        return [
            'src/include',
            'external/armanpy/include',
            'external/armadillo/include',
            numpy.get_include()
        ]

    class DelayedMember(property):
        def __init__(self, name):
            self._name = name

        def __get__(self, obj, objtype=None):
            result = getattr(obj, '_' + self._name, [])

            if self._name in obj._hooks:
                result = obj._hooks[self._name]() + result

            return result

        def __set__(self, obj, value):
            setattr(obj, '_' + self._name, value)

    include_dirs = DelayedMember('include_dirs')


field_ext = DelayedExtension('gridcells.analysis._fields',
                             ['src/fields.cpp', 'src/fields.i'],
                             depends=['src/fields.cpp',
                                      'src/include/fields.hpp',
                                      'src/fields.i'] + ARMANPY_DEPS,
                             swig_opts=default_swig_opts)

common_ext = DelayedExtension('gridcells.core._common',
                              ['src/common.cpp', 'src/common.i'],
                              depends=['src/common.cpp',
                                       'src/include/common.hpp',
                                       'src/common.i'] + ARMANPY_DEPS,
                              swig_opts=default_swig_opts)

spikes_ext = DelayedExtension('gridcells.analysis._spikes',
                              ['src/spikes.cpp', 'src/spikes.i'],
                              depends=['src/spikes.cpp',
                                       'src/include/spikes.hpp',
                                       'src/spikes.i'] + ARMANPY_DEPS,
                              swig_opts=default_swig_opts)

signal_ext = DelayedExtension('gridcells.analysis._signal',
                              ['src/signal.cpp', 'src/signal.i'],
                              depends=['src/signal.cpp',
                                       'src/include/signal.hpp',
                                       'src/signal.i'] + ARMANPY_DEPS,
                              swig_opts=default_swig_opts)

all_extensions = [
    field_ext,
    common_ext,
    spikes_ext,
    signal_ext
]

setup(
    name='gridcells',
    version='0.1.2',
    description='Package for grid cell analysis and simulation.',
    author='Lukas Solanka',
    author_email='lsolanka@gmail.com',
    url='https://github.com/lsolanka/gridcells',
    license='GPL',
    packages=all_packages,
    ext_modules=all_extensions,
    install_requires=['numpy>=1.8.0',
                      'scipy>=0.13.3']
)
