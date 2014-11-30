'''Setup script for GridCells.'''
from setuptools import setup, Extension

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
                      swig_opts=default_swig_opts)
common_ext = DelayedExtension('gridcells.core._common',
                       ['src/common.cpp', 'src/common.i'],
                       swig_opts=default_swig_opts)

spikes_ext = DelayedExtension('gridcells.analysis._spikes',
                       ['src/spikes.cpp', 'src/spikes.i'],
                       swig_opts=default_swig_opts)

all_extensions = [
    field_ext,
    common_ext,
    spikes_ext
]

setup(
    name='gridcells',
    version='0.1.0',
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
