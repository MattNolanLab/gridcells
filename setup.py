from distutils.core import setup, Extension
import numpy

setup(
        name='gridcells',
        version='0.1.0',
        description='Package for grid cell analysis and simulation.',
        author='Lukas Solanka',
        author_email='lsolanka@gmail.com',
        packages=['gridcells'],
        ext_modules=[Extension('gridcells.analysis._fields',
                                    ['src/fields.cpp', 'src/fields.i'],
                                     swig_opts=['-builtin',
                                                '-O',
                                                '-ignoremissing',
                                                '-c++',
                                                '-Iexternal/armanpy/include',
                                                '-Isrc/include'])
                    ],
        include_dirs=['src/include', 'external/armanpy/include',
                      numpy.get_include()],
        install_requires=['numpy>=1.8.0',
                          'scipy>=0.13.3']
)
