from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='gridcells',
      version='0.1',
      description='Grid cells package',
      author='Lukas Solanka',
      author_email='lsolanka@gmail.com',
      packages=['gridcells'],
      ext_modules = cythonize("gridcells/fields.pyx"),
      include_dirs = [np.get_include()],
      install_requires=['numpy>=1.8.0',
                        'scipy>=0.13.3',
                        'Cython >= 0.20.1'])
