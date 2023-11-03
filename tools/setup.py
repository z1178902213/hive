from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(['robort.py', 'find_contour.py', 'common.py', 'find_worm.py', 'yolo_process.py']))