#!/usr/bin/env python3
from setuptools import setup, find_packages
import re
import sys
import os
from edftpy import __version__, __author__, __contact__, __license__

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


description = "EDFTpy"
long_description = """EDFTpy"""
scripts=[]

setup(name='edftpy',
      description=description,
      long_description=long_description,
      url='https://gitlab.com/pavanello-research-group/edftpy',
      version=__version__,
      author=__author__,
      author_email=__contact__,
      license=__license__,
      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      packages=find_packages(),
      scripts=scripts,
      include_package_data=True,
      install_requires=['numpy>=1.8.0', 'scipy>=0.10.0'])
