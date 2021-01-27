#!/usr/bin/env python3
from setuptools import setup, find_packages
import re
import sys
import os
from edftpy import __version__, __author__, __contact__, __license__


description = "eDFTpy"
long_description = """eDFTpy"""
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
      install_requires=['numpy>=1.11.0', 'scipy>=0.18.0', 'ase'])
