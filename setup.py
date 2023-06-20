#!/usr/bin/env python3
from setuptools import setup, find_packages
import sys
import os
import re

def parse_requirements():
    requires = []
    with open('requirements.txt', 'r') as fr :
        for line in fr :
            pkg = line.strip()
            requires.append(pkg)
    return requires


with open('edftpy/__init__.py') as fd :
    lines = fd.read()
    __version__ = re.search('__version__ = "(.*)"', lines).group(1)
    __author__ = re.search('__author__ = "(.*)"', lines).group(1)
    __contact__ = re.search('__contact__ = "(.*)"', lines).group(1)
    __license__ = re.search('__license__ = "(.*)"', lines).group(1)

assert sys.version_info >= (3, 6)
description = "eDFTpy"
long_description = """eDFTpy"""

scripts=['scripts/edftpy']

extras_require = {
        'libxc' : ['pylibxc @ git+https://gitlab.com/libxc/libxc.git'],
        'all' : [
            'pyfftw',
            ],
        }

release = 0
if release :
    VERSION = {'version' : __version__}
else :
    VERSION = {
            'use_scm_version': {'version_scheme': 'post-release'},
            'setup_requires': [
                'setuptools_scm',
                'importlib-metadata>=0.12;python_version<"3.8"'],
            }

setup(name='edftpy',
      description=description,
      long_description=long_description,
      url='https://gitlab.com/pavanello-research-group/edftpy',
      author=__author__,
      author_email=__contact__,
      license=__license__,
      **VERSION,
      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      packages=find_packages(),
      scripts=scripts,
      include_package_data=True,
      extras_require = extras_require,
      install_requires= parse_requirements())
