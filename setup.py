#!/usr/bin/env python3
from setuptools import setup, find_packages
import sys
import os
from edftpy import __version__, __author__, __contact__, __license__

def parse_requirements():
    requires = []
    with open('requirements.txt', 'r') as fr :
        for line in fr :
            pkg = line.strip()
            if pkg.startswith('git+'):
                pip_install_git(pkg)
            else:
                requires.append(pkg)
    return requires

def pip_install_git(link):
    os.system('pip install --upgrade {}'.format(link))
    return


assert sys.version_info >= (3, 6)

description = "eDFTpy"
long_description = """eDFTpy"""

scripts=['scripts/edftpy']

extras_require = {
        'libxc' : ['libxc @ git+https://gitlab.com/libxc/libxc.git'],
        'f90wrap' : ['f90wrap @ git+https://github.com/jameskermode/f90wrap.git'],
        }

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
