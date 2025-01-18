#!/usr/bin/env python3
from setuptools import setup, find_packages
import subprocess
import sys
res = subprocess.run([sys.executable, 'tools/gitversion.py', '--write', 'edftpy/version.py'], capture_output=True, text=True)
version = res.stdout.split()[-1].strip()

setup(name='edftpy',
      version = version,
      packages=find_packages(),
      include_package_data=True
      )
