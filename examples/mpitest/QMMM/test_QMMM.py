#!/usr/bin/env python3
import numpy as np
from edftpy.interface import conf2init
from edftpy.api.parse_config import config2optimizer
from edftpy.config import read_conf

def test_QMMM():
    fname = 'qmmm.in'
    config = read_conf(fname)
    graphtopo = conf2init(config, parallel = True)
    optimizer = config2optimizer(config, graphtopo = graphtopo)
    assert len(optimizer.drivers) == 2
    optimizer.optimize()
    graphtopo.assert_check(np.isclose(optimizer.energy, -17.34156, atol = 1E-3))


if __name__ == "__main__":
    test_QMMM()
