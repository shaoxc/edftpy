#!/usr/bin/env python3
import numpy as np
from edftpy.interface import conf2init
from edftpy.api.parse_config import config2optimizer
from edftpy.config import read_conf

def test_scf():
    fname = 'edftpy_2.ini'
    config = read_conf(fname)
    graphtopo = conf2init(config, parallel = True)
    optimizer = config2optimizer(config, graphtopo = graphtopo)

    assert len(optimizer.drivers) == 2
    graphtopo.assert_check(np.alltrue(graphtopo.graph.sub_shift[0]<0)) # Grid : SUB_HO_0 [-11 -14 -13] [48 56 50] [ 90 108  96]
    graphtopo.assert_check(np.alltrue(graphtopo.graph.sub_shift[1]>1)) # Grid : SUB_HO_1 [14 11 12] [48 56 50] [ 90 108  96]

    optimizer.optimize()

    graphtopo.assert_check(np.isclose(optimizer.energy, -34.666, atol = 1E-3))
    graphtopo.assert_check(np.isclose(optimizer.energy_all['SUB_0'], -0.249, atol = 1E-3))
    graphtopo.assert_check(np.isclose(optimizer.energy_all['SUB_1'], -0.249, atol = 1E-3))


if __name__ == "__main__":
    test_scf()
