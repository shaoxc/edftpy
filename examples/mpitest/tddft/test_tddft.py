#!/usr/bin/env python3
import numpy as np
from edftpy.interface import conf2init
from edftpy.api.parse_config import config2optimizer
from edftpy.config import read_conf

fname = 'edftpy_2.ini'

def test_tddft_0():
    config = read_conf(fname)
    graphtopo = conf2init(config, parallel = True)
    optimizer = config2optimizer(config, graphtopo = graphtopo)

    assert len(optimizer.drivers) == 2
    graphtopo.assert_check(np.alltrue(graphtopo.graph.sub_shift[0]<0)) # Grid : SUB_HO_0 [-11 -14 -13] [48 56 50] [ 90 108  96]
    graphtopo.assert_check(np.alltrue(graphtopo.graph.sub_shift[1]>1)) # Grid : SUB_HO_1 [14 11 12] [48 56 50] [ 90 108  96]

    optimizer.optimize()
    graphtopo.assert_check(np.isclose(optimizer.dipole[0], 0.017, atol=1E-3))
    optimizer.stop_run(save = ['W', 'D'])

def test_tddft_scf_1():
    config = read_conf(fname)
    config['TD']['restart'] = 'scf'
    graphtopo = conf2init(config, parallel = True)
    optimizer = config2optimizer(config, graphtopo = graphtopo)

    optimizer.optimize()
    graphtopo.assert_check(np.isclose(optimizer.dipole[0], 0.017, atol=1E-3))

def test_tddft_restart_2():
    config = read_conf(fname)
    config['TD']['restart'] = 'restart'
    config['TD']['iter'] = 5
    graphtopo = conf2init(config, parallel = True)
    optimizer = config2optimizer(config, graphtopo = graphtopo)

    optimizer.optimize()
    graphtopo.assert_check(np.isclose(optimizer.dipole[0], 0.031, atol=1E-3))


if __name__ == "__main__":
    test_tddft_0()
    test_tddft_scf_1()
    test_tddft_restart_2()
