#!/usr/bin/env python3
# import sys; sys.settrace()
import argparse
import time

from edftpy.interface import optimize_density_conf, conf2init, conf2output
from edftpy.config import read_conf
from edftpy.mpi import sprint

def get_conf():
    parser = argparse.ArgumentParser(description='Process task')
    parser.add_argument('confs', nargs = '*')
    parser.add_argument('-i', '--ini', '--input', dest='input', type=str, action='store',
            default='config.ini', help='Input file (default: config.ini)')
    parser.add_argument('--mpi', '--mpi4py', dest='mpi', action='store_true',
            default=False, help='Use mpi4py to be parallel')

    args = parser.parse_args()
    return args

def run_job(args):
    from edftpy.mpi import pmi
    if len(args.confs) == 0 :
        args.confs.append(args.input)
    for fname in args.confs:
        config = read_conf(fname)
        parallel = args.mpi or pmi.size > 0
        graphtopo = conf2init(config, parallel)
        sprint("Begin on : {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        sprint("#" * 80)

        optimizer = optimize_density_conf(config, graphtopo = graphtopo)

        graphtopo.timer.End("TOTAL")
        if graphtopo.rank == 0 :
            graphtopo.timer.output(config)
        sprint("-" * 80)
        #-----------------------------------------------------------------------
        conf2output(config, optimizer)
        #-----------------------------------------------------------------------
    sprint("#" * 80)
    sprint("Finished on : {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

def main():
    args = get_conf()
    run_job(args)
