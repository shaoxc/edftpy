#!/usr/bin/env python3
from edftpy.cui import run

commands = {
        '--run' : 'edftpy.cui.run',
        '--convert' : 'edftpy.cui.convert',
        }

def get_args():
    parser = run.get_parse()
    #-----------------------------------------------------------------------
    parser.description = 'eDFTpy tasks :\n' + '\n\t'.join(commands.keys())
    parser.add_argument('--convert', dest='convert', action='store_true',
            default=False, help='Convert files formats')
    #-----------------------------------------------------------------------
    args = parser.parse_args()
    return args


def main():
    import sys
    from importlib import import_module
    for job in commands :
        if job in sys.argv :
            module = import_module(commands[job])
            command = getattr(module, 'main')
            return command()
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        run.main()
    else :
        get_args()
