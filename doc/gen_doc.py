#!/usr/bin/env python3

from edftpy.config.config import default_option

header = r"""
.. _config:

===========================
Run eDFTpy with input files
===========================

eDFTpy is a set of python modules. However,you can run it using the `edftpy` executable. Here's a quick guide to the available keywords.

.. graphviz::

    digraph config {
           a0 [shape=box3d, label="JOB", href="../tutorials/config.html#job", target="_top"];
           a1 [shape=box3d, label="PATH", href="../tutorials/config.html#path", target="_top"];
           a6 [shape=box3d, label="GSYSTEM", href="../tutorials/config.html#gsystem", target="_top"];
           a2 [shape=box3d, label="MATH", href="../tutorials/config.html#math", target="_top"];
           a3 [shape=box3d, label="PP", href="../tutorials/config.html#pp", target="_top"];
           a5 [shape=box3d, label="OPT", href="../tutorials/config.html#opt", target="_top"];
           a4 [shape=box3d, label="OUTPUT", href="../tutorials/config.html#output", target="_top"];
           a7 [shape=box3d, label="SUB", href="../tutorials/config.html#sub", target="_top"];
           b601 [shape=rectangle, label="cell", href="../tutorials/config.html#cell", target="_top"];
           b602 [shape=rectangle, label="grid", href="../tutorials/config.html#grid", target="_top"];
           b603 [shape=rectangle, label="density", href="../tutorials/config.html#density", target="_top"];
           b604 [shape=rectangle, label="exc", href="../tutorials/config.html#exc", target="_top"];
           b605 [shape=rectangle, label="kedf", href="../tutorials/config.html#kedf", target="_top"];
           b606 [shape=rectangle, label="decompose", href="../tutorials/config.html#decompose", target="_top"];
           b707 [shape=rectangle, label="mix", href="../tutorials/config.html#mix", target="_top"];
           b708 [shape=rectangle, label="opt", href="../tutorials/config.html#id2", target="_top"];
           b709 [shape=rectangle, label="kpoints", href="../tutorials/config.html#kpoints", target="_top"];

           a6->b601[color=blue];
           a6->b602[color=blue];
           a6->b603[color=blue];
           a6->b604[color=blue];
           a6->b605[color=blue];
           a6->b606[color=blue];
           a7->b601;
           a7->b602;
           a7->b603;
           a7->b604;
           a7->b605;
           a7->b606;
           a7->b707;
           a7->b708;
           a7->b709;
        }

.. warning::
    `PP`_ is a mandatory input (i.e., no default is avaliable for it).

.. note::
    Defaults work well for most arguments.

    When *Options* is empty, it can accept any value.

.. _pylibxc: https://tddft.org/programs/libxc/
.. _dftd4: https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dftd4

"""

def gen_list_table(dicts, parent = None, top = False, add = False, ncol = 4):
    fstr = '\n'
    if parent :
        keys = []
        for k, v in dicts.items():
            if k == 'comment' : continue
            if hasattr(v, 'level') and v.level == 'devel' : continue
            if isinstance(v, dict):
                for k2, v2 in v.items() :
                    if k2 == 'comment' : continue
                    if hasattr(v2, 'level') and v2.level == 'devel' : continue
                    keys.append(k + '-' + k2)
            else :
                keys.append(k)
    else :
        keys = list(dicts.keys())

    try:
        keys.remove('comment')
    except :
        pass

    if len(keys) > 0 :
        keys = sorted(keys)
        fstr += ".. list-table::\n\n"
        for i, key in enumerate(keys):
            if i%ncol == 0 :
                shift = '\t\t*'
            else :
                shift = '\t\t '

            if parent :
                if '-' in key :
                    fstr += "{0} - :ref:`{1}<{1}>`\n".format(shift, key)
                else :
                    if top :
                        if add :
                            fstr += "{0} - :ref:`{2}--{1}<{2}--{1}>`\n".format(shift, key, parent)
                        else :
                            fstr += "{0} - :ref:`{1}<{2}--{1}>`\n".format(shift, key, parent)
                    else :
                        if add :
                            fstr += "{0} - :ref:`{2}-{1}<{2}-{1}>`\n".format(shift, key, parent)
                        else :
                            fstr += "{0} - :ref:`{1}<{2}-{1}>`\n".format(shift, key, parent)
            else :
                fstr += "{0} - `{1}`_\n".format(shift, key)
        if len(keys) > ncol :
            for j in range(ncol - i%ncol - 1):
                fstr += '\t\t  -\n'
    return fstr + '\n'

def gen_config_sub(item):
    if item.comment:
        lines = str(item.comment)
        lines = lines.replace('\\\\n', '\n')
        lines = lines.replace('\\\\t', '\t')
        # fstr += "\t{0}\n".format(item.comment)
        fstr = "\t{0}\n".format(lines)
    else :
        fstr = ""
    fstr += '\n'
    fstr += "\t- *Options* : {0}\n\n".format(item.options)
    fstr += "\t- *Default* : {0}\n\n".format(item.default)
    if item.unit :
        fstr += "\t- *Unit* : {0}\n\n".format(item.unit)
    if item.example :
        fstr += "\t- *e.g.* : \n\n\t\t\t{0}\n".format(item.example)
    if item.note:
        fstr += ".. note::\n {0}\n".format(item.note)
    if item.warning:
        fstr += ".. warning::\n {0}\n".format(item.warning)
    return fstr

def gen_config_rst():
    configentries = default_option()['CONFDICT']
    with open('./source/tutorials/config.rst', 'w') as f:
        f.write(header)
        gsystem = configentries['GSYSTEM']
        for section in configentries:
            #-----------------------------------------------------------------------
            fstr = "\n.. _{0}:\n\n".format(section)
            fstr += '\n{0}\n'.format(section)
            fstr += '-'*20 + '\n\n'.format(section)
            item = None

            if not isinstance(configentries[section], dict):
                item = configentries[section]
                item.default = item.comment
            elif 'comment' in configentries[section] :
                item = configentries[section]['comment']

            if item :
                lines = str(item.default)
                lines = lines.replace('\\\\n', '\n')
                lines = lines.replace('\\\\t', '\t')
                # fstr += "\t{0}\n\n".format(item.default)
                fstr += "\t{0}\n\n".format(lines)
                if item.example :
                    fstr += "\t*e.g.* : \n\n\t\t{0}\n".format(item.example)
                if item.note:
                    fstr += ".. note::\n {0}\n".format(item.note)
                if item.warning:
                    fstr += ".. warning::\n {0}\n".format(item.warning)
            f.write(fstr)
            if not isinstance(configentries[section], dict): continue
            fstr = gen_list_table(configentries[section], section, top = True)
            f.write(fstr)
            if section == 'GSYSTEM' : continue
            #-----------------------------------------------------------------------
            for key, item in configentries[section].items() :
                if key == 'comment' : continue
                if isinstance(item, dict):
                    fstr = '\n{0}\n'.format(key)
                    fstr += '^'*20 + '\n\n'.format(key)
                    if 'comment' in item :
                        item2 = item['comment']
                        lines = str(item2.default)
                        lines = lines.replace('\\\\n', '\n')
                        lines = lines.replace('\\\\t', '\t')
                        fstr += "\t{0}\n\n".format(lines)

                    #add GSYSTEM
                    if section== 'SUB' and key in gsystem :
                        for k2, v2 in gsystem[key].items() :
                            if k2 not in item : item[k2] = v2

                    fstr += gen_list_table(item, key, add = True)
                    f.write(fstr)

                    for key2, item2 in item.items():
                        if key2 == 'comment' : continue
                        if item2.level == 'devel' : continue
                        parent = key.lower()
                        fstr = "\n.. _{0}-{1}:\n\n".format(parent, key2)
                        fstr += "**{0}**\n".format(key + '-' + key2)
                        fstr += gen_config_sub(item2)
                        f.write(fstr)

                else :
                    if item.level == 'devel' : continue
                    fstr = "\n.. _{0}--{1}:\n\n".format(section, key)
                    fstr += "**{0}**\n".format(key)
                    fstr += gen_config_sub(item)
                    f.write(fstr)


if __name__ == "__main__":
    gen_config_rst()
