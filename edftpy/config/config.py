import numpy as np
import configparser
import copy
import os
from dftpy.constants import ENERGY_CONV, LEN_CONV
from dftpy.config.config import ConfigEntry, readJSON, DefaultOptionFromEntries
from dftpy.config.config import default_json as dftpy_default_json

from dftpy.config.config import PrintConf as print_conf

def merge_dftpy_entries(entries):

    def recursive_merge(dftpy_entries, entries):
        if not isinstance(entries, dict):
            return entries
        lsame = False
        for k, v in entries.items() :
            if isinstance(v, dict):
                entries[k] = recursive_merge(dftpy_entries, v)
            if k == 'dftpy' :
                vn = dftpy_entries[v.default.upper()]
                lsame = True
                del entries[k]
                break
        if lsame :
            vnm = vn.copy()
            for key in vn.keys():
                if key in entries :
                    del vnm[key]
            entries.update(vnm)
        return entries

    dftpy_entries = dftpy_default_json()
    for k, v in entries.items() :
        entries[k] = recursive_merge(dftpy_entries, v)

def entries2conf(entries):
    conf = {}
    merge_dftpy_entries(entries)
    for k, v in entries.items() :
        conf[k] = v
    return DefaultOptionFromEntries(conf)

def default_option():
    import os
    json_file = os.path.join(os.path.dirname(__file__), 'configentries.json')
    configentries = readJSON(json_file)
    return entries2conf(configentries)

def conf_special_format(conf):
    for section in conf :
        if 'grid' in conf[section] :
            if conf[section]['grid']['spacing'] :
                conf[section]['grid']['ecut'] = (
                    np.pi ** 2
                    / (2 * conf[section]["grid"]["spacing"] ** 2)
                    * ENERGY_CONV["Hartree"]["eV"]
                    / LEN_CONV["Angstrom"]["Bohr"] ** 2
                )
            elif conf[section]['grid']['ecut'] :
                conf[section]['grid']['spacing'] = (
                    np.sqrt(np.pi ** 2 / conf[section]['grid']['ecut'] * 0.5 / ENERGY_CONV["eV"]["Hartree"])
                    * LEN_CONV["Bohr"]["Angstrom"]
                )
        if 'kedf' in conf[section] :
            conf[section]['kedf']['name'] = conf[section]['kedf']['kedf']

    return conf

def option_format(config, mapfunc = None, final = True):
    conf = {}
    if 'CONFDICT' in config :
        mapfunc = config['CONFDICT']
        #-----------------------------------------------------------------------
        subkeys = [key for key in config.keys() if key.startswith('SUB')]
        for key in subkeys:
            mapfunc[key] = copy.deepcopy(mapfunc['SUB'])
        del config['SUB']
        #-----------------------------------------------------------------------
    for key, value in config.items() :
        if key == 'CONFDICT':
            continue
        elif value :
            if isinstance(mapfunc[key], dict):
                conf[key] = option_format(value, mapfunc[key], False)
            else :
                conf[key] = mapfunc[key].format(str(value))
        else :
            conf[key] = value

    if final :
        conf = conf_special_format(conf)
    return conf

def dict_format(config):
    conf = default_option()
    #-----------------------------------------------------------------------
    subkeys = [key for key in config if key.startswith('SUB')]
    for key in subkeys:
        conf[key] = copy.deepcopy(conf['SUB'])
    #-----------------------------------------------------------------------
    for section, dicts in config.items() :
        for key, value in dicts.items():
            keys = key.split('-')
            dicts = conf[section]
            for k2 in keys[:-1]:
                tmp = dicts[k2]
                if tmp is not None :
                    dicts = tmp
            dicts[keys[-1]] = value
    conf = option_format(conf)
    return conf

def read_conf(infile):
    config = configparser.ConfigParser()

    if isinstance(infile, dict):
        conf_read = infile
    elif isinstance(infile, str):
        if os.path.isfile(infile):
            config.read(infile)
        elif '=' in infile :
            config.read_string(infile)
        else :
            raise AttributeError("!!!ERROR : %s not exist", infile)
        conf_read = {section : dict(config.items(section)) for section in config.sections()}

    conf = dict_format(conf_read)
    return conf

def read_conf_full(infile):
    config = configparser.ConfigParser()
    config.read(infile)

    conf = default_option()
    #-----------------------------------------------------------------------
    subkeys = [key for key in config.sections() if key.startswith('SUB')]
    for key in subkeys:
        conf[key] = copy.deepcopy(conf['SUB'])
    #-----------------------------------------------------------------------
    for section in config.sections():
        for key in config.options(section):
            value = config.get(section, key)
            keys = key.split('-')
            dicts = conf[section]
            for k2 in keys[:-1]:
                tmp = dicts[k2]
                if tmp is not None :
                    dicts = tmp
            dicts[keys[-1]] = value
    conf = option_format(conf)
    return conf

def entries2conf_2(entries):
    conf = {}
    merge_dftpy_entries(entries)
    for k, v in entries.items() :
        conf[k] = merge_entries(v)
    return DefaultOptionFromEntries(conf)

def option_format_2(config):
    conf = {}
    for section in config :
        if section == 'CONFDICT':
            continue
        conf[section] = {}
        for key, value in config[section].items() :
            if value and isinstance(value, ConfigEntry):
                conf[section][key] = config['CONFDICT'][section][key].format(str(value))
            else :
                conf[section][key] = value

    conf = conf_special_format(conf)
    return conf

def conf_special_format_2(conf):
    for section in conf :
        if 'grid-spacing' in conf[section] :
            if conf[section]['grid-spacing'] :
                conf[section]['grid-ecut'] = (
                    np.pi ** 2
                    / (2 * conf["GRID"]["spacing"] ** 2)
                    * ENERGY_CONV["Hartree"]["eV"]
                    / LEN_CONV["Angstrom"]["Bohr"] ** 2
                )
            else:
                conf[section]['grid-spacing'] = (
                    np.sqrt(np.pi ** 2 / conf[section]['grid-ecut'] * 0.5 / ENERGY_CONV["eV"]["Hartree"])
                    * LEN_CONV["Bohr"]["Angstrom"]
                )

    return conf

def read_conf_2(infile):
    config = configparser.ConfigParser()
    config.read(infile)

    conf = default_option()
    for section in config.sections():
        for key in config.options(section):
            if section != 'PP' and key not in conf[section]:
                print('!WARN : "%s.%s" not in the dictionary' % (section, key))
            else:
                conf[section][key] = config.get(section, key)
    conf = option_format_2(conf)
    return conf

def merge_entries(entries, parent = None):
    merged = {}
    for k, v in entries.items() :
        if parent :
            k2 = parent + '-' + k
        else :
            k2 = k
        if isinstance(v, dict):
            merged.update(merge_entries(v, k))
        else :
            merged[k2] = v
    return merged

