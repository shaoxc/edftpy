import os
import re
import xml.sax
import numpy as np
from gpaw.setup_data import SetupData, PAWXMLParser
import gzip

class PPXml(object):
    '''
    First, use the `gpaw` module to read xml-format pseudopotential
    '''
    def __init__(self, infile, symbol = 'Al', xctype = 'LDA', name='paw'):
        gpaw_pp = SetupData(symbol, xctype, name=name, readxml=False)
        self.pp = gpaw_pp
        if infile.endswith('.gz'):
            fd = gzip.open(infile)
        else:
            fd = open(infile, 'rb')
        source = fd.read()
        self.read_xml(source=source)

    def read_xml(self, source=None, world=None):
        PAWXMLParser(self.pp).parse(source=source, world=world)
        nj = len(self.pp.l_j)
        self.pp.e_kin_jj.shape = (nj, nj)

    def get_pseudo_valence_density(self):
        info = {}
        r = self.pp.rgd.r_g
        v = self.pp.nvt_g
        return r, v, info
