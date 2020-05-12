import numpy as np
import gzip
import xml.etree.ElementTree as ET
try:
    from gpaw.setup_data import SetupData, PAWXMLParser
except Exception :
    pass

class PPXmlGPAW(object):
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

class PPXml(object):
    def __init__(self, infile, symbol = 'Al', xctype = 'LDA', name='paw'):
        '''
        Ref : https://wiki.fysik.dtu.dk/gpaw/setups/pawxml.html
        '''
        if infile.endswith('.gz'):
            fd = gzip.GzipFile(infile)
        else:
            fd = infile
        source = fd
        self.r = None  # radial_grid
        self.nvt = None # pseudo_valence_density
        self.nct = None # pseudo_core_density
        self.vloc = None # zero_potential
        self.read_xml(source=source)

    def read_xml(self, source=None, world=None):
        tree = ET.iterparse(source,events=['start', 'end'])
        for event, elem in tree:
            if event == 'end':
                if elem.tag=='radial_grid':
                    self.r = self.radial_grid(elem.attrib)
                elif elem.tag=='pseudo_valence_density':
                    self.nvt = np.fromstring(elem.text, dtype=float, sep=" ")
                elif elem.tag == 'pseudo_core_density':
                    self.nct = np.fromstring(elem.text, dtype=float, sep=" ")
                elif elem.tag == 'zero_potential':
                    self.vloc = np.fromstring(elem.text, dtype=float, sep=" ")

    def radial_grid(self, dicts):
        istart = int(dicts['istart'])
        iend = int(dicts['iend'])
        x = np.arange(istart, iend + 1, dtype = 'float')
        eq = dicts['eq']
        if eq == 'r=d*i':
            d = float(dicts['d'])
            r = d * x
        elif eq == 'r=a*exp(d*i)':
            a = float(dicts['a'])
            d = int(dicts['d'])
            r = a * np.exp(d * x)
        elif eq == 'r=a*(exp(d*i)-1)':
            a = float(dicts['a'])
            d = float(dicts['d'])
            r = a * (np.exp(d * x) - 1)
        elif eq == 'r=a*i/(1-b*i)':
            a = float(dicts['a'])
            b = float(dicts['b'])
            r = a * x / (1 - b * x)
        elif eq == 'r=a*i/(n-i)':
            a = float(dicts['a'])
            n = int(dicts['n'])
            r = a * x / (n - x)
        elif eq == 'r=(i/n+a)^5/a-a^4':
            a = float(dicts['a'])
            n = int(dicts['n'])
            r = (x/n + a) ** 5/a - a ** 4
        else :
            raise AttributeError("!ERROR : not support eq= ", eq)
        return r

    def get_pseudo_valence_density(self):
        info = {}
        r = self.r
        v = self.nvt
        return r, v, info
