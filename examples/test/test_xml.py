import unittest
import numpy as np
import os
from edftpy.io.pp_xml import PPXml

class Test(unittest.TestCase):
    def test_read_xml(self):
        data_path = os.environ.get('EDFTPY_DATA_PATH')
        if not data_path : data_path = 'DATA'
        infile = data_path + os.sep + 'Al.LDA.gz'
        pp = PPXml(infile)

        r, v, info = pp.get_pseudo_valence_density()

        self.assertTrue(np.isclose(r[1], 0.0008908685968819599))
        self.assertTrue(np.isclose(v[1], 0.07836203580181))


if __name__ == "__main__":
    unittest.main()
