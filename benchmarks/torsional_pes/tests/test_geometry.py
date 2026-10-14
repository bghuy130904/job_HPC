"""Geometric invariants: actual torsion, rigid framework, O-H bond length."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'methods/shared'))
from scan import read_xyz, rotate_oh, dihedral
import json

class GeometryTests(unittest.TestCase):
    def test_rigid_scan(self):
        spec = json.loads((ROOT/'inputs/scan.json').read_text())
        for molecule in spec['molecules'].values():
            symbols, base = read_xyz(ROOT/'inputs'/molecule['xyz'])
            base = np.asarray(base)
            indices = molecule['dihedral']
            o, h = indices[-2:]
            for angle in [0,10,60,90,120,180]:
                coords = rotate_oh(base,indices,angle)
                fixed = [i for i in range(len(symbols)) if i != h]
                np.testing.assert_array_equal(coords[fixed],base[fixed])
                self.assertAlmostEqual(np.linalg.norm(coords[h]-coords[o]),
                                       np.linalg.norm(base[h]-base[o]),places=12)
                self.assertAlmostEqual((dihedral(coords,indices)-angle+180)%360-180,0,places=8)

if __name__ == '__main__':
    unittest.main()
