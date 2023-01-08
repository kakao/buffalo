# -*- coding: utf-8 -*-
import os
import math
import unittest
import tempfile

from buffalo import aux
from buffalo import MatrixMarket, MatrixMarketOptions


class TestPrepro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''%%MatrixMarket matrix coordinate integer general\n%\n%\n5 3 5\n1 1 1\n2 1 3\n3 3 1\n4 2 1\n5 2 2''')
            cls.mm_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''lucas\ngony\njason\nlomego\nhan''')
            cls.uid_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''apple\nmango\nbanana''')
            cls.iid_path = f.name
        cls.temp_files = []

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.mm_path)
        os.remove(cls.uid_path)
        os.remove(cls.iid_path)
        for path in cls.temp_files:
            os.remove(path)

    def test0_onebased(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = self.uid_path
        opt.input.iid = self.iid_path
        opt.data.value_prepro = aux.Option({'name': 'OneBased'})
        mm = MatrixMarket(opt)
        mm.create()
        self.temp_files.append(opt.data.path)
        self.assertTrue(True)
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['vali', 'idmap', 'rowwise', 'colwise']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 5)
        self.assertEqual(header['num_users'], 5)
        self.assertEqual(header['num_items'], 3)

        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 5)
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 0, 2, 1, 1])
        self.assertEqual([int(vv) for _, _, vv in data], [1, 1, 1, 1, 1])
        self.assertEqual(data[2], (2, 2, 1.0))

    def test1_minmax(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = self.uid_path
        opt.input.iid = self.iid_path
        opt.data.value_prepro = aux.Option({'name': 'MinMaxScalar',
                                            'min': 3, 'max': 5.0})
        mm = MatrixMarket(opt)
        mm.create()
        self.assertTrue(True)
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['vali', 'idmap', 'rowwise', 'colwise']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 5)
        self.assertEqual(header['num_users'], 5)
        self.assertEqual(header['num_items'], 3)

        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 5)
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 0, 2, 1, 1])
        self.assertEqual([int(vv) for _, _, vv in data], [3, 5, 3, 3, 4])
        self.assertEqual(data[2], (2, 2, 3.0))

    def test2_implicit_als(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = self.uid_path
        opt.input.iid = self.iid_path
        opt.data.value_prepro = aux.Option({'name': 'ImplicitALS',
                                            'epsilon': 0.5})
        mm = MatrixMarket(opt)
        mm.create()
        self.assertTrue(True)
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['vali', 'idmap', 'rowwise', 'colwise']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 5)
        self.assertEqual(header['num_users'], 5)
        self.assertEqual(header['num_items'], 3)

        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 5)
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 0, 2, 1, 1])
        self.assertAlmostEqual(data[2][2], math.log(1 + 1.0 / 0.5))

    def test3_sppmi(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = self.uid_path
        opt.input.iid = self.iid_path
        opt.data.value_prepro = aux.Option({'name': 'SPPMI'})
        self.assertRaises(RuntimeError, MatrixMarket, opt)


if __name__ == '__main__':
    unittest.main()
