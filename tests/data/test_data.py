# -*- coding: utf-8 -*-
import os
import unittest
import tempfile

from buffalo.data import MatrixMarket, MatrixMarketOptions


class TestMatrixMarket(unittest.TestCase):
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

    def data_serialize(self, x):
        a, b, c = x
        return (a, list(b), list(c))

    def test0_get_default_option(self):
        MatrixMarketOptions().get_default_option()
        self.assertTrue(True)

    def test1_is_valid_option(self):
        opt = MatrixMarketOptions().get_default_option()
        self.assertTrue(MatrixMarketOptions().is_valid_option(opt))
        opt['type'] = 1
        self.assertRaises(RuntimeError, MatrixMarketOptions().is_valid_option, opt)
        opt['type'] = 'matrix_market'
        self.assertTrue(MatrixMarketOptions().is_valid_option(opt))

    def test2_create(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = self.uid_path
        opt.input.iid = self.iid_path
        mm = MatrixMarket(opt)
        mm.create()
        self.temp_files.append(opt.data.path)
        self.assertTrue(True)
        db = mm.db
        self.assertEqual(sorted(db.keys()), sorted(['header', 'idmap', 'rowwise', 'colwise']))
        self.assertEqual(db['header']['num_nnz'][0], 5)
        self.assertEqual(db['header']['num_users'][0], 5)
        self.assertEqual(db['header']['num_items'][0], 3)

        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 5)
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 0, 2, 1, 1])
        self.assertEqual(data[2], (2, 2, 1.0))

        data = [(u, kk, vv) for u, kk, vv in mm.iterate(axis='colwise')]
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 1, 3, 4, 2])

        self.assertEqual(self.data_serialize(mm.get_data(0)), (0, [0], [1.]))

        self.assertEqual(self.data_serialize(mm.get_data(0, axis='colwise')),
                         (0, [0, 1], [1., 3.]))

if __name__ == '__main__':
    unittest.main()
