# -*- coding: utf-8 -*-
import os
import unittest
import tempfile

from buffalo.data.mm import MatrixMarket, MatrixMarketOptions


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
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['vali', 'idmap', 'rowwise', 'colwise']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 5)
        self.assertEqual(header['num_users'], 5)
        self.assertEqual(header['num_items'], 3)

        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 5)
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 0, 2, 1, 1])
        self.assertEqual(data[2], (2, 2, 1.0))

        data = [(u, kk, vv) for u, kk, vv in mm.iterate(axis='colwise')]
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 1, 3, 4, 2])

    def test3_create2(self):
        opt = MatrixMarketOptions().get_default_option()
        opt.input.main = self.mm_path
        opt.input.uid = None
        opt.input.iid = None
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
        self.assertEqual(data[2], (2, 2, 1.0))

        data = [(u, kk, vv) for u, kk, vv in mm.iterate(axis='colwise')]
        self.assertEqual([int(kk) for _, kk, _ in data], [0, 1, 3, 4, 2])

    def test4_get(self):
        # TODO: implement
        pass

if __name__ == '__main__':
    unittest.main()
