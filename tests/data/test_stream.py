# -*- coding: utf-8 -*-
import os
import unittest
import tempfile

from buffalo.data.stream import Stream, StreamOptions


class TestStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''apple mango mango apple pie juice coke\npie\njuice coke grape''')
            cls.main_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''kim\nlee\npark''')
            cls.uid_path = f.name
        cls.temp_files = []

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.main_path)
        os.remove(cls.uid_path)
        for path in cls.temp_files:
            os.remove(path)

    def test0_get_default_option(self):
        StreamOptions().get_default_option()
        self.assertTrue(True)

    def test1_is_valid_option(self):
        opt = StreamOptions().get_default_option()
        self.assertTrue(StreamOptions().is_valid_option(opt))
        opt['type'] = 1
        self.assertRaises(RuntimeError, StreamOptions().is_valid_option, opt)
        opt['type'] = 'stream'
        self.assertTrue(StreamOptions().is_valid_option(opt))

    def test2_create(self):
        opt = StreamOptions().get_default_option()
        opt.input.main = self.main_path
        opt.input.uid = self.uid_path
        mm = Stream(opt)
        mm.create()
        self.temp_files.append(opt.data.path)
        self.assertTrue(True)
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['idmap', 'rowwise', 'colwise', 'vali']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 9)  # due to validation samples
        self.assertEqual(header['num_users'], 3)
        self.assertEqual(header['num_items'], 6)

        mm.build_idmaps()
        data = [(u, kk) for u, kk in mm.iterate(use_repr_name=True)]
        self.assertEqual(len(data), 9)
        self.assertEqual([kk for _, kk in data], ['apple', 'mango', 'mango', 'apple', 'pie', 'juice', 'pie', 'juice', 'coke'])

    def test3_to_matrix(self):
        opt = StreamOptions().get_default_option()
        opt.input.main = self.main_path
        opt.input.uid = self.uid_path
        opt.data.internal_data_type = 'matrix'
        mm = Stream(opt)
        mm.create()
        self.assertTrue(True)
        db = mm.handle
        self.assertEqual(sorted(db.keys()), sorted(['idmap', 'rowwise', 'colwise', 'vali']))
        header = mm.get_header()
        self.assertEqual(header['num_nnz'], 7)  # due to validation samples
        self.assertEqual(header['num_users'], 3)
        self.assertEqual(header['num_items'], 6)

        mm.build_idmaps()
        data = [(u, kk, vv) for u, kk, vv in mm.iterate()]
        self.assertEqual(len(data), 7)
        self.assertEqual([uu for uu, _, _ in data], [0, 0, 0, 0, 1, 2, 2])

        data = [(u, kk, vv) for u, kk, vv in mm.iterate(axis='colwise')]
        data = [(u, kk, vv) for u, kk, vv in mm.iterate(axis='colwise', use_repr_name=True)]
        data.sort()
        self.assertEqual([uu for uu, _, _ in data], ['apple', 'coke', 'juice', 'juice', 'mango', 'pie', 'pie'])


if __name__ == '__main__':
    unittest.main()
