# -*- coding: utf-8 -*-
from buffalo.misc import aux


def load(opt):
    from buffalo.data.mm import MatrixMarket
    if isinstance(opt, (str,)):
        opt = aux.Option(opt)

    assert isinstance(opt, (dict, aux.Option)), 'opt must be either str, or dict/aux.Option but {}'.format(type(opt))
    if opt['type'] == 'matrix_market':
        return MatrixMarket(opt)
    raise RuntimeError('Unexpected data.type: {}'.format(opt['type']))
