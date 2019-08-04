# -*- coding: utf-8 -*-
from buffalo.misc import aux


def load(opt):
    from buffalo.data.mm import MatrixMarket
    from buffalo.data.stream import Stream
    if isinstance(opt, (str,)):
        opt = aux.Option(opt)

    assert isinstance(opt, (dict, aux.Option)), 'opt must be either str, or dict/aux.Option but {}'.format(type(opt))
    if opt['type'] == 'matrix_market':
        return MatrixMarket(opt)
    if opt['type'] == 'stream':
        return Stream(opt)
    raise RuntimeError('Unexpected data.type: {}'.format(opt['type']))
