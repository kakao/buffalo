# -*- coding: utf-8 -*-
import math


class PreProcess(object):
    def __init__(self, opt):
        self.opt = opt

    def pre(self, header):
        pass

    def __call__(self, v):
        return v

    def post(self, db):
        pass


class OneBased(PreProcess):
    def __init__(self, opt):
        super(OneBased, self).__init__(opt)

    def pre(self, header):
        return

    def __call__(self, v):
        return 1.0

    def post(self, db):
        return


class MinMaxScalar(PreProcess):
    def __init__(self, opt):
        super(MinMaxScalar, self).__init__(opt)
        self.value_min = 987654321.0
        self.value_max = 0.0

    def pre(self, header):
        return

    def __call__(self, v):
        self.value_min = min(self.value_min, v)
        self.value_max = max(self.value_max, v)
        return v

    def post(self, db):
        sz = db['val'].shape[0]
        per = db['val'].chunks[0]
        for idx in range(0, sz, per):
            chunk = db['val'][idx:idx + per]
            chunk = (chunk - self.value_min) / (self.value_max - self.value_min)
            chunk = chunk * (self.opt.max - self.opt.min) + self.opt.min
            db['val'][idx:idx + per] = chunk


class ImplicitALS(PreProcess):
    def __init__(self, opt):
        super(ImplicitALS, self).__init__(opt)

    def pre(self, header):
        return

    def __call__(self, v):
        return math.log(1 + v / self.opt.epsilon)

    def post(self, db):
        return


class SPPMI(PreProcess):
    """NotImplemented Yet"""
    def __init__(self, opt):
        super(SPPMI, self).__init__(opt)

    def pre(self, header):
        return

    def __call__(self, v):
        return v

    def post(self, db):
        return
