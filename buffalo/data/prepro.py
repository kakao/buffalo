# -*- coding: utf-8 -*-


class PreProcess(object):
    def pre(self, header):
        pass

    def __call__(self, v):
        return v

    def post(self, db):
        pass


class OneBased(PreProcess):
    def pre(self, header):
        return

    def __call__(self, v):
        return 1.0

    def post(self, db):
        return
