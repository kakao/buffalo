# -*- coding: utf-8 -*-
# import time

import cupy as cp
import numpy as np
import cupyx as cpx

from buffalo.misc import log


class CupyALS(object):
    def __init__(self, opt):
        self.logger = log.get_logger("cupy-als")
        self.opt = opt
        self.err = cp.float32(0.0)

    def init_variable(self, name, shape):
        setattr(self, name, None)
        _F = cp.random.normal(scale=1.0/np.square(self.opt.d),
                              size=shape, dtype=cp.float32)
        setattr(self, name, cp.abs(_F))

    def get_variable(self, name):
        return cp.asnumpy(getattr(self, name))

    def get_err(self):
        ret = np.float32(cp.asnumpy(self.err))
        self.err = cp.float32(0.0)
        return ret

    def precompute(self, int_group):
        F = self.Q if int_group == 0 else self.P
        self.FF = cp.matmul(cp.transpose(F), F)

    def dot(self, X, Y):
        return cp.sum(X * Y, axis=1)

    def partial_update(self, start_x, next_x, rows, keys, vals, int_group):
        rows, keys, vals = map(cp.array, [rows, keys, vals])
        rows -= start_x
        if int_group == 0:
            P, Q, reg = self.P, self.Q, self.opt.reg_u
        else:
            P, Q, reg = self.Q, self.P, self.opt.reg_i

        # compute ys
        Fgtr = Q[keys]
        coeff = vals * self.opt.alpha
        ys = cp.zeros(shape=(next_x - start_x, self.opt.d), dtype=cp.float32)
        cpx.scatter_add(ys, rows, Fgtr * cp.expand_dims(1.0 + coeff, axis=1))

        # prepare cg
        Axs = cp.matmul(P[start_x: next_x], self.FF) + reg * P[start_x: next_x]
        dots = self.dot(P[rows + start_x], Fgtr)
        cpx.scatter_add(Axs, rows, Fgtr * cp.expand_dims(dots * coeff, axis=1))
        rs = ys - Axs
        ps = cp.copy(rs)
        rss_old = cp.sum(cp.square(rs), axis=1)

        # iterate cg steps
        for i in range(self.opt.num_cg_max_iters):
            Aps = cp.matmul(ps, self.FF) + ps * reg
            _dots = coeff * self.dot(ps[rows], Fgtr)
            cpx.scatter_add(Aps, rows, Fgtr * cp.expand_dims(_dots, axis=1))
            pAps = self.dot(Aps, ps)
            alphas = rss_old / (pAps + self.opt.eps)
            P[start_x: next_x] += (ps * cp.expand_dims(alphas, axis=1))
            rs -= cp.expand_dims(alphas, axis=1) * Aps
            rss_new = cp.sum(cp.square(rs), axis=1)
            betas = rss_new / (rss_old + self.opt.eps)
            ps = rs + (cp.expand_dims(betas, axis=1) * ps)
            rss_old = rss_new

        # compute loss for the part of data fidelity (compute only in item-side)
        if self.opt.compute_loss_on_training and int_group == 1:
            self.err += cp.sum(cp.square(vals - dots))

        return 0.0
