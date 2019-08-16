# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import cupyx as cpx
import tensorflow as tf


class CupyALS(object):
    def __init__(self):
        pass

    def initialize_cupy(self):
        assert self.accelerator, 'no accelerator'
        assert self.data, 'Data is not setted'
        header = self.data.get_header()
        num_users, num_items, d = \
            header["num_users"], header["num_items"], self.opt.d
        for attr, shape in [('P', (num_users, d)),
                            ('Q', (num_items, d))]:
            setattr(self, attr, None)
            F = cp.random.normal(scale=1.0/(d ** 2), size=shape, dtype=cp.float32)
            setattr(self, attr, F)

    def _generate_rows(start_x, next_x, indptr):
        # generate rows from indptr
        rows = np.arange(next_x - start_x, dtype=np.int64)
        ends = indptr[start_x: next_x]
        begs = np.empty(next_x - start_x, dtype=np.int64)
        begs[0] = 0 if start_x == 0 else indptr[start_x - 1]
        begs[1:] = indptr[start_x: next_x - 1]
        return np.repeat(rows, ends - begs)

    def _update_cg(self, xs, As, ys):
        # update conjugate gradient
        xs = cp.expand_dims(xs, axis=1)
        ys = cp.expand_dims(ys, axis=1)
        rs = ys - tf.matmul(xs, As)
        rnorms = cp.squeeze(cp.linalg.norm(rs, axis=2), axis=1)
        ynorms = cp.squeeze(cp.linalg.norm(ys, axis=2), axis=1)
        zero_inits = cp.where(rnorms > ynorms)[0]
        xs[zero_inits, :, :] = 0.0
        rs[zero_inits, :, :] = ys[zero_inits, :, :]
        ps = cp.copy(rs)
        rnorms_old = cp.expand_dims(cp.expand_dims(rnorms, axis=1), axis=1)
        for i in range(self.opt.num_cg_max_iters):
            pAps = cp.matmul(ps, cp.matmul(ps, As))
            alphas = rnorms / pAps
            xs += alphas * ps
            rs -= alphas * cp.matmul(ps, As)
            rnorms_new = cp.linalg.norm(rs, axis=2, keepdims=True)
            if cp.all(cp.square(rnorms_new) < self.opt.cg_tolerance):
                break
            betas = rnorms_new / rnorms_old
            ps = rs + betas * ps
            rnorms_old = rnorms_new
        return cp.squeeze(xs, axis=1)

    def partial_update_cupy(self, start_x, next_x, indptr, keys, vals, int_group):
        err = 0.0
        rows = self._generate_rows(start_x, next_x, indptr)
        if int_group == 0:
            P, Q = self.P, self.Q
        else:
            P, Q = self.Q, self.P

        # compute As
        As = cp.repeat(self.FF + self.opt.reg_u * cp.eye(self.opt.dim, dtype=cp.float32),
                       next_x - start_x, axis=0)
        Fgtr = Q[keys]
        _Fgtr = cp.multiply(Fgtr, cp.expand_dims(vals * self.opt.alpha, axis=1))
        Fgtr_ext = cp.transpose(cp.expand_dims(Fgtr, axis=1), axes=[0, 2, 1])
        _Fgtr_ext = cp.expand_dims(_Fgtr, axis=1)
        FtFgtr = cp.matmul(Fgtr_ext, _Fgtr_ext)
        cpx.scatter_add(As, rows, FtFgtr)

        # compute ys
        ys = cp.zeros(shape=(next_x - start_x, self.opt.dim), dtype=cp.float32)
        cpx.scatter_add(ys, rows, _Fgtr)

        # compute loss for the part of data fidelity (compute only in item-side)
        if self.opt.compute_loss and self.int_group:
            Ps = P[start_x: next_x]
            _err = cp.sum(cp.matmul(Ps, self.FF), cp.transpose(Ps))
            dots = cp.sum(cp.multiply(Ps[rows], Fgtr), axis=1)
            _err_neg = cp.sum(cp.square(dots))
            _err_pos = cp.sum(cp.multiply(cp.square(dots - 1), vals)) * self.opt.alpha
            err += np.float32(cp.asnumpy(_err - _err_neg + _err_pos))

        # update cg
        P[start_x: next_x] = self._update_cg(P[start_x: next_x], As, ys)

        # compute loss for the part of regularization
        if self.opt.compute_loss:
            _err = cp.linalg.norm(self.U[start_x: next_x])
            err += self.opt.reg_u * np.float32(cp.asnumpy(_err)) ** 2
        return err
