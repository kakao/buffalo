# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from buffalo.misc import log


class TFALS(object):
    def __init__(self, opt, name="tf_als"):
        self.logger = log.get_logger("tf-als")
        self.opt = opt
        self.name = name
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

    def initialize_model(self, P, Q):
        with tf.variable_scope(self.name):
            self.P = tf.get_variable("P", initializer=P)
            self.Q = tf.get_variable("Q", initializer=Q)
            self.FF = tf.get_variable("FF", shape=(self.opt.d, self.opt.d),
                                      initializer=tf.keras.initializers.Zeros,
                                      dtype=tf.float32)
            self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def get_variable(self, name):
        return self.sess.run(getattr(self, name))

    def precompute(self, int_group):
        if int_group == 0:
            self.sess.run(self.precomputeQ)
        else:
            self.sess.run(self.precomputeP)

    def build_graph(self):
        self.start_x = tf.placeholder(dtype=tf.int32, name="start_x")
        self.next_x = tf.placeholder(dtype=tf.int32, name="next_x")
        self.rows = tf.placeholder(dtype=tf.int32, shape=(None, ), name="rows")
        self.keys = tf.placeholder(dtype=tf.int32, shape=(None, ), name="keys")
        self.vals = tf.placeholder(dtype=tf.float32, shape=(None, ), name="vals")
        for int_group in [0, 1]:
            self._build_graph(int_group)

    def _dot(self, X, Y):
        return tf.reduce_sum(X * Y, axis=1)

    def _build_graph(self, int_group):
        if int_group == 0:
            P, Q, reg = self.P, self.Q, self.opt.reg_u
        else:
            P, Q, reg = self.Q, self.P, self.opt.reg_i
        start_x, next_x, rows, keys, vals = \
            self.start_x, self.next_x, self.rows, self.keys, self.vals

        # compute ys
        Fgtr = tf.gather(Q, keys)
        coeff = self.vals * self.opt.alpha
        ys = tf.scatter_nd(tf.expand_dims(rows, axis=1),
                           Fgtr * tf.expand_dims(coeff + 1, axis=1),
                           shape=(next_x - start_x, self.opt.d))

        # prepare cg
        _P = P[start_x:next_x]
        Axs = tf.matmul(_P, self.FF) + reg * _P
        dots = self._dot(tf.gather(_P, rows), Fgtr)
        Axs = tf.tensor_scatter_add(Axs, tf.expand_dims(rows, axis=1),
                                    Fgtr * tf.expand_dims(dots * coeff, axis=1))
        rs = ys - Axs
        ps = rs
        rss_old = tf.reduce_sum(tf.square(rs), axis=1)

        # iterate cg steps
        for i in range(self.opt.num_cg_max_iters):
            Aps = tf.matmul(ps, self.FF) + ps * reg
            _dots = coeff * self._dot(tf.gather(ps, rows), Fgtr)
            Aps = tf.tensor_scatter_add(Aps, tf.expand_dims(rows, axis=1),
                                        Fgtr * tf.expand_dims(_dots, axis=1))
            pAps = self._dot(Aps, ps)
            alphas = rss_old / (pAps + self.opt.eps)
            _P = _P + ps * tf.expand_dims(alphas, axis=1)
            rs = rs - tf.expand_dims(alphas, axis=1) * Aps
            rss_new = tf.reduce_sum(tf.square(rs), axis=1)
            betas = rss_new / (rss_old + self.opt.eps)
            ps = rs + (tf.expand_dims(betas, axis=1) * ps)
            rss_old = rss_new

        if int_group == 1:
            if self.opt.compute_loss_on_training:
                self.err = tf.reduce_sum(tf.square(vals - dots))
            else:
                self.err = tf.constant(0.0, dtype=tf.float32)

        name = "updateP" if int_group == 0 else "updateQ"
        _update = P[start_x:next_x].assign(_P)
        with self.graph.control_dependencies([_update]):
            update = tf.constant(True)
        setattr(self, name, update)

        _FF = tf.assign(self.FF, tf.matmul(P, P, transpose_a=True))
        with self.graph.control_dependencies([_FF]):
            FF = tf.constant(True)
        name = "precomputeP" if int_group == 0 else "precomputeQ"
        setattr(self, name, FF)

    def _generate_rows(self, start_x, next_x, indptr):
        ends = indptr[start_x:next_x]
        begs = np.empty(next_x - start_x, dtype=np.int64)
        begs[0] = 0 if start_x == 0 else indptr[start_x - 1]
        begs[1:] = ends[:-1]
        ret = np.arange(next_x - start_x, dtype=np.int32)
        ret = np.repeat(ret, ends - begs)
        return ret, len(ret)

    def partial_update(self, start_x, next_x, indptr, keys, vals, int_group):
        rows, sz = self._generate_rows(start_x, next_x, indptr)
        feed_dict = {self.start_x: start_x, self.next_x: next_x,
                     self.rows: rows, self.keys: keys[:sz], self.vals: vals[:sz]}
        err = 0.0
        if int_group == 0:
            _ = self.sess.run(self.updateP, feed_dict=feed_dict)
        else:
            _, _err = self.sess.run([self.updateQ, self.err], feed_dict=feed_dict)
            err += _err
        return err
