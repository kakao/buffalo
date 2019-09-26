# -*- coding: utf-8 -*-
import os
import time
import unittest
from collections import Counter

from buffalo.misc import aux, log
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions


def prepare_dataset():
    logger = log.get_logger()
    if not os.path.isdir('ext/ml-100k/'):
        logger.warn('Cannot find the ./ext/ml-100k directory')
    else:
        if not os.path.isfile('./ext/ml-100k/main'):
            logger.info('preprocessing for matrix market format of ml-100k...')
            in_path = "./ext/ml-100k/u.data"
            stream_out_path = "./ext/ml-100k/stream"
            aux.psort(in_path, field_seperator="\t", key=4)
            aux.psort(in_path, field_seperator="\t", key=1)

            with open('./ext/ml-100k/main', 'w') as fout:
                fout.write('%%MatrixMarket matrix coordinate integer general\n%\n%\n943 1682 80000\n')
                with open(in_path) as fin:
                    for line in fin:
                        u, i, v, ts = line.strip().split('\t')
                        fout.write('%s %s %s\n' % (u, i, v))

            iids = []
            with open('./ext/ml-100k/iid', 'w') as fout:
                with open('./ext/ml-100k/u.item', encoding='ISO-8859-1') as fin:
                    iids = [line.strip().split('|')[1].replace(' ', '_') for line in fin]
                iids = [f"{idx}.{key}" for idx, key in enumerate(iids)]
                fout.write("\n".join(iids))

            with open('./ext/ml-100k/uid', 'w') as fout:
                for line in open('./ext/ml-100k/u.user'):
                    userid = line.strip().split('|')[0]
                    fout.write('%s\n' % userid)

            logger.info('preprocessing for stream format of ml-100k...')
            probe, bag = None, []
            with open(in_path, "r") as fin, open(stream_out_path, "w") as fout:
                for line in fin:
                    u, i, v, ts = line.strip().split("\t")
                    if not probe:
                        probe = u
                    elif probe != u:
                        fout.write(" ".join(bag) + "\n")
                        probe, bag = u, []
                    bag.append(iids[int(i) - 1])
                if bag:
                    fout.write(" ".join(bag))

    if not os.path.isdir('ext/ml-20m'):
        logger.warn('Cannot find the ./ml-20m directory')
    else:
        if not os.path.isfile('./ext/ml-20m/main'):
            logger.info('preprocessing for matrix market format of ml-20m...')
            uids, iids = {}, {}
            in_path = "./ext/ml-20m/ratings.csv"
            aux.psort(in_path, field_seperator=",", key=4)
            aux.psort(in_path, field_seperator=",", key=1)
            with open(in_path) as fin:
                fin.readline()
                for line in fin:
                    uid = line.split(',')[0]
                    if uid not in uids:
                        uids[uid] = len(uids) + 1
            with open('./ext/ml-20m/uid', 'w') as fout:
                for uid, _ in sorted(uids.items(), key=lambda x: x[1]):
                    fout.write('%s\n' % uid)
            with open('./ext/ml-20m/movies.csv') as fin:
                fin.readline()
                for line in fin:
                    iid = line.split(',')[0]
                    iids[iid] = len(iids) + 1
            with open('./ext/ml-20m/iid', 'w') as fout:
                for iid, _ in sorted(iids.items(), key=lambda x: x[1]):
                    fout.write('%s\n' % iid)
            with open('./ext/ml-20m/main', 'w') as fout:
                fout.write('%%MatrixMarket matrix coordinate real general\n%\n%\n138493 27278 20000263\n')
                with open('./ext/ml-20m/ratings.csv') as fin:
                    fin.readline()
                    for line in fin:
                        uid, iid, r, *_ = line.split(',')
                        uid, iid = uids[uid], iids[iid]
                        fout.write(f'{uid} {iid} {r}\n')
            logger.info('preprocessing for stream format of ml-20m...')
            probe, bag = None, []
            stream_out_path = "./ext/ml-20m/stream"
            with open(in_path, "r") as fin, open(stream_out_path, "w") as fout:
                fin.readline()
                for line in fin:
                    u, i, v, ts = line.strip().split(",")
                    if not probe:
                        probe = u
                    elif probe != u:
                        fout.write(" ".join(bag) + "\n")
                        probe, bag = u, []
                    bag.append(i)
                if bag:
                    fout.write(" ".join(bag))
    if not os.path.isdir('ext/text8'):
        logger.warn('Cannot find the text8 directory')
    else:
        if not os.path.isfile('./ext/text8/main'):
            with open('./ext/text8/text8') as fin:
                words = fin.readline().strip().split()
                with open('./ext/text8/main', 'w') as fout:
                    for i in range(0, len(words), 1000):
                        fout.write('%s\n' % ' '.join(words[i:i + 1000]))


if __name__ == '__main__':
    prepare_dataset()
