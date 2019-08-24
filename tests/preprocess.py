# -*- coding: utf-8 -*-
import os
import time
import unittest
from collections import Counter

from buffalo.misc import aux, log
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions


def iterate_brunch_data_files(data_root):
    for fname in os.listdir(data_root):
        if len(fname) != len('2018100100_2018100103'):
            continue
        path = os.path.join(data_root, fname)
        yield path, fname


def make_mm_from_stream(stream_dir, to_dir):
    os.makedirs(to_dir, exist_ok=True)

    user_path, main_path = os.path.join(stream_dir, 'uid'), os.path.join(stream_dir, 'main')
    uids = {u.strip(): idx + 1 for idx, u in enumerate(open(user_path))}

    iids = set()
    num_nnz = 0
    with open(main_path) as fin:
        for line in fin:
            items = set()
            for item in line.strip().split():
                items.add(item)
            iids |= items
            num_nnz += len(items)
    iids = {i: idx + 1 for idx, i in enumerate(iids)}

    with open(os.path.join(to_dir, 'main'), 'w') as fout:
        fout.write('%MatrixMarket matrix coordinate integer general\n')
        fout.write('%d %d %d\n' % (len(uids), len(iids), num_nnz))
        for idx, line in enumerate(open(main_path)):
            items = line.strip().split()
            items = Counter(items)
            for item, cnt in items.items():
                fout.write('%s %s %s\n' % (idx + 1, iids[item], cnt))
    with open(os.path.join(to_dir, 'uid'), 'w') as fout:
        uids = sorted(uids.items(), key=lambda x: x[1])
        for u, _ in uids:
            fout.write(u + '\n')
    with open(os.path.join(to_dir, 'iid'), 'w') as fout:
        iids = sorted(iids.items(), key=lambda x: x[1])
        for i, _ in iids:
            fout.write(i + '\n')


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

    if not os.path.isdir('brunch'):
        logger.warn('Cannot find the brunch directory')
    else:
        if not os.path.isfile('./ext/brunch/main'):
            os.makedirs('./ext/brunch/tmp', exist_ok=True)
            to_dir = './ext/brunch/tmp'

            logger.info('dividing...')
            num_chunks = 30
            fouts = {i: open(os.path.join(to_dir, str(i)), 'w')
                    for i in range(num_chunks)}
            for path, fname in iterate_brunch_data_files('./ext/brunch'):
                for line in open(path):
                    uid = line.strip().split()[0]
                    fid = hash(uid) % num_chunks
                    fouts[fid].write(line)
            for val in fouts.values():
                val.close()

            logger.info('merging...')
            with open('./ext/brunch/main', 'w') as fout, \
                    open('./ext/brunch/uid', 'w') as fout_uid:
                for fid in fouts.keys():
                    seens = {}
                    chunk_path = os.path.join(to_dir, str(fid))
                    for line in open(chunk_path):
                        line = line.strip().split()
                        uid, seen = line[0], line[1:]
                        seens.setdefault(uid, []).extend(seen)
                    for uid, seen in seens.items():
                        fout.write(' '.join(seen) + '\n')
                        fout_uid.write(uid + '\n')
                for fid in fouts.keys():
                    chunk_path = os.path.join(to_dir, str(fid))
                    os.remove(chunk_path)
    make_mm_from_stream('./ext/brunch/', './ext/brunch/mm')



if __name__ == '__main__':
    prepare_dataset()
