import json
import time

import fire

from buffalo.algo import ALS, ALSOption
from buffalo.data import MatrixMarketOptions
from buffalo.misc import aux, log
from buffalo.parallel import ParALS


def example1():
    log.set_log_level(log.DEBUG)
    als_option = ALSOption().get_default_option()
    als_option.validation = aux.Option({"topk": 10})
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = "../tests/ext/ml-100k/main"
    data_option.input.iid = "../tests/ext/ml-100k/iid"

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    print("MovieLens 100k metrics for validations\n%s" % json.dumps(als.get_validation_results(), indent=2))

    print("Similar movies to Star_Wars_(1977)")
    for rank, (movie_name, score) in enumerate(als.most_similar("49.Star_Wars_(1977)")):
        print(f"{rank + 1:02d}. {score:.3f} {movie_name}")


def example2():
    log.set_log_level(log.INFO)
    als_option = ALSOption().get_default_option()
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = "../tests/ext/ml-100k/main"
    data_option.input.iid = "../tests/ext/ml-100k/iid"
    # data_option.data.path = "./ml20m.h5py"
    data_option.data.use_cache = True

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    als.normalize("item")
    als.build_itemid_map()

    print("Make item recommendation on als.ml20m.par.top10.tsv with Parallel(Thread=4)")
    par = ParALS(als)
    par.num_workers = 4
    all_items = als._idmanager.itemids
    start_t = time.time()
    with open("als.ml20m.par.top10.tsv", "w") as fout:
        for idx in range(0, len(all_items), 128):
            topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
            for q, p in zip(all_items[idx:idx + 128], topks):
                fout.write("%s\t%s\n" % (q, "\t".join(p)))
    print("took: %.3f secs" % (time.time() - start_t))

    try:
        from n2 import HnswIndex

        index = HnswIndex(als.Q.shape[1])
        for f in als.Q:
            index.add_data(f)
        index.build(n_threads=4)
        index.save("ml20m.n2.index")
        index.unload()
        print("Make item recommendation on als.ml20m.par.top10.tsv with Ann(Thread=1)")
        par.set_hnsw_index("ml20m.n2.index", "item")
        par.num_workers = 4
        start_t = time.time()
        with open("als.ml20m.ann.top10.tsv", "w") as fout:
            for idx in range(0, len(all_items), 128):
                topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
                for q, p in zip(all_items[idx:idx + 128], topks):
                    fout.write("%s\t%s\n" % (q, "\t".join(p)))
        print("took: %.3f secs" % (time.time() - start_t))
    except ImportError:
        print("n2 is not installed. skip it")


if __name__ == "__main__":
    fire.Fire({"example1": example1,
               "example2": example2})
