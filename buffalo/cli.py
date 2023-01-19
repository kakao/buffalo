import fire

from buffalo.algo.als import ALS as _ALS
from buffalo.misc import aux, log


class ALS:
    def __init__(self):
        self.logger = log.get_logger("ALS")

    def run(self, opt_path):
        opt = aux.Option(opt_path)
        als = _ALS(opt_path)
        als.init_factors()
        loss = als.train()
        self.logger.info(f"ALS finished with loss({loss}).")
        if opt.save_factors:
            self.logger.info(f"Saving model to {opt.model_path}.")
            als.save(opt.model_path)

    def optimize(self, opt_path):
        als = _ALS(opt_path)
        als.init_factors()
        als.optimize()
        optimize_loss = als.get_optimization_data()["best"]
        self.logger.info(f"ALS optimization is done with best loss({optimize_loss})")


def execute(algo_name, method, opt_path):
    algo_cls = globals()[algo_name]
    algo = algo_cls()
    getattr(algo, method)(opt_path)


def run(algo_name, opt_path):
    execute(algo_name, "run", opt_path)


def optimize(algo_name, opt_path):
    execute(algo_name, "optimize", opt_path)


def _cli_buffalo():
    fire.Fire({"run": run,
               "optimize": optimize})
