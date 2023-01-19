import numpy as np

from buffalo.parallel._core import quickselect


class Evaluable(object):
    def __init__(self, *args, **kargs):
        pass

    def prepare_evaluation(self):
        if not self.opt.validation or not self.data.has_group("vali"):
            return
        if hasattr(self.data, "vali_data") is False:
            self.data._prepare_validation_data()

    def show_validation_results(self):
        results = self.get_validation_results()
        if not results:
            return "No validation results"
        return "Validation results: " + ", ".join(f"{k}: {v:0.5f}" for k, v in results.items())

    def get_validation_results(self):
        if not self.opt.validation or not self.data.has_group("vali"):
            return

        results = {}
        results.update(self._evaluate_ranking_metrics())
        results.update(self._evaluate_score_metrics())
        return results

    def get_topk(self, scores, k, sorted=True, num_threads=4):
        # NOTE: Is it necessary condition?
        # assert k < scores.shape[1], f"k ({k}) should be smaller than cols ({scores.shape[1]})"
        is_many = True
        if len(scores.shape) == 1:
            scores = scores.reshape(1, scores.shape[0])
            is_many = False
        k = min(k, scores.shape[1])
        assert k > 0, f"k({k}) or cols({scores.shape[1]}) should be greater than 0"
        result = np.empty(shape=(scores.shape[0], k), dtype=np.int32)
        quickselect(scores, result, sorted, num_threads)
        return result if is_many else result[0]

    def _evaluate_ranking_metrics(self):
        if hasattr(self.data, "vali_data") is False:
            self.prepare_evaluation()

        batch_size = self.opt.validation.get("batch", 128)
        topk = self.opt.validation.topk

        gt = self.data.vali_data["vali_gt"]
        rows = self.data.vali_data["vali_rows"]
        validation_seen = self.data.vali_data["validation_seen"]
        validation_max_seen_size = self.data.vali_data["validation_max_seen_size"]
        num_items = self.data.get_header()["num_items"]

        # can significantly save evaluation time
        if self.opt.validation.eval_samples:
            size = min(self.opt.validation.eval_samples, len(rows))
            rows = np.random.choice(rows, size=size, replace=False)

        NDCG = 0.0
        AP = 0.0
        HIT = 0.0
        AUC = 0.0
        N = 0.0

        idcgs = np.cumsum(1.0 / np.log2(np.arange(2, topk + 2)))
        dcgs = 1.0 / np.log2(np.arange(2, topk + 2))

        def filter_seen_items(_topk, seen, topk):
            ret = []
            for t in _topk:
                if t not in seen:
                    ret.append(t)
                    if len(ret) >= topk:
                        break
            return ret

        for index in range(0, len(rows), batch_size):
            recs = self._get_topk_recommendation(rows[index:index + batch_size],
                                                 topk=topk + validation_max_seen_size)
            for row, _topk in recs:
                seen = validation_seen.get(row, set())

                if len(seen) == 0:
                    continue

                _topk = filter_seen_items(_topk, seen, topk)
                _gt = gt[row]

                # accuracy
                hit = len(set(_topk) & _gt) / len(_gt)
                HIT += hit

                # ndcg, map
                idcg = idcgs[min(len(_gt), topk) - 1]
                dcg = 0.0
                hit, miss, ap = 0.0, 0.0, 0.0

                # AUC
                num_pos_items = len(_gt)
                num_neg_items = num_items - num_pos_items
                auc = 0.0

                for i, r in enumerate(_topk):
                    if r in _gt:
                        hit += 1
                        ap += (hit / (i + 1.0))
                        dcg += dcgs[i]
                    else:
                        miss += 1
                        auc += hit
                auc += ((hit + num_pos_items) / 2.0) * (num_neg_items - miss)
                auc /= (num_pos_items * num_neg_items)

                ndcg = dcg / idcg
                NDCG += ndcg
                ap /= min(len(_gt), topk)
                AP += ap
                N += 1.0
                AUC += auc
        NDCG /= N
        AP /= N
        ACC = HIT / N
        AUC = AUC / N
        ret = {"ndcg": NDCG, "map": AP, "accuracy": ACC, "auc": AUC}
        return ret

    def _evaluate_score_metrics(self):
        if hasattr(self.data, "vali_data") is False:
            self.prepare_evaluation()

        vali_data = self.data.vali_data
        row = vali_data["row"]
        col = vali_data["col"]
        val = vali_data["val"]
        scores = self._get_scores(row, col)
        ERROR = 0.0
        RMSE = 0.0
        for r, c, p, v in zip(row, col, scores, val):
            err = p - v
            ERROR += abs(err)
            RMSE += err * err
        RMSE /= len(scores)
        RMSE = RMSE ** 0.5
        ERROR /= len(scores)
        return {"rmse": RMSE, "error": ERROR}
