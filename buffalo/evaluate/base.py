# -*- coding: utf-8 -*-
import math


class Evaluable(object):
    def __init__(self, *args, **kargs):
        pass

    def prepare_evaluation(self):
        if not self.opt.validation or not self.data.has_group('vali'):
            return
        vali = self.data.get_group('vali')
        validation_seen = {}
        rows = set([r for r in vali['row'][::]])
        max_seen_size = 0
        for rowid in rows:
            seen, *_ = self.data.get(rowid)
            validation_seen[rowid] = set(seen)
            max_seen_size = max(len(seen), max_seen_size)
        self._validation_seen = validation_seen
        self._validation_max_seen_size = max_seen_size

    def show_validation_results(self):
        results = self.get_validation_results()
        if not results:
            return 'No validation results'
        return 'Validation results: ' + ', '.join(f'{k}: {v:0.5f}' for k, v in results.items())

    def get_validation_results(self):
        if not self.opt.validation or not self.data.has_group('vali'):
            return

        results = {}
        results.update(self._evaluate_ranking_metrics())
        results.update(self._evaluate_score_metrics())
        return results

    def _evaluate_ranking_metrics(self):
        batch_size = self.opt.validation.get('batch', 128)
        topk = self.opt.validation.topk
        vali = self.data.get_group('vali')
        row = vali['row'][::]
        col = vali['col'][::]
        rows, gt = set(), {}
        for r, c in zip(row, col):
            gt.setdefault(r, set()).add(c)
            rows.add(r)
        rows = list(rows)

        NDCG = 0.0
        AP = 0.0
        HIT = 0.0
        N = 0.0
        for index in range(0, len(rows), batch_size):
            recs = self._get_topk_recommendation(rows[index:index + batch_size],
                                                 topk=topk + self._validation_max_seen_size)
            for row, _topk in recs:
                seen = self._validation_seen.get(row, set())
                _topk = [t for t in _topk if t not in seen][:topk]
                _gt = gt[row]

                # accuracy
                hit = len(set(_topk) & _gt) / len(_gt)
                HIT += hit

                # ndcg, map
                idcg = sum([1.0 / math.log(i + 2, 2)
                            for i in range(min(len(_gt), len(_topk)))])
                dcg = 0.0
                hit, ap = 0.0, 0.0
                for i, r in enumerate(_topk):
                    if r in _gt:
                        hit += 1
                        ap += (hit / (i + 1.0))
                    if r not in _gt:
                        continue
                    rank = i + 1
                    dcg += 1.0 / math.log(rank + 1, 2)
                ndcg = dcg / idcg
                NDCG += ndcg
                ap /= min(len(_gt), len(_topk))
                AP += ap
                N += 1.0
        NDCG /= N
        AP /= N
        ACC = HIT / N
        return {'ndcg': NDCG, 'map': AP, 'accuracy': ACC}

    def _evaluate_score_metrics(self):
        vali = self.data.get_group('vali')
        row = vali['row'][::]
        col = vali['col'][::]
        val = vali['val'][::]
        row_col_pairs = zip(row, col)
        vals = {(r, c): v for r, c, v in zip(row, col, val)}
        scores = self.get_scores(row_col_pairs)
        ERROR = 0.0
        RMSE = 0.0
        for (r, c), p in scores.items():
            err = p - vals[(r, c)]
            ERROR += abs(err)
            RMSE += err * err
        RMSE /= len(scores)
        RMSE = RMSE ** 0.5
        ERROR /= len(scores)
        return {'rmse': RMSE, 'error': ERROR}
