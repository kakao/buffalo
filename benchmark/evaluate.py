import numpy as np


def evaluate_ranking_metrics(recs, topk, vali_data, total_items):
    gt = vali_data["vali_gt"]
    validation_seen = vali_data["validation_seen"]

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

    for userid, _topk in recs:
        seen = validation_seen.get(userid, set())

        if len(seen) == 0:
            continue

        _topk = filter_seen_items(_topk, seen, topk)
        _gt = gt[userid]

        # accuracy
        hit = len(set(_topk) & _gt) / len(_gt)
        HIT += hit

        # ndcg, map
        idcg = idcgs[min(len(_gt), topk) - 1]
        dcg = 0.0
        hit, miss, ap = 0.0, 0.0, 0.0

        # AUC
        num_pos_items = len(_gt)
        num_neg_items = total_items - num_pos_items
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
