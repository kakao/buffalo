# WARP (Weighted Approximate-Rank Pairwise)

## Compare with BPRMF

**Parameters**

- `num_iters`: 100
- `d`: 40

**Top10** accuracy of validation samples for MovieLens20M:

method | NDCG | AUC | ACCURACY | MAP |
-- | -- | -- | -- | -- 
BPRMF | 0.0460685  | 0.540647 | 0.0816532 | 0.0353039
WARP | 0.0826329  | 0.568376 | 0.137097 | 0.0654066

**Top10** accuracy of validation samples for MovieLens100K:

method | NDCG | AUC | ACCURACY | MAP |
-- | -- | -- | -- | -- 
BPRMF | 0.0870232 | 0.559738 | 0.125085 | 0.0640449
WARP | 0.17361 | 0.62401 | 0.25332 | 0.12941

Please run following command to reproduce this experiment: `$> python test_accuracy.py compare_warp_brp ml20m`

## Compare with LightFM

**Parameters**

- `num_iters`: 100
- `d`: 40

**Top10** accuracy of validation samples for MovieLens100K:

method | NDCG | AUC | ACCURACY | MAP |
-- | -- | -- | -- | -- 
BUFFALO| 0.16562 | 0.62012 | 0.00610| 0.16562
LIGHTFM| 0.03657 | 0.50008 | 0.24548| 0.00365

Please run following command to reproduce this experiment: `$> python test_accuracy.py accuracy warp ml100k --libs=buffalo,lightfm`
