# Benchmark
We ran benchmark buffalo with the well known open source libraries so far.

- [Apache Spark](https://spark.apache.org)
- [Quora QMF](https://github.com/quora/qmf)
- [lyst LightFM](https://github.com/lyst/lightfm)
- [Implicit](https://github.com/benfred/implicit)

We tested two algorithms, ALS, BPRMF and measured training speed and memory usage for vairous datasets. Some of libraries not support either ALS or BPRMF, in that case excluded.

- Test Environments
  - CPU Machine
    - CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (6 cores)
    - RAM: 64GB
    - SSD
  - GPU Machine
    - CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
    - GPU: NVIDIA Tesla V100 (Cuda 10, we use only 1 card)
    - RAM: 64GB
    - SSD
  - gcc/g++-7.2.1
  - Python 3.6

To download the databases, please see the README.md on ./tests/.
- Databases
  - KakaoBrunch12M
    - Sampled user reading history from Kakao Brunch service, which kind of blog service, during 6 months.
  - KakaoReco730
    - Sampled user reading history from Kakao annoymous service during 8 days.
  - Movielens20M
  - Movielens100K


## Alternating Least Square

- Fixed options (otherwise we let them as default except controlled options)
  - Buffalo
    - `optimizer=manual_cg num_cg_max_iters=3 compute_loss_on_training=False`
  - Implicit
    - `dtype=np.float32 use_cg=True use_native=True calculate_training_loss=False`
  - Spark
    - `implicitPrefs=True intermediateStroageLevel=MEMORY_ONLY finalStorageLevel=MEMORY_ONLY`
  - QMF
    - `asis`

Note that there is no Python version of QMF. Since we ran benchmark by Python script, we have to capture printed datetime information from standard output of QMF.


## Bayesian Personalized Ranking Matrix Factorization

- Fixed options (otherwise we let them as default except controlled options)
  - Buffalo
    - `compute_loss_on_training=False`
  - Implicit
    - `dtype=np.float32 verify_negative_samples=True calculate_training_loss=False`
  - LightFM
    - `loss=bpr max_sampled=1`
  - QMF
    - `num_negative_samples=1 eval_num_neg=0`
