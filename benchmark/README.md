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
  - KakaoReco730
    - Sampled user reading history from Kakao annoymous service during 8 days.
  - KakaoBrunch12M
    - Sampled user reading history from [Kakao Brunch](https://brunch.co.kr) service, which kind of blog service, during 6 months.
  - Movielens20M
    - https://grouplens.org/datasets/movielens/


## Database Statistics
  | # USERS | # ITEMS | # NNZ
-- | -- | -- | --
MovieLens20M | 138,000 | 27,000 | 20M
KakaoBrunch12M | 306,291 | 505,926 | 12M
KakaoReco730M | 21,940,315 | 1,467,298 | 730M

* M stands for Million.
* NNZ stands for Number of Non-Zero entries.


## Alternating Least Square

- Fixed options (otherwise we let them as default except controlled options)
  - Buffalo
    - `optimizer=manual_cg num_cg_max_iters=3 compute_loss_on_training=False`
  - Buffalo-gpu
    - Same as buffalo
  - Implicit
    - `dtype=np.float32 use_cg=True use_native=True calculate_training_loss=False`
  - Implicit
    - Same as implicit
  - Spark
    - `implicitPrefs=True intermediateStroageLevel=MEMORY_ONLY finalStorageLevel=MEMORY_ONLY`
  - QMF
    - `asis`

Note that there is no Python version of QMF. Since we ran benchmark by Python script, we have to capture printed datetime information from standard output of QMF.

There is a restriction where the number of the latent dimensions must be 32 times when using GPU for implicit. For example, 80 demensions has been upscaled to 96 but not for 160. Therefore, it is not an accurate comparison between implicit-gpu and buffalo-gpu.

### KakaoReco730M
The biggest one and only buffalo and implicit can handle this with the system resource in tolerable time. Even implicit did not run with GPU accelerator mode due to lack of memory of GPU card. For buffalo-gpu, the memory management option `batch_mb` also worked consistently in GPU accelerator mode, allowing it to work with KakaoReco730M that data size cannot fit in memory.
fig. KakaoReco730M

- In this experiment, we set number of iteration to 2. 

### KakaoBrunch12M

method | D=10 | D=20 | D=40 | D=80 | D=160
-- | -- | -- | -- | -- | --
buffalo | 7.42076 | 9.37377 | 14.5747 | 37.2637 | 122.594
implicit | 82.8058 | 123.175 | 197.386 | 290.844 | 408.721
qmf | 63.5593 | 120.24 | 267.801 | 571.326 | 1442.09
pyspark | 55.7363 | 79.9613 | 148.842 | 485.194 | 2349.76
Buffalo-gpu | 4.08037 | 4.01108 | 5.20191 | 6.2576 | 9.16236
implicit-gpu | 5.66232 | 4.88898 | 7.8817 | 8.05288 | 11.3726

method | T=1 | T=2 | T=4 | T=8 | T=16
-- | -- | -- | -- | -- | --
buffalo | 57.4951 | 28.7919 | 15.8516 | 9.41654 | 6.14486
implicit | 212.646 | 156.561 | 128.528 | 122.587 | 125.323
qmf | 201.709 | 113.166 | 73.3526 | 124.546 | 144.251
pyspark | 370.907 | 193.428 | 116.088 | 77.8977 | 55.7786


### Movielens20M


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

Implicit also provides GPU accelerator mode for BPRMF, but buffalo didn't. Implicit-gpu run much faster than buffalo. We plan to add GPU accelerator feature on BPRMF in the near future, so we will update the benchmarks afterwards.

## KakaoBrunch12M

## Movielens20M
