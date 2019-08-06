# Benchmark
We ran the benchmark with the Implicit library. The Implicit library provides ALS, BPRMF as like ours and was written in Python that easy to compare.

- Machine
  - CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (6 cores)
  - Ram: 64GB

- The number of latent feature vector dimension

| method   |     D=10 |     D=20 |    D=30 |    D=40 |    D=50 |    D=60 |    D=70 |    D=80 |     D=90 |   D=100 |    D=150 |   D=200 |
|----------|----------|----------|---------|---------|---------|---------|---------|---------|----------|---------|----------|---------|
| buffalo  |  4.52909 |  6.35531 | 10.7146 | 11.5622 | 16.2517 | 21.6558 | 27.6353 | 29.1399 |  35.1215 | 39.8375 |  78.1201 | 117.606 |
| implicit | 38.947   | 53.3205  | 70.2035 | 56.5195 | 76.7012 | 90.9826 | 74.1561 | 92.2424 | 105.24   | 85.6067 | 138.357  | 130.04  |


- The number of threads

| method   |     T=1 |     T=16 |     T=2 |     T=4 |     T=8 |
|----------|---------|----------|---------|---------|---------|
| buffalo  | 69.3051 |  9.52505 | 34.0029 | 18.8265 | 11.9599 |
| implicit | 90.0916 | 58.1543  | 68.9974 | 59.0597 | 58.3854 |

This showed that Buffalo library can better utilize the CPU.
