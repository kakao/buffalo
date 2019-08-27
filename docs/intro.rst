Introduction
============

Buffalo is a production-ready open source which fast and scalable. Buffalo maximizes performance by effectively using system resources even on low-spec machines. The implementation is optimized for CPU and SSD, but it also shows good performance with GPU accelerator. Buffalo is one of the recommender system libraries developed by Kakao and has been reliably used for various Kakao services.

Buffalo provides the following algorithms:

  - Alternating Least Squares [1]_ 
  - Bayesian Personalized Ranking Matrix Factorization [2]_
  - Word2Vec [3]_
  - CoFactors [4]_

all algorithms are optimized for multi-threading and some algorithm supports GPU accelerators.
 
One of the good things about the library is its very low memory usage compared to other libraries. Chunked data management and batch learning with HDF5 can handle large data, even bigger than memory size, on laptop machine. Check out the benchmarks page for more details on Buffalo performance.

Besides, buffalo provides a variety of convenient features for research and production purposes such as tensorboard integration, hyper-parameter optimization and so on.


Installation
------------
Type `pip install buffalo`.


Basic Usage
-----------
We highly recommend starting with the unittests. Checkout ./tests directory, `./tests/algo/test_algo.py` will be a good starting point.


Database
--------
First, `database`. we call `database` in term of the data file format used by the buffalo internally. Buffalo take data that the Matrix Market or Stream format as input and converts it into a database class which store rawdata using h5py(http://www.h5py.org). The main reason to make custom database is to use the least amount of memory without compromising data size and learning speed.

The Stream data format consists of two files:

  - main 

    - Assumed that the data is reading history of users from some blog service, then each line is a reading history corresponding to each row of UID files. (i.e. users lists)
    - The reading history is sererated by spaces, and the past is the left and the right is the most recent history.
    - e.g. `A B C D D E` means that a user read the contents in the order A B C D D E.

  - uid

    - Each line is repersentational value of a user corresponding to each row in the MAIN file (e.g. user name)
    - Do not allow spaces in each line

For Matrix Market format, please refer to https://math.nist.gov/MatrixMarket/formats.html.

  - main

    - Matrix Market data file.

  - uid

    - Each line is the actual userkey corresponding to the row id in the MM file.

  - iid

    - Each line is the actual itemkey corresponding to the colum id in the MM file.

uid and iid are the data needed to provide human readable results only, not required.
