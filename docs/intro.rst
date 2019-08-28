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
We highly recommend starting with the unit-test codes. Checkout ./tests directory, `./tests/algo/test_algo.py` will be a good starting point.

.. code-block:: bash

    $ tests> nosetests ./tests/algo/test_algo.py -v


Database
--------
We call term `database` as a data file format used by the buffalo internally. Buffalo take data that the Matrix Market or Stream format as input and converts it into a database class which store rawdata using h5py(http://www.h5py.org). The main reason to make custom database is to use the least amount of memory without compromising capacity of data volumn and learning speed.

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


Hyper paremter Optimization
---------------------------
The Algo classes inherited Optimizable class which is helper class to provide hyper parameter optimization. Basically it depends on hyperopt(http://hyperopt.github.io/hyperopt/), well known library, include all of the capabilities.

The option of the optimization should be stored in optimize field in Algo option. The following is the description of the option. You can check practical example on the unittests.

- loss(str): Target loss to optimize.
- max_trials(int, option): Maximum experiments for optimization. If not given, run forever.
- min_trials(int, option): Minimum experiments before deploying model. (Since the best parameter may not be found after min_trials, the first best parameter is always deployed)
- deployment(bool): Set True to train model with the best parameter. During the optimization, it try to dump the model which beated the previous best loss.
- start_with_default_parameters(bool): If set to True, the loss value of the default parameter is used as the starting loss to beat.
- space(dict): Parameter space definition. For more information, pleases reference hyperopt's express.

  - Note) Due to hyperopt's randint does not provide lower value, we had to implement it a bait tricky. Pleases see optimize.py to check how we deal with randint.


Logging
-------
It is recommend to use the log library of buffalo for consistent log format.

.. code-block:: python

    >>> from aurochs.misc import log
    >>> print(log.NOTSET, log.WARN, log.INFO, log.DEBUG, log.TRACE)
    (0, 1, 2, 3, 4, 5)
    >>> log.set_log_level(log.WARN)  # this set log-level on Python, C++ both sides.
    >>> log.get_log_level()
    1
    >>> 

    >>> from aurochs.misc import log, aux
    >>> logger = aux.get_logger()
    >>> with log.pbar(logger.debug, desc='Test', mininterval=1):
        for(i in range(100)):
            time.sleep(0.1)

`log.pbar` is a wrapper class of tqdm(https://tqdm.github.io), except it use Python Logger for logging instead sys.stdout(see first argument).
