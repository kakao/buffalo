.. Buffalo documentation master file, created by
   sphinx-quickstart on Sat Aug 17 13:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Buffalo's documentation!
===================================

.. image:: ./buffalo.png
   :width: 320px 

Buffalo is a production-ready open source which fast and scalable. Buffalo maximizes performance by effectively using system resources even on low-spec machines. The implementation is optimized for CPU and SSD, but it also shows good performance on the GPU. Buffalo is one of the recommender system libraries developed by Kakao and has been reliably used for various Kakao services.

Buffalo provides the following algorithms:

  - Alternating Least Squares
  - Bayesian Personalized Ranking Matrix Factorization
  - Word2Vec
  - CoFactors
all algorithms are optimized for multi-threading and some algorithm supports GPU accelerators.
 
One of the good things about the library is its very low memory usage compared to other libraries. Chunked data management and batch learning with HDF5 can handle large data, even bigger than memory size, on laptop machine. Check out the benchmarks page for more details on Buffalo performance.

Besides, buffalo provides a variety of convenient features for research and production purposes such as tensorboard integration, hyper-parameter optimization and so on.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Quickstart <quickstart>
    Algorithms <algo>
    Parallels <parallels>
    Examples <example>
    Benchmark <benchmark>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
