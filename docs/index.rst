.. Buffalo documentation master file, created by
   sphinx-quickstart on Sat Aug 17 13:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Buffalo's documentation!
===================================

.. image:: ./buffalo.png
   :width: 320px 

Buffalo is a production-ready open source which fast and scalable. Buffalo maximizes performance by effectively using system resources even on low-spec machines. The implementation is optimized for CPU and SSD, but it also shows good performance with GPU accelerator. Buffalo is one of the recommender system libraries developed by Kakao and has been reliably used for various Kakao services.

Buffalo provides the following algorithms:

  - Alternating Least Squares [1]_ 
  - Bayesian Personalized Ranking Matrix Factorization [2]_
  - Word2Vec [3]_
  - CoFactors [4]_

all algorithms are optimized for multi-threading and some algorithm supports GPU accelerators.
 
One of the good things about the library is its very low memory usage compared to other libraries. Chunked data management and batch learning with HDF5 can handle large data, even bigger than memory size, on laptop machine. Check out the benchmarks page for more details on Buffalo performance.

Besides, buffalo provides a variety of convenient features for research and production purposes such as tensorboard integration, hyper-parameter optimization and so on.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Introduction <intro>
    Algorithms <algo>
    Parallels <parallels>
    Benchmarks <https://github.daumkakao.com/toros/buffalo/tree/dev/benchmark>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========
.. [1] Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
.. [2] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.
.. [3] Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.
.. [4] Liang, Dawen, et al. "Factorization meets the item embedding: Regularizing matrix factorization with item co-occurrence." Proceedings of the 10th ACM conference on recommender systems. ACM, 2016.
