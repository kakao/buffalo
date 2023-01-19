.. Buffalo documentation master file, created by
   sphinx-quickstart on Sat Aug 17 13:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Buffalo's documentation!
===================================

.. image:: ./buffalo.png
   :width: 320px

Buffalo is a fast and scalable production-ready open source project for recommendation systems. Buffalo effectively utilizes system resources, enabling high performance even on low-spec machines. The implementation is optimized for CPU and SSD. Even so, it shows good performance with GPU accelerator, too. Buffalo, developed by Kakao, has been reliably used in production for various Kakao services.

Buffalo provides the following algorithms:

  - Alternating Least Squares (ALS) [1]_
  - Bayesian Personalized Ranking Matrix Factorization (BPR) [2]_
  - Word2Vec (W2V) [3]_
  - CoFactors (CFR) [4]_
  - Probabilistic Latent Semantic Indexing (PLSI) [5]_
  - Colalborative Metric Learning (WARP) [6]_

All algorithms are optimized for multi-threading and ALS, and BPR support GPU accelerators.

One of the best things about this library is a very low memory usage compared to other competing libraries. With chunked data management and batch learning with HDF5, handling a large-scale data, even bigger than memory size on laptop machine, is made possible. Check out the benchmarks page(https://github.com/kakao/buffalo/tree/master/benchmark) for more details on Buffalo performance.

Plus, Buffalo provides a variety of convenient features for research and production purposes.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Introduction <intro>
    Algorithms <algo>
    Parallels <parallels>
    Benchmarks <https://github.com/kakao/buffalo/tree/master/benchmark>


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
.. [5] Hofmann, Thomas. "Learning the similarity of documents: An information-geometric approach to document retrieval and categorization." Advances in neural information processing systems 12 (1999).
.. [6] Hsieh, Cheng-Kang, et al. "Collaborative metric learning." Proceedings of the 26th international conference on world wide web. 2017.