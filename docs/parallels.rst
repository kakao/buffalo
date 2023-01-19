Parallels
==========
Provides parallel processing feature to algorithm classes.

It is written in C++/OpenMP to maximize CPU utilization. Even with a single thread, it works faster than the default implementation of Algo classes. Parallels also provides a boosting feature to execute `most_similar` function, which is based on approximate nearest neighbors library N2. For performance and examples usage, please refer to the benchmark page and unit test codes.


.. autoclass:: buffalo.parallel.base.Parallel
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.ParALS
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.ParBPRMF
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.ParW2V
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.ParCFR
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:
