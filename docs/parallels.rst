Parallels
==========
Provides parallel processing feature to algorithm classes.

It was written by C++/OpenMP to maximize CPU utilization. It works faster even with a single thread than default implementation of Algo classes. Parallels also provide a boosting feature to execute `most_similar` function which based on approximate nearest neighbors library N2. For the performance and examples usage of Parallels, please refer to the benchmark page and unit test codes.


.. autoclass:: buffalo.parallel.base.Parallel
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.parallel.base.ParALS
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.parallel.base.ParBPRMF
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.parallel.base.ParW2V
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.parallel.base.ParCFR
   :members: most_similar, topk_recommendation
   :exclude-members:
   :show-inheritance:
   :undoc-members:
