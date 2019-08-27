Algorithms
===============


Algo
---------
.. autoclass:: buffalo.algo.base.Algo
   :members: 
   :exclude-members: get_option, periodical, save_best_only, early_stopping
   :show-inheritance:
   :undoc-members:

Serializable
`````````````
.. autoclass:: buffalo.algo.base.Serializable
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

TensorboardExtention
````````````````````
.. autoclass:: buffalo.algo.base.TensorboardExtention
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Optimizable
````````````
.. autoclass:: buffalo.algo.optimize.Optimizable
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Evaluable
`````````
.. autoclass:: buffalo.evaluate.base.Evaluable
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Alternating Least Squares
-------------------------
.. autoclass:: buffalo.algo.als.ALS
   :members: 
   :exclude-members: get_evaluation_metrics, init_factors, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.algo.options.ALSOption
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Bayesian Personalized Ranking Matrix Factorization
--------------------------------------------------
.. autoclass:: buffalo.algo.bpr.BPRMF
   :members: 
   :exclude-members: get_evaluation_metrics, init_factors, set_data, compute_loss, prepare_sampling, sampling_loss_samples, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.algo.options.BPRMFOption
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:

CoFactors
---------
.. autoclass:: buffalo.algo.cfr.CFR
   :members: 
   :exclude-members: compute_scale, get_evaluation_metrics, partial_update, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.algo.options.CFROption
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:


Word2Vec
--------
.. autoclass:: buffalo.algo.w2v.W2V
   :members: 
   :exclude-members: get_evaluation_metrics, build_vocab, get_sampling_distribution, init_factors, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.algo.options.W2VOption
   :members: 
   :exclude-members:
   :show-inheritance:
   :undoc-members:
