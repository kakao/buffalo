Algorithms
==========
Buffalo provides the following algorithm implementations:

  - Alternating Least Squares
  - Bayesian Personalized Ranking Matrix Factorization
  - Weighted Approximate-Rank Pairwise
  - Word2Vec
  - CoFactors

All algorithms inherit common parent classes such as Algo, Serializable, Evaluable.


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

Evaluable
`````````
.. autoclass:: buffalo.evaluate.base.Evaluable
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:
   
Alternating Least Squares
-------------------------
.. autoclass:: buffalo.ALS
   :members:
   :exclude-members: get_evaluation_metrics, init_factors, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.ALSOption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Bayesian Personalized Ranking Matrix Factorization
--------------------------------------------------
.. autoclass:: buffalo.BPRMF
   :members:
   :exclude-members: get_evaluation_metrics, init_factors, set_data, compute_loss, prepare_sampling, sampling_loss_samples, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.BPRMFOption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:

Weighted Approximate-Rank Pairwise
--------------------------------------------------
.. autoclass:: buffalo.WARP
   :members:
   :exclude-members: get_evaluation_metrics, init_factors, set_data, compute_loss, prepare_sampling, sampling_loss_samples, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.WARPOption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:


CoFactors
---------
.. autoclass:: buffalo.CFR
   :members:
   :exclude-members: compute_scale, get_evaluation_metrics, partial_update, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.CFROption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:


Word2Vec
--------
.. autoclass:: buffalo.W2V
   :members:
   :exclude-members: get_evaluation_metrics, build_vocab, get_sampling_distribution, init_factors, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.W2VOption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:


pLSI
--------------------------------------------------
.. autoclass:: buffalo.PLSI
   :members:
   :exclude-members: get_evaluation_metrics, init_factors, set_data
   :show-inheritance:
   :undoc-members:

.. autoclass:: buffalo.PLSIOption
   :members:
   :exclude-members:
   :show-inheritance:
   :undoc-members:
