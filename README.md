<center><img src="https://github.daumkakao.com/storage/user/993/files/61384d80-a48b-11e9-8d12-0149a6a0fa37" width="320px"></center>


# Buffalo
A Matrix Factorization Library

```python
$> cd tests && python36
>>> from buffalo.misc import aux
>>> from buffalo.algo.als import ALS
>>> from buffalo.algo.options import AlsOption
>>> from buffalo.data.mm import MatrixMarketOptions
>>> option = AlsOption().get_default_option()
>>> option.validation = aux.Option({'topk': 10})  # set topk param for validation scores
>>> data_opt = MatrixMarketOptions().get_default_option()
>>> data_opt.input.main = './ml-100k/main'
>>> als = ALS(option, data_opt=data_opt)
>>> als.init()
>>> als.train()
>>> results = als.get_validation_results()
>>> print(resulsts)
{'ndcg': 0.03247253115122813, 'map': 0.021315653632726805, 'accuracy': 0.06377032520325204, 'rmse': 2.9231147330905136, 'error': 2.7129669839143755}
```

# Installation
## Building from the source
To install buffalo, run:
```
$> python setup.py install
```

## Requirements
- Python 3.6+
- cmake 2.8.8+
- gcc/g++(need C++14)
