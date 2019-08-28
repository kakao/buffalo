# Testing
Unit-test was created using python standard unittest library.

```bash
tests $> nosetests ./algo/test_algo.py -v
```

## External Resources
Some tests rely on external databases. Before run the tests, please download below database and place it properly.
  - MovieLens 100k: https://grouplens.org/datasets/movielens/100k/
    - Unarchiving it to ./ext/ml-100k
  - MovieLens 20m: https://grouplens.org/datasets/movielens/20m/
    - Unarchiving it to ./ext/ml-20m
  - Text8: http://mattmahoney.net/dc/text8.zip
    - Unarchving it to ./ext/text8
  - Question-words: https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt
    - Place it to ./ext/text8
  - KakaoBrunch12M: https://arena.kakao.com/datasets
    - Place it to ./ext/kakao-brunch-12m
  - KakaoReco730M: https://arena.kakao.com/datasets
    - Place it to ./ext/kakao-reco-730m

## Preprcessing
Before run the tests, pre-processing has to be done. Type `python preprocessing.py`.
