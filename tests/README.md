# Testing
Unit-test was created using python standard unittest library. (don't forget to run preprocessing.py)

```bash
buffalo.git/tests $> nosetests ./algo/test_algo.py -v  # see the current directory
```

## External Resources
Some tests rely on external databases. Before run the tests, please download below database and place it properly.
  - MovieLens 100k: https://grouplens.org/datasets/movielens/100k/
    - Download it to ./ext/ directory, then unzip it.
  - MovieLens 20m: https://grouplens.org/datasets/movielens/20m/
    - Download it to ./ext/ directory, then unzip it.
  - Text8: http://mattmahoney.net/dc/text8.zip
    - Download it to ./ext/text8 directory, then unzip it.
  - Question-words: https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt
    - Download it to ./ext/text8 directory
  - KakaoBrunch12M: https://arena.kakao.com/datasets?id=1
    - Place it to ./ext/kakao-brunch-12m (optional, only for benchmark)
  - KakaoReco730M: https://arena.kakao.com/datasets?id=2
    - Place it to ./ext/kakao-reco-730m (optional, only for benchmark)

## Preprcessing
Before run the tests, pre-processing has to be done. Type `python preprocessing.py`.
