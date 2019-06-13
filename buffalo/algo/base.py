# -*- coding: utf-8 -*-
import abc


class Algo(abc.ABC):
    def __init__(self, *argv, **kwargs):
        pass


class AlgoOption(abc.ABC):
    def __init__(self, *argv, **kwargs):
        pass

    @abc.abstractmethod
    def get_default_option(self) -> dict:
        pass

    @abc.abstractmethod
    def is_valid_option(self, opt) -> bool:
        pass
