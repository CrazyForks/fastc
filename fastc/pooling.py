#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum


class Pooling(Enum):
    MEAN = 'mean'
    MEAN_MASKED = 'mean-masked'
    MAX = 'max'
    MAX_MASKED = 'max-masked'
    CLS = 'cls'
    SUM = 'sum'
    ATTENTION_WEIGHTED = 'attention-weighted'
    DEFAULT = MEAN

    @classmethod
    def from_value(cls, value: str) -> 'Pooling':
        try:
            return cls._value2member_map_[value]
        except KeyError:
            raise ValueError(f"{value} is not a valid pooling value")


ATTENTION_POOLING_STRATEGIES = set([
    Pooling.ATTENTION_WEIGHTED,
])
