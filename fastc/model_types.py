#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum


class ModelTypes(Enum):
    CENTROIDS = 'centroids'
    LOGISTIC_REGRESSION = 'logistic-regression'

    @classmethod
    def from_value(cls, value: str) -> 'ModelTypes':
        try:
            return cls._value2member_map_[value]
        except KeyError:
            raise ValueError(f"{value} is not a valid model type value")
