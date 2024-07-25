#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum


class Kernels(Enum):
    NEAREST_CENTROID = 'nearest-centroid'
    LOGISTIC_REGRESSION = 'logistic-regression'

    # Backwards compatibility
    CENTROIDS = 'centroids'

    @classmethod
    def from_value(cls, value: str) -> 'Kernels':
        try:
            return cls._value2member_map_[value]
        except KeyError:
            raise ValueError(f"{value} is not a valid model type value")
