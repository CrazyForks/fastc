#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
from os import environ as env

from .fastc import Fastc  # noqa: F401
from .fastc import Kernels  # noqa: F401
from .fastc import Pooling, SentenceClassifier  # noqa: F401
from .template import ModelTemplates, Template  # noqa: F401

# Backwards compatibility
ModelTypes = Kernels

env['TOKENIZERS_PARALLELISM'] = 'true'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
