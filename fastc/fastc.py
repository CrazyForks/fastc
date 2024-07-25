#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from huggingface_hub import hf_hub_download
from transformers import logging

from .classifiers.centroids import NearestCentroidClassifier
from .classifiers.embeddings import Pooling
from .classifiers.logistic_regression import LogisticRegressionClassifier
from .kernels import Kernels
from .template import Template

logging.set_verbosity_error()


class Fastc:
    def __new__(
        cls,
        model: str = None,
        embeddings_model: str = None,
        kernel: Kernels = None,
        model_type: Kernels = None,  # Backwards compatibility
        template: str = None,
        pooling: Pooling = None,
        **kwargs,
    ):
        model_data = None
        label_names_by_id = None

        # Backwards compatibility
        if kernel is None and model_type is not None:
            kernel = model_type

        if model is not None:
            config = cls._get_config(model)
            model_config = config['model']

            try:
                kernel = Kernels.from_value(model_config['kernel'])
            except KeyError:
                # Backwards compatibility
                kernel = Kernels.from_value(model_config['type'])

            model_data = model_config['data']
            embeddings_model = model_config['embeddings']

            labels = model_config.get('labels')
            if labels is None:
                # Backwards compatibility
                label_names_by_id = {label: label for label in model_data.keys()}  # noqa: E501
            else:
                label_names_by_id = {v: k for k, v in labels.items()}

            pooling = Pooling.from_value(model_config.get(
                'pooling',
                Pooling.MEAN.value,  # Backwards compatibility
            ))

            if 'template' in model_config:
                template_text = model_config['template']['text']
                template_variables = model_config['template']['variables']
                template = Template(template_text, **template_variables)

        if embeddings_model is None:
            embeddings_model = 'deepset/tinyroberta-6l-768d'

        if kernel is None:
            kernel = Kernels.LOGISTIC_REGRESSION

        if template is None:
            template = Template()

        if pooling is None:
            pooling = Pooling.DEFAULT

        classifier_kwargs = {
            'embeddings_model': embeddings_model,
            'model_data': model_data,
            'template': template,
            'pooling': pooling,
            'label_names_by_id': label_names_by_id,
            **kwargs,
        }

        if kernel == Kernels.LOGISTIC_REGRESSION:
            return LogisticRegressionClassifier(**classifier_kwargs)

        if kernel == Kernels.NEAREST_CENTROID:
            return NearestCentroidClassifier(**classifier_kwargs)

        # Backwards compatibility
        if kernel == Kernels.CENTROIDS:
            return NearestCentroidClassifier(**classifier_kwargs)

        raise ValueError("Unsupported model type {}".format(kernel))

    @staticmethod
    def _get_config(model: str):
        if os.path.isdir(model):
            file_path = os.path.join(model, 'config.json')
        elif os.path.isfile(model):
            file_path = model
        else:
            file_path = hf_hub_download(
                repo_id=model,
                filename='config.json'
            )
        with open(file_path, 'r') as model_file:
            model = json.load(model_file)
        return model


# Backwards compatibility
SentenceClassifier = Fastc
