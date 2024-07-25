#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Generator, List

import torch
import torch.nn.functional as F

from ..kernels import Kernels
from ..template import Template
from .embeddings import Pooling
from .interface import ClassifierInterface


class NearestCentroidClassifier(ClassifierInterface):
    def __init__(
        self,
        embeddings_model: str,
        template: Template,
        pooling: Pooling,
        label_names_by_id: Dict[int, str],
        model_data: Dict[int, List[float]] = None,
    ):
        super().__init__(
            embeddings_model=embeddings_model,
            template=template,
            pooling=pooling,
            label_names_by_id=label_names_by_id,
        )

        self._centroids = {}
        self._normalized_centroids = {}

        if model_data is not None:
            self._load_model(model_data)

    def _load_model(self, centroids: Dict):
        self._centroids = {
            self._label_ids_by_name[label]: torch.tensor(centroid)
            for label, centroid in centroids.items()
        }
        self._normalized_centroids = {
            label: self._normalize(centroid)
            for label, centroid in self._centroids.items()
        }

    def train(self):
        if self._texts_by_label is None:
            raise ValueError("Dataset is not loaded.")

        for label, texts in self._texts_by_label.items():
            texts = [self._template.format(text) for text in texts]
            embeddings = list(self.embeddings_model.get_embeddings(
                texts=texts,
                pooling=self._pooling,
                title='Generating embeddings [{}]'.format(
                    self._label_names_by_id[label],
                ),
                show_progress=True,
            ))
            embeddings = torch.stack(embeddings)
            centroid = torch.mean(embeddings, dim=0)
            self._centroids[label] = centroid
            self._normalized_centroids[label] = self._normalize(centroid)

    def predict(
        self,
        texts: List[str],
    ) -> Generator[Dict[int, float], None, None]:
        if self._normalized_centroids is None:
            raise ValueError("Model is not trained.")

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        texts = [self._template.format(text) for text in texts]
        embeddings = self.embeddings_model.get_embeddings(
            texts=texts,
            pooling=self._pooling,
        )

        normalized_embeddings = [
            self._normalize(embedding)
            for embedding in embeddings
        ]

        for text_embedding in normalized_embeddings:
            dot_products = {
                label: torch.dot(text_embedding, centroid).item()
                for label, centroid in self._normalized_centroids.items()
            }

            dot_product_values = torch.tensor(list(dot_products.values()))
            softmax_scores = F.softmax(dot_product_values, dim=0).tolist()

            scores = {
                self._label_names_by_id[label]: score
                for label, score in zip(dot_products.keys(), softmax_scores)
            }

            result = {
                'label': max(scores, key=scores.get),
                'scores': scores,
            }

            yield result

    def _get_info(self):
        info = super()._get_info()
        info['model']['kernel'] = Kernels.NEAREST_CENTROID.value
        info['model']['data'] = {
            self._label_names_by_id[key]: value.tolist()
            for key, value in self._centroids.items()
        }
        return info
