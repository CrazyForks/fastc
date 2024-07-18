#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import io
import warnings
from typing import Dict, Generator, List

import joblib
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..model_types import ModelTypes
from ..template import Template
from .embeddings import Pooling
from .interface import ClassifierInterface
from .loader import Loader


class LogisticRegressionClassifier(ClassifierInterface):
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

        self._lr_pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression()
        )

        if model_data is None:
            return

        self._load_model(model_data)

    def _load_model(
        self,
        model_data: str,
    ):
        buffer = io.BytesIO(base64.b64decode(model_data))
        self._model = joblib.load(buffer)

    def train(self):
        X = []
        y = []

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

            normalized_embeddings = [
                self._normalize(embedding)
                for embedding in embeddings
            ]

            X.extend(normalized_embeddings)
            y.extend([label] * len(texts))

        param_distributions = {
            'logisticregression__C': loguniform(1e-5, 1e5),
            'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # noqa: E501
            'logisticregression__max_iter': [100, 200, 500, 1000, 2000, 5000, 10000],  # noqa: E501
            'logisticregression__tol': loguniform(1e-6, 1e-2),
            'logisticregression__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'logisticregression__class_weight': ['balanced', None],
            'logisticregression__warm_start': [True, False],
            'logisticregression__l1_ratio': uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            self._lr_pipeline,
            param_distributions=param_distributions,
            n_iter=128,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=0,
        )

        loader = Loader("Training")
        loader.start()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random_search.fit(X, y)

        loader.stop()

        self._model = random_search.best_estimator_

    def predict(
        self,
        texts: List[str],
    ) -> Generator[Dict[int, float], None, None]:
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
            probabilities = self._model.predict_proba([text_embedding])[0]

            scores = {
                self._label_names_by_id[label]: probability
                for label, probability in enumerate(probabilities)
            }

            result = {
                'label': max(scores, key=scores.get),
                'scores': scores,
            }

            yield result

    def _get_info(self):
        info = super()._get_info()
        info['model']['type'] = ModelTypes.LOGISTIC_REGRESSION.value

        buffer = io.BytesIO()
        joblib.dump(self._model, buffer, protocol=5)
        buffer.seek(0)
        model_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        info['model']['data'] = model_base64
        return info
