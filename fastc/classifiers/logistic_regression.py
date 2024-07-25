#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import io
import warnings
from typing import Dict, Generator, List, Union

import joblib
from scipy.stats import loguniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..kernels import Kernels
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
        cross_validation_splits: int = None,
        cross_validation_repeats: int = None,
        iterations: int = None,
        parameters: Union[Dict, List] = None,
        seed: int = None,
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

        if cross_validation_splits is None:
            cross_validation_splits = 5
        self._cross_validation_splits = cross_validation_splits

        if cross_validation_repeats is None:
            cross_validation_repeats = 3
        self._cross_validation_repeats = cross_validation_repeats

        if iterations is None:
            iterations = 100
        self._iterations = iterations

        self._parameters = parameters
        self._seed = seed

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

        common_params = {
            'C': loguniform(1e-6, 1e3),
            'max_iter': [3000],
            'tol': loguniform(1e-6, 1e-2),
            'class_weight': [None, 'balanced'],
            'warm_start': [True, False],
            'fit_intercept': [True, False],
            'intercept_scaling': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        }

        if self._seed is not None:
            common_params['random_state'] = [self._seed]

        compatible_params = [
            {
                'solver': ['liblinear'],
                'penalty': ['l1'],
                'dual': [False],
            },
            {
                'solver': ['liblinear'],
                'penalty': ['l2'],
                'dual': [True, False],
            },
            {
                'solver': ['saga'],
                'penalty': ['l1', 'l2'],
                'l1_ratio': loguniform(1e-2, 1)
            },
            {
                'solver': ['saga'],
                'penalty': ['elasticnet'],
                'l1_ratio': loguniform(1e-2, 1)
            },
            {
                'solver': ['saga'],
                'penalty': [None],
            },
            {
                'solver': ['newton-cg'],
                'penalty': ['l2'],
            },
            {
                'solver': ['newton-cg'],
                'penalty': [None],
            },
            {
                'solver': ['lbfgs'],
                'penalty': ['l2'],
            },
            {
                'solver': ['lbfgs'],
                'penalty': [None],
            },
            {
                'solver': ['sag'],
                'penalty': ['l2'],
            },
            {
                'solver': ['sag'],
                'penalty': [None],
            },
        ]

        def prefix_params(params: Dict):
            return {
                'logisticregression__' + key: value
                for key, value in params.items()
            }

        if self._parameters is None:
            parameters = []
            for item in compatible_params:
                new_item = item | common_params

                if new_item['penalty'] == [None]:
                    del new_item['C']

                if new_item['penalty'] != ['elasticnet']:
                    if 'l1_ratio' in new_item:
                        del new_item['l1_ratio']

                parameters.append(prefix_params(new_item))

        random_search = RandomizedSearchCV(
            self._lr_pipeline,
            param_distributions=parameters,
            n_iter=self._iterations,
            cv=RepeatedStratifiedKFold(
                n_splits=self._cross_validation_splits,
                n_repeats=self._cross_validation_repeats,
                random_state=self._seed,
            ),
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=self._seed,
        )

        loader = Loader("Training")
        loader.start()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
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
        info['model']['kernel'] = Kernels.LOGISTIC_REGRESSION.value

        buffer = io.BytesIO()
        joblib.dump(self._model, buffer, protocol=5)
        buffer.seek(0)
        model_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        info['model']['data'] = model_base64
        return info
