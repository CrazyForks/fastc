#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, Generator, List, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi

from ..pooling import ATTENTION_POOLING_STRATEGIES
from ..template import Template
from .embeddings import EmbeddingsModel, Pooling

FASTC_FORMAT_VERSION = 2.0


class ClassifierInterface:
    def __init__(
        self,
        embeddings_model: str,
        template: Template,
        pooling: Pooling,
        label_names_by_id: Dict[int, str],
    ):
        output_attentions = False
        if pooling in ATTENTION_POOLING_STRATEGIES:
            output_attentions = True
        self._embeddings_model = EmbeddingsModel(
            embeddings_model,
            output_attentions=output_attentions,
        )
        self._template = template
        self._pooling = pooling
        self._texts_by_label = None
        self._label_names_by_id = label_names_by_id

        self._label_ids_by_name = None
        if label_names_by_id is not None:
            self._label_ids_by_name = {
                v: k for k, v in label_names_by_id.items()
            }

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=2, dim=-1)

    def load_dataset(self, dataset: List[Tuple[str, str]]):
        if not isinstance(dataset, list):
            raise TypeError('Dataset must be a list of tuples.')

        texts_by_label = {}
        label_names_by_id = {}
        label_index = 0

        for text, label in dataset:
            if label not in label_names_by_id:
                label_names_by_id[label] = label_index
                label_index += 1

            label_id = label_names_by_id[label]

            if label_id not in texts_by_label:
                texts_by_label[label_id] = []

            texts_by_label[label_id].append(text)

        self._texts_by_label = texts_by_label
        self._label_names_by_id = {v: k for k, v in label_names_by_id.items()}

    @property
    def embeddings_model(self):
        return self._embeddings_model

    def train(self):
        raise NotImplementedError

    def predict(self, texts: List[str]) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        raise NotImplementedError

    def predict_one(self, text: str) -> Dict[int, float]:
        return list(self.predict([text]))[0]

    def _get_info(self):
        return {
            'version': FASTC_FORMAT_VERSION,
            'model': {
                'embeddings': self._embeddings_model_name,
                'pooling': self._pooling.value,
                'template': {
                    'text': self._template._template,
                    'variables': self._template._variables,
                },
                'labels': {v: k for k, v in self._label_names_by_id.items()},
            },
        }

    def save_model(
        self,
        path: str,
        description: str = None,
    ):
        os.makedirs(path, exist_ok=True)

        model_info = self._get_info()
        if description is not None:
            model_info['description'] = description

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(
                model_info,
                f,
                indent=4,
                ensure_ascii=False,
            )

    @property
    def _embeddings_model_name(self):
        return self._embeddings_model._model.name_or_path

    def push_to_hub(
        self,
        repo_id: str,
        tags: List[str] = None,
        languages: List[str] = None,
        private: bool = False,
    ):
        if tags is None:
            tags = []
        tags = ['fastc', 'fastc-{}'.format(FASTC_FORMAT_VERSION)] + tags

        self.save_model('/tmp/fastc')

        api = HfApi()

        api.create_repo(
            repo_id=repo_id,
            repo_type='model',
            private=private,
            exist_ok=True,
        )

        readme = (
            '---\n'
            'base_model: {}\n'
        ).format(self._embeddings_model_name)

        if languages is not None:
            readme += 'language:\n'
            for language in languages:
                readme += '- {}\n'.format(language)

        readme += 'tags:\n'
        for tag in tags:
            readme += '- {}\n'.format(tag)

        readme += '---\n\n'

        repo_name = repo_id.split('/')[1]
        readme += (
            '# {}\n\n'
            '## Install fastc\n'
            '```bash\npip install fastc\n```\n\n'
            '## Model Inference\n'
            '```python\n'
            'from fastc import Fastc\n\n'
            'model = Fastc(\'{}\')\n'
            'label = model.predict_one(text)[\'label\']\n'
            '```'
        ).format(repo_name, repo_id)

        readme_path = '/tmp/fastc/README.md'
        model_path = '/tmp/fastc/config.json'

        with open(readme_path, 'w') as readme_file:
            readme_file.write(readme)

        for file_path in [readme_path, model_path]:
            base_name = os.path.basename(file_path)
            api.upload_file(
                repo_id=repo_id,
                repo_type='model',
                path_or_fileobj=file_path,
                path_in_repo=base_name,
                commit_message='Updated {} via fastc'.format(base_name),
            )
            os.remove(file_path)
