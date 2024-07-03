#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, Generator, List, Tuple

from huggingface_hub import HfApi

from ..template import Template
from .embeddings import EmbeddingsModel

FASTC_FORMAT_VERSION = 2.0


class FastcInterface:
    def __init__(
        self,
        embeddings_model: str,
        template: Template = None,
    ):
        self._embeddings_model = EmbeddingsModel(embeddings_model)
        self._template = template
        self._texts_by_label = None

    def load_dataset(self, dataset: List[Tuple[str, int]]):
        if not isinstance(dataset, list):
            raise TypeError('Dataset must be a list of tuples.')

        texts_by_label = {}
        for text, label in dataset:
            if label not in texts_by_label:
                texts_by_label[label] = []
            texts_by_label[label].append(text)

        self._texts_by_label = texts_by_label

    @property
    def embeddings_model(self):
        return self._embeddings_model

    def train(self):
        raise NotImplementedError

    def predict_one(self, text: str) -> Dict[int, float]:
        raise NotImplementedError

    def predict(self, texts: List[str]) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        raise NotImplementedError

    def save_model(self, path: str):
        raise NotImplementedError

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

    def _get_info(self):
        return {
            'version': FASTC_FORMAT_VERSION,
            'model': {
                'embeddings': self._embeddings_model_name,
                'template': {
                    'text': self._template._template,
                    'variables': self._template._variables,
                },
            },
        }
