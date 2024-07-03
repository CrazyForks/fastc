#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generator, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class EmbeddingsModel:
    _instances = {}

    def __new__(cls, model_name):
        if model_name not in cls._instances:
            instance = super(EmbeddingsModel, cls).__new__(cls)
            cls._instances[model_name] = instance
            instance._initialized = False
        return cls._instances[model_name]

    def __init__(self, model_name):
        if not self._initialized:
            self.model_name = model_name
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
            self._initialized = True

    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
        title: Optional[str] = None,
        show_progress: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        for text in tqdm(
            texts,
            desc=title,
            unit='text',
            disable=not show_progress,
        ):
            inputs = self._tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            outputs = self._model(**inputs)

            token_embeddings = outputs.last_hidden_state[0]
            sentence_embedding = torch.mean(
                token_embeddings,
                dim=0,
            )
            yield sentence_embedding
