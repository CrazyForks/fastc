#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generator, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..pooling import Pooling


class EmbeddingsModel:
    _instances = {}

    def __new__(
        cls,
        model_name,
        output_attentions,
    ):
        if (
            model_name not in cls._instances
            or (
                output_attentions
                and not cls._instances[model_name]._output_attentions
            )
        ):
            instance = super(EmbeddingsModel, cls).__new__(cls)
            cls._instances[model_name] = instance
            instance._initialized = False

        return cls._instances[model_name]

    def __init__(
        self,
        model_name,
        output_attentions,
    ):
        if not self._initialized:
            self.model_name = model_name
            self._output_attentions = output_attentions
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(
                model_name,
                output_attentions=output_attentions,
            )
            self._model.eval()
            self._initialized = True

    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
        pooling: Pooling,
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
            attention_mask = inputs['attention_mask']

            if pooling == Pooling.MEAN:
                sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
                yield from sentence_embeddings

            elif pooling == Pooling.MEAN_MASKED:
                last_hidden_state = outputs.last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(),
                    0.0,
                )
                sentence_embeddings = (
                    last_hidden_state.sum(dim=1)
                    / attention_mask.sum(dim=1)
                    [..., None]
                )
                yield from sentence_embeddings

            elif pooling == Pooling.ATTENTION_WEIGHTED:
                attention_weights = outputs.attentions[-1]
                attention_weights = attention_weights.masked_fill(
                    ~attention_mask[:, None, None, :].bool(),
                    0.0,
                )
                attention_weights = attention_weights.mean(dim=1)
                attention_weights = F.normalize(attention_weights, p=1, dim=-1)
                sentence_embeddings = torch.einsum(
                    'bsl,bsd->bsd',
                    attention_weights,
                    outputs.last_hidden_state,
                ).sum(dim=1)
                yield from sentence_embeddings

            elif pooling == Pooling.CLS:
                sentence_embeddings = outputs.last_hidden_state[:, 0]
                yield from sentence_embeddings

            elif pooling == Pooling.MAX:
                sentence_embeddings = torch.max(
                    outputs.last_hidden_state,
                    dim=1,
                )[0]
                yield from sentence_embeddings

            elif pooling == Pooling.MAX_MASKED:
                last_hidden_state = outputs.last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(),
                    float('-inf'),
                )
                sentence_embeddings = torch.max(last_hidden_state, dim=1)[0]
                yield from sentence_embeddings

            elif pooling == Pooling.SUM:
                sentence_embeddings = outputs.last_hidden_state.sum(dim=1)
                yield from sentence_embeddings

            else:
                raise ValueError("Unsupported pooling strategy.")
