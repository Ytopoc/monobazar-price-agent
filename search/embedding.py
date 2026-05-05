# Wrapper around multilingual-e5-large embeddings.
from __future__ import annotations

import logging
import time
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(
            "Model loaded in %.1fs  (dim=%d, device=%s)",
            time.time() - t0,
            self.embedding_dim,
            self._model.device,
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def embed_passages(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        self._ensure_loaded()
        prefixed = [f"passage: {t}" for t in texts]
        logger.info("Encoding %d passages (batch_size=%d) ...", len(prefixed), self.batch_size)
        t0 = time.time()
        embeddings = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        logger.info("Encoded %d passages in %.1fs", len(texts), time.time() - t0)
        return embeddings.astype(np.float32)

    def embed_query(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        self._ensure_loaded()
        prefixed = f"query: {text}"
        vec = self._model.encode(
            [prefixed],
            batch_size=1,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()
