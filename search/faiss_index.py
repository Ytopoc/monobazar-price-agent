# Per-category FAISS indexes for comparable lookup.
from __future__ import annotations

import logging
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from config.category_config import CATEGORY_DICT

logger = logging.getLogger(__name__)


def _has_non_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _safe_read_index(index_file: Path) -> faiss.Index:
    if not _has_non_ascii(index_file):
        return faiss.read_index(str(index_file))

    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(str(index_file), tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        os.unlink(tmp_path)


def _safe_write_index(index: faiss.Index, index_file: Path) -> None:
    if not _has_non_ascii(index_file):
        faiss.write_index(index, str(index_file))
        return

    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        faiss.write_index(index, tmp_path)
        shutil.copy2(tmp_path, str(index_file))
    finally:
        os.unlink(tmp_path)


class CategoryFAISSIndex:
    def __init__(self, category_id: int) -> None:
        self.category_id = category_id
        self.category_name = CATEGORY_DICT.get(category_id, str(category_id))
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                f"must have the same length"
            )
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._metadata = list(metadata)
        logger.info(
            "Built FAISS index for category %d (%s): %d vectors, dim=%d",
            self.category_id, self.category_name, self._index.ntotal, dim,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 30) -> List[Dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Index for category %d is empty", self.category_id)
            return []

        k = min(top_k, self._index.ntotal)
        qv = query_vector.astype(np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        distances, indices = self._index.search(qv, k)

        results: List[Dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            entry = dict(self._metadata[idx])
            entry["cosine_similarity"] = float(dist)
            entry["rank"] = rank
            results.append(entry)
        return results

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        if self._index is None:
            raise RuntimeError("Index not built yet - call build() first")
        v = vector.astype(np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        self._index.add(v)
        self._metadata.append(metadata)

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("Nothing to save - index not built")
        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)

        index_file = dirpath / f"{self.category_id}.faiss"
        meta_file = dirpath / f"{self.category_id}_meta.pkl"

        _safe_write_index(self._index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Saved index for category %d: %s (%d vectors), %s",
            self.category_id, index_file, self._index.ntotal, meta_file,
        )

    def load(self, path: str | Path) -> None:
        dirpath = Path(path)
        index_file = dirpath / f"{self.category_id}.faiss"
        meta_file = dirpath / f"{self.category_id}_meta.pkl"

        self._index = _safe_read_index(index_file)
        with open(meta_file, "rb") as f:
            self._metadata = pickle.load(f)

        logger.info(
            "Loaded index for category %d: %d vectors, %d metadata entries",
            self.category_id, self._index.ntotal, len(self._metadata),
        )

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"CategoryFAISSIndex(category_id={self.category_id}, "
            f"size={self.size})"
        )


class FAISSIndexManager:
    def __init__(self) -> None:
        self._indices: Dict[int, CategoryFAISSIndex] = {}

    def build_all(
        self,
        df: pd.DataFrame,
        embedder: "TextEmbedder",
        extractor_factory=None,
    ) -> None:
        from features.extractor import CategoryFeatureExtractor

        if extractor_factory is None:
            extractor_factory = CategoryFeatureExtractor

        category_ids = sorted(df["category_id"].unique())
        logger.info("Building indices for %d categories: %s", len(category_ids), category_ids)

        for cat_id in category_ids:
            cat_df = df[df["category_id"] == cat_id].reset_index(drop=True)
            logger.info(
                "--- Category %d (%s): %d listings ---",
                cat_id, CATEGORY_DICT.get(cat_id, "?"), len(cat_df),
            )

            search_texts = self._build_search_texts(cat_df)

            try:
                extractor = extractor_factory(cat_id)
            except ValueError:
                logger.warning("No extractor for category %d, skipping attrs", cat_id)
                extractor = None

            metadata = self._build_metadata(cat_df, extractor)
            embeddings = embedder.embed_passages(search_texts, show_progress=True)

            idx = CategoryFAISSIndex(cat_id)
            idx.build(embeddings, metadata)
            self._indices[cat_id] = idx

        logger.info("All indices built. Total vectors: %d", self.total_size)

    def search(
        self,
        category_id: int,
        query_vector: np.ndarray,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        if category_id not in self._indices:
            raise KeyError(
                f"No index for category {category_id}. "
                f"Available: {list(self._indices.keys())}"
            )
        return self._indices[category_id].search(query_vector, top_k)

    def save_all(self, path: str | Path) -> None:
        for cat_id, idx in self._indices.items():
            idx.save(path)
        logger.info("Saved %d indices to %s", len(self._indices), path)

    def load_all(self, path: str | Path) -> None:
        dirpath = Path(path)
        faiss_files = sorted(dirpath.glob("*.faiss"))
        for f in faiss_files:
            cat_id = int(f.stem)
            idx = CategoryFAISSIndex(cat_id)
            idx.load(dirpath)
            self._indices[cat_id] = idx
        logger.info("Loaded %d indices from %s", len(self._indices), path)

    @property
    def total_size(self) -> int:
        return sum(idx.size for idx in self._indices.values())

    @property
    def categories(self) -> List[int]:
        return sorted(self._indices.keys())

    def stats(self) -> Dict[int, int]:
        return {cat_id: idx.size for cat_id, idx in sorted(self._indices.items())}

    def __repr__(self) -> str:
        parts = [f"  {cid}: {sz:,} vectors" for cid, sz in self.stats().items()]
        return f"FAISSIndexManager({len(self._indices)} indices)\n" + "\n".join(parts)

    @staticmethod
    def _build_search_texts(cat_df: pd.DataFrame) -> List[str]:
        texts = []
        for _, row in cat_df.iterrows():
            title = str(row.get("title", "") or "")
            desc = str(row.get("description", "") or "")[:300]
            texts.append(f"{title}. {desc}")
        return texts

    @staticmethod
    def _build_metadata(
        cat_df: pd.DataFrame,
        extractor: Any = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for _, row in cat_df.iterrows():
            title = str(row.get("title", "") or "")
            desc = str(row.get("description", "") or "")

            structured_attrs: Dict[str, Any] = {}
            if extractor is not None:
                try:
                    structured_attrs = extractor.extract(
                        title, desc,
                        photo_count=int(row.get("photo_count", 0)),
                        created_at=row.get("created_at"),
                    )
                except Exception:
                    pass

            record = {
                "advertisement_id": row.get("advertisement_id"),
                "title": title,
                "status": row.get("status"),
                "original_price": float(row["original_price"]) if pd.notna(row.get("original_price")) else None,
                "sold_price": float(row["sold_price"]) if pd.notna(row.get("sold_price")) else None,
                "created_at": str(row.get("created_at", "")),
                "modified_at": str(row.get("modified_at", "")),
                "days_to_sell": int(row["days_to_sell"]) if pd.notna(row.get("days_to_sell")) else None,
                "sold_via_bargain": bool(row.get("sold_via_bargain", False)),
                "photo_count": int(row.get("photo_count", 0)),
                "structured_attrs": structured_attrs,
            }
            records.append(record)
        return records
