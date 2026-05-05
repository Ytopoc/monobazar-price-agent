# LightGBM inference: predicts price and time-to-sell.
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.trainer import (
    CATEGORICAL_ATTRS,
    NUMERIC_ATTRS,
    COMP_COLS,
    PricingModelTrainer,
)

logger = logging.getLogger(__name__)


def _round_price(price: float) -> float:
    if price <= 0:
        return 0.0
    if price > 5000:
        return round(price / 100) * 100
    if price > 500:
        return round(price / 50) * 50
    if price > 50:
        return round(price / 10) * 10
    return round(price / 5) * 5


class PricingPredictor:
    def __init__(
        self,
        models: Dict[str, Any],
        category_stats: Dict[str, float],
    ) -> None:
        self._model_q25 = models.get("model_q25")
        self._model_q50 = models.get("model_q50")
        self._model_q75 = models.get("model_q75")
        self._model_days = models.get("model_days")

        self._feature_names: List[str] = models.get("feature_names", [])
        self._categorical_features: List[str] = models.get("categorical_features", [])
        self._label_encoders: Dict[str, Dict[str, int]] = models.get("label_encoders", {})

        self._cat_stats = category_stats

    def predict(
        self,
        features: Dict[str, Any],
        category_id: int,
    ) -> Dict[str, Any]:
        X = self._build_feature_row(features, category_id)

        q25 = float(self._predict_single(self._model_q25, X))
        q50 = float(self._predict_single(self._model_q50, X))
        q75 = float(self._predict_single(self._model_q75, X))

        price_min = self._cat_stats.get("price_min", 10)
        price_max = self._cat_stats.get("price_max", 1_000_000)
        q25 = np.clip(q25, price_min, price_max)
        q50 = np.clip(q50, price_min, price_max)
        q75 = np.clip(q75, price_min, price_max)

        comp_median = features.get("comp_sold_median", 0)
        if comp_median <= 0:
            comp_median = features.get("comp_active_median", 0)
        if comp_median > 0 and q50 > 0:
            ratio = q50 / comp_median
            if ratio < 0.5 or ratio > 2.0:
                blend = 0.3
                scale25 = q25 / q50 if q50 > 0 else 0.9
                scale75 = q75 / q50 if q50 > 0 else 1.1
                q50 = q50 * blend + comp_median * (1 - blend)
                q25 = q50 * scale25
                q75 = q50 * scale75
            elif ratio < 0.75 or ratio > 1.5:
                blend = 0.6
                scale25 = q25 / q50 if q50 > 0 else 0.9
                scale75 = q75 / q50 if q50 > 0 else 1.1
                q50 = q50 * blend + comp_median * (1 - blend)
                q25 = q50 * scale25
                q75 = q50 * scale75

        _CONDITION_MULTIPLIER = {
            "new": 1.0,
            "like_new": 1.0,
            "very_good": 0.97,
            "good": 0.93,
            "fair": 0.80,
            "needs_repair": 0.50,
        }
        condition = str(features.get("condition", "good")).lower()
        cond_mult = _CONDITION_MULTIPLIER.get(condition, 0.90)
        logger.info(
            "Condition adjustment: condition=%s, multiplier=%.2f, q50 before=%.0f, after=%.0f",
            condition, cond_mult, q50, q50 * cond_mult,
        )
        q25 *= cond_mult
        q50 *= cond_mult
        q75 *= cond_mult

        battery_pct = features.get("battery_pct")
        if battery_pct is not None:
            try:
                bp = float(battery_pct)
                if bp < 70:
                    bat_mult = 0.85
                elif bp < 85:
                    bat_mult = 0.93
                elif bp < 93:
                    bat_mult = 0.97
                else:
                    bat_mult = 1.0
                if bat_mult < 1.0:
                    logger.info(
                        "Battery adjustment: %s%% -> multiplier=%.2f",
                        bp, bat_mult,
                    )
                    q25 *= bat_mult
                    q50 *= bat_mult
                    q75 *= bat_mult
            except (ValueError, TypeError):
                pass

        q25 = min(q25, q50)
        q75 = max(q75, q50)

        bargain_rate = self._cat_stats.get("bargain_rate", 0.0)
        bargain_buffer = bargain_rate * 0.15

        price_fast = q25
        price_balanced = q50 * (1 + bargain_buffer * 0.5)
        price_max_rec = q75 * (1 + bargain_buffer)

        price_fast = _round_price(price_fast)
        price_balanced = _round_price(price_balanced)
        price_max_rec = _round_price(price_max_rec)
        q25_r = _round_price(q25)
        q50_r = _round_price(q50)
        q75_r = _round_price(q75)

        days_fast, days_balanced, days_max = self._predict_days(X, q25, q50, q75)

        median_val = max(q50, 1)
        interval_ratio = (q75 - q25) / median_val
        confidence = max(0.0, min(1.0, 1.0 - interval_ratio))

        return {
            "price_q25": q25_r,
            "price_q50": q50_r,
            "price_q75": q75_r,
            "price_fast": price_fast,
            "price_balanced": price_balanced,
            "price_max": price_max_rec,
            "days_fast": days_fast,
            "days_balanced": days_balanced,
            "days_max": days_max,
            "confidence": round(confidence, 3),
        }

    def predict_batch(
        self,
        feature_rows: List[Dict[str, Any]],
        category_id: int,
    ) -> List[Dict[str, Any]]:
        return [self.predict(f, category_id) for f in feature_rows]

    def _build_feature_row(
        self,
        features: Dict[str, Any],
        category_id: int,
    ) -> pd.DataFrame:
        row: Dict[str, Any] = {}

        cat_cols = CATEGORICAL_ATTRS.get(category_id, [])
        num_cols = NUMERIC_ATTRS.get(category_id, [])

        for col in cat_cols:
            val = features.get(col)
            if val is None:
                val = "__missing__"
            val = str(val)
            encoder = self._label_encoders.get(col, {})
            encoded = encoder.get(val, encoder.get("__other__", 0))
            row[col] = encoded

        for col in num_cols:
            val = features.get(col)
            if isinstance(val, bool):
                row[col] = int(val)
            elif val is not None:
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = np.nan
            else:
                row[col] = np.nan

        row["original_price"] = float(features.get("original_price", 0))

        for comp_col in COMP_COLS:
            row[comp_col] = float(features.get(comp_col, 0.0))

        df = pd.DataFrame([row])

        for col in self._feature_names:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self._feature_names]

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def _predict_single(self, model: Any, X: pd.DataFrame) -> float:
        if model is None:
            return 0.0
        if isinstance(model, lgb.Booster):
            return float(model.predict(X)[0])
        else:
            return float(model.predict(X)[0])

    def _predict_days(
        self,
        X: pd.DataFrame,
        q25: float,
        q50: float,
        q75: float,
    ) -> tuple[float, float, float]:
        if self._model_days is not None:
            base_days = max(1.0, float(self._predict_single(self._model_days, X)))
        else:
            base_days = max(1.0, self._cat_stats.get("avg_days_to_sell", 14.0))

        days_balanced = round(base_days, 1)
        days_fast = round(max(1.0, base_days * 0.6), 1)
        days_max = round(base_days * 1.8, 1)

        return days_fast, days_balanced, days_max

    @classmethod
    def from_saved(
        cls,
        models_dir: str | Path,
        category_id: int,
        category_stats: Dict[str, float],
    ) -> "PricingPredictor":
        models = PricingModelTrainer.load_models(models_dir, category_id)
        return cls(models=models, category_stats=category_stats)
