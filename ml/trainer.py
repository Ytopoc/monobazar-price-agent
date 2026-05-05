# LightGBM training with leave-one-out FAISS features.
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.category_config import CATEGORY_DICT
from features.comparable import compute_comparable_features

logger = logging.getLogger(__name__)

CATEGORICAL_ATTRS: Dict[int, List[str]] = {
    4:    ["model", "brand", "color", "condition"],
    512:  ["brand", "model_line", "condition", "gender"],
    743:  ["brand", "theme", "condition", "completeness"],
    795:  ["condition", "cover_type", "language"],
    1261: ["brand", "season", "condition", "width_profile"],
    1320: ["material", "brand", "type", "condition"],
    1677: ["series", "franchise", "condition", "completeness"],
}

NUMERIC_ATTRS: Dict[int, List[str]] = {
    4:    ["storage_gb", "battery_pct", "title_length", "desc_length",
           "photo_count", "month", "day_of_week", "neverlock", "has_box", "has_emoji"],
    512:  ["size", "title_length", "desc_length", "photo_count",
           "month", "day_of_week", "is_original", "has_emoji"],
    743:  ["piece_count", "title_length", "desc_length", "photo_count",
           "month", "day_of_week", "has_box", "has_emoji"],
    795:  ["title_length", "desc_length", "photo_count",
           "month", "day_of_week", "is_set", "has_emoji"],
    1261: ["size_r", "width", "profile", "tread_pct", "title_length",
           "desc_length", "photo_count", "month", "day_of_week", "has_emoji"],
    1320: ["title_length", "desc_length", "photo_count",
           "month", "day_of_week", "has_emoji"],
    1677: ["title_length", "desc_length", "photo_count",
           "month", "day_of_week", "is_rare", "has_emoji"],
}

COMP_COLS = [
    "comp_sold_median", "comp_sold_mean", "comp_sold_std",
    "comp_sold_min", "comp_sold_max", "comp_sold_count",
    "comp_active_median", "comp_active_count",
    "comp_sold_ratio", "comp_avg_days_to_sell",
    "comp_bargain_rate", "comp_avg_discount",
    "comp_avg_similarity", "comp_top1_similarity",
]

RARE_THRESHOLD = 5

_LGBM_BASE_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


class PricingModelTrainer:
    def __init__(
        self,
        faiss_manager: Any,
        embedder: Any,
    ) -> None:
        self.faiss_manager = faiss_manager
        self.embedder = embedder
        self._label_encoders: Dict[int, Dict[str, Dict[str, int]]] = {}

    def build_training_features(
        self,
        df_sold: pd.DataFrame,
        category_id: int,
        top_k: int = 30,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        from features.extractor import CategoryFeatureExtractor

        logger.info(
            "Building training features for category %d (%d SOLD records, top_k=%d)",
            category_id, len(df_sold), top_k,
        )
        t0 = time.time()

        extractor = CategoryFeatureExtractor(category_id)
        cat_cols = CATEGORICAL_ATTRS.get(category_id, [])
        num_cols = NUMERIC_ATTRS.get(category_id, [])

        rows: List[Dict[str, Any]] = []

        search_texts = [
            f"{row['title']}. {str(row.get('description',''))[:300]}"
            for _, row in df_sold.iterrows()
        ]
        all_query_vecs = self.embedder.embed_passages(search_texts, show_progress=True)

        for i, (_, row) in enumerate(df_sold.iterrows()):
            ad_id = row["advertisement_id"]

            query_vec = all_query_vecs[i:i+1]

            raw_results = self.faiss_manager.search(
                category_id, query_vec, top_k=top_k + 5,
            )

            neighbours = [r for r in raw_results if r.get("advertisement_id") != ad_id][:top_k]

            comp_feats = compute_comparable_features(neighbours)

            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            struct_attrs = extractor.extract(
                title, desc,
                photo_count=int(row.get("photo_count", 0)),
                created_at=row.get("created_at"),
            )

            feat_row: Dict[str, Any] = {}

            for col in cat_cols:
                feat_row[col] = struct_attrs.get(col)

            for col in num_cols:
                val = struct_attrs.get(col)
                if isinstance(val, bool):
                    feat_row[col] = int(val)
                elif val is not None:
                    try:
                        feat_row[col] = float(val)
                    except (ValueError, TypeError):
                        feat_row[col] = np.nan
                else:
                    feat_row[col] = np.nan

            feat_row["original_price"] = float(row["original_price"])

            for comp_col in COMP_COLS:
                feat_row[comp_col] = comp_feats.get(comp_col, 0.0)

            rows.append(feat_row)

        X = pd.DataFrame(rows)
        y_price = df_sold["sold_price"].astype(float).reset_index(drop=True)
        y_days = df_sold["days_to_sell"].astype(float).reset_index(drop=True)

        X, encoder_map = self._encode_categoricals(X, cat_cols, category_id, fit=True)

        logger.info(
            "Built %d feature rows (%d columns) in %.1fs",
            len(X), len(X.columns), time.time() - t0,
        )
        return X, y_price, y_days

    def train(
        self,
        X: pd.DataFrame,
        y_price: pd.Series,
        y_days: pd.Series,
        category_id: int,
        temporal_split_date: str = "2026-03-01",
        created_at_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        cat_cols_encoded = [c for c in X.columns
                           if c in CATEGORICAL_ATTRS.get(category_id, [])]

        if created_at_series is not None:
            split_dt = pd.to_datetime(temporal_split_date, utc=True)
            train_mask = created_at_series < split_dt
            test_mask = ~train_mask
        else:
            n = len(X)
            split_idx = int(n * 0.8)
            train_mask = pd.Series([True] * split_idx + [False] * (n - split_idx))
            test_mask = ~train_mask

        X_train, X_test = X[train_mask].reset_index(drop=True), X[test_mask].reset_index(drop=True)
        y_price_train = y_price[train_mask].reset_index(drop=True)
        y_price_test = y_price[test_mask].reset_index(drop=True)
        y_days_train = y_days[train_mask].reset_index(drop=True)
        y_days_test = y_days[test_mask].reset_index(drop=True)

        logger.info(
            "Category %d: train=%d, test=%d (split=%s)",
            category_id, len(X_train), len(X_test), temporal_split_date,
        )

        if len(X_train) < 20:
            logger.warning("Too few training samples (%d), skipping", len(X_train))
            return {}

        feature_names = list(X.columns)
        cat_feature_names = [c for c in cat_cols_encoded if c in feature_names]

        callbacks = [lgb.log_evaluation(period=0)]

        models: Dict[str, Any] = {}

        logger.info("  Training model_q25 ...")
        models["model_q25"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.25,
            **_LGBM_BASE_PARAMS,
        )
        models["model_q25"].fit(
            X_train, y_price_train,
            eval_set=[(X_test, y_price_test)] if len(X_test) > 0 else None,
            callbacks=callbacks + ([lgb.early_stopping(50, verbose=False)] if len(X_test) > 0 else []),
            categorical_feature=cat_feature_names if cat_feature_names else "auto",
        )

        logger.info("  Training model_q50 ...")
        models["model_q50"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.50,
            **_LGBM_BASE_PARAMS,
        )
        models["model_q50"].fit(
            X_train, y_price_train,
            eval_set=[(X_test, y_price_test)] if len(X_test) > 0 else None,
            callbacks=callbacks + ([lgb.early_stopping(50, verbose=False)] if len(X_test) > 0 else []),
            categorical_feature=cat_feature_names if cat_feature_names else "auto",
        )

        logger.info("  Training model_q75 ...")
        models["model_q75"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.75,
            **_LGBM_BASE_PARAMS,
        )
        models["model_q75"].fit(
            X_train, y_price_train,
            eval_set=[(X_test, y_price_test)] if len(X_test) > 0 else None,
            callbacks=callbacks + ([lgb.early_stopping(50, verbose=False)] if len(X_test) > 0 else []),
            categorical_feature=cat_feature_names if cat_feature_names else "auto",
        )

        valid_days_train = y_days_train.notna()
        valid_days_test = y_days_test.notna()

        if valid_days_train.sum() >= 20:
            logger.info("  Training model_days (%d samples) ...", valid_days_train.sum())
            models["model_days"] = lgb.LGBMRegressor(
                objective="regression",
                **_LGBM_BASE_PARAMS,
            )
            eval_set_days = (
                [(X_test[valid_days_test], y_days_test[valid_days_test])]
                if valid_days_test.sum() > 0 else None
            )
            models["model_days"].fit(
                X_train[valid_days_train], y_days_train[valid_days_train],
                eval_set=eval_set_days,
                callbacks=callbacks + ([lgb.early_stopping(50, verbose=False)] if eval_set_days else []),
                categorical_feature=cat_feature_names if cat_feature_names else "auto",
            )
        else:
            logger.warning("  Too few valid days_to_sell (%d), skipping model_days", valid_days_train.sum())
            models["model_days"] = None

        metrics = {}
        if len(X_test) > 0 and len(y_price_test) > 0:
            metrics = self._evaluate(models, X_test, y_price_test, y_days_test)
            self._print_metrics(category_id, metrics)

        result = {
            "model_q25": models["model_q25"],
            "model_q50": models["model_q50"],
            "model_q75": models["model_q75"],
            "model_days": models.get("model_days"),
            "feature_names": feature_names,
            "categorical_features": cat_feature_names,
            "metrics": metrics,
            "label_encoders": self._label_encoders.get(category_id, {}),
        }
        return result

    def _evaluate(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_price_test: pd.Series,
        y_days_test: pd.Series,
    ) -> Dict[str, float]:
        y_true = y_price_test.values
        pred_q25 = models["model_q25"].predict(X_test)
        pred_q50 = models["model_q50"].predict(X_test)
        pred_q75 = models["model_q75"].predict(X_test)

        mae = float(np.mean(np.abs(y_true - pred_q50)))
        mask_nonzero = y_true > 0
        mape = float(np.mean(np.abs((y_true[mask_nonzero] - pred_q50[mask_nonzero]) / y_true[mask_nonzero]))) * 100

        in_range = ((y_true >= pred_q25) & (y_true <= pred_q75))
        coverage = float(np.mean(in_range)) * 100

        interval_width = float(np.mean((pred_q75 - pred_q25) / np.maximum(pred_q50, 1)))

        metrics = {
            "mae": mae,
            "mape": mape,
            "coverage_50": coverage,
            "interval_width": interval_width,
            "test_size": len(y_true),
        }

        if models.get("model_days") is not None:
            valid = y_days_test.notna()
            if valid.sum() > 0:
                pred_days = models["model_days"].predict(X_test[valid])
                days_mae = float(np.mean(np.abs(y_days_test[valid].values - pred_days)))
                metrics["days_mae"] = days_mae

        return metrics

    def _print_metrics(self, category_id: int, metrics: Dict[str, float]) -> None:
        name = CATEGORY_DICT.get(category_id, str(category_id))
        logger.info("  --- Test metrics for [%d] %s ---", category_id, name)
        logger.info("    MAPE:          %.1f%%", metrics.get("mape", 0))
        logger.info("    MAE:           %.0f UAH", metrics.get("mae", 0))
        logger.info("    Coverage[25-75]: %.1f%%", metrics.get("coverage_50", 0))
        logger.info("    Interval width: %.2f", metrics.get("interval_width", 0))
        logger.info("    Test size:      %d", metrics.get("test_size", 0))
        if "days_mae" in metrics:
            logger.info("    Days MAE:      %.1f days", metrics["days_mae"])

    def _encode_categoricals(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        category_id: int,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        df = df.copy()

        if fit:
            encoders: Dict[str, Dict[str, int]] = {}
            for col in cat_cols:
                if col not in df.columns:
                    continue
                df[col] = df[col].fillna("__missing__").astype(str)
                counts = df[col].value_counts()
                rare = set(counts[counts < RARE_THRESHOLD].index)
                df[col] = df[col].apply(lambda x: "__other__" if x in rare else x)
                unique_vals = sorted(df[col].unique())
                mapping = {v: i for i, v in enumerate(unique_vals)}
                encoders[col] = mapping
                df[col] = df[col].map(mapping).astype("category")

            self._label_encoders[category_id] = encoders
        else:
            encoders = self._label_encoders.get(category_id, {})
            for col in cat_cols:
                if col not in df.columns or col not in encoders:
                    continue
                df[col] = df[col].fillna("__missing__").astype(str)
                mapping = encoders[col]
                df[col] = df[col].apply(
                    lambda x, m=mapping: m.get(x, m.get("__other__", 0))
                ).astype("category")

        return df, encoders

    def save_models(self, models: Dict[str, Any], path: str | Path) -> None:
        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)

        save_data = {}
        for key in ("model_q25", "model_q50", "model_q75", "model_days"):
            model = models.get(key)
            if model is not None:
                save_data[key] = model.booster_.model_to_string()
            else:
                save_data[key] = None

        save_data["feature_names"] = models.get("feature_names", [])
        save_data["categorical_features"] = models.get("categorical_features", [])
        save_data["metrics"] = models.get("metrics", {})
        save_data["label_encoders"] = models.get("label_encoders", {})

        out_file = dirpath / "models.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved models to %s", out_file)

    def save_category_models(
        self,
        models: Dict[str, Any],
        path: str | Path,
        category_id: int,
    ) -> None:
        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)

        save_data = {}
        for key in ("model_q25", "model_q50", "model_q75", "model_days"):
            model = models.get(key)
            if model is not None:
                save_data[key] = model.booster_.model_to_string()
            else:
                save_data[key] = None

        save_data["feature_names"] = models.get("feature_names", [])
        save_data["categorical_features"] = models.get("categorical_features", [])
        save_data["metrics"] = models.get("metrics", {})
        save_data["label_encoders"] = models.get("label_encoders", {})

        out_file = dirpath / f"{category_id}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved category %d models to %s", category_id, out_file)

    @staticmethod
    def load_models(path: str | Path, category_id: int) -> Dict[str, Any]:
        filepath = Path(path) / f"{category_id}.pkl"
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        result: Dict[str, Any] = {}
        for key in ("model_q25", "model_q50", "model_q75", "model_days"):
            model_str = save_data.get(key)
            if model_str is not None:
                result[key] = lgb.Booster(model_str=model_str)
            else:
                result[key] = None

        result["feature_names"] = save_data.get("feature_names", [])
        result["categorical_features"] = save_data.get("categorical_features", [])
        result["metrics"] = save_data.get("metrics", {})
        result["label_encoders"] = save_data.get("label_encoders", {})
        return result
