"""
Method 2: ML Candidate Ranking for pin repositioning.
Uses XGBoost or Random Forest to rank candidate positions.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Feature columns used for training (excludes metadata columns)
FEATURE_COLS = [
    "dist_from_pin_m",
    "avg_geocode_dist_m",
    "min_building_dist_m",
    "n_buildings_nearby",
    "n_geocoders",
    "confidence",
    "source_count",
    "is_current_pin",
    "is_geocode",
    "is_building",
    "cat_food",
    "cat_lodging",
    "cat_retail",
    "cat_services",
    "cat_health",
    "cat_auto",
]


def train_ranker(features_df: pd.DataFrame, labels: pd.Series,
                  model_type: str = "xgboost",
                  test_size: float = 0.2,
                  random_state: int = 42) -> dict:
    """
    Train a candidate ranking model.

    Args:
        features_df: DataFrame with feature columns + metadata
        labels: Binary labels (1 = best candidate)
        model_type: "xgboost" or "random_forest"
        test_size: Fraction for test split
        random_state: Random seed

    Returns dict with: model, metrics, feature_importance, split_indices
    """
    X = features_df[FEATURE_COLS].fillna(-1)
    y = labels

    # Stratified split by place_id to avoid data leakage
    place_ids = features_df["place_id"].unique()
    train_places, test_places = train_test_split(
        place_ids, test_size=test_size, random_state=random_state
    )
    train_mask = features_df["place_id"].isin(train_places)
    test_mask = features_df["place_id"].isin(test_places)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1])),
                random_state=random_state,
                eval_metric="logloss",
            )
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            model_type = "random_forest"

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.0

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    else:
        importance = {}

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"candidate_ranker_{model_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")

    return {
        "model": model,
        "model_type": model_type,
        "metrics": {
            "classification_report": report,
            "auc_roc": round(auc, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "feature_importance": dict(sorted(importance.items(), key=lambda x: -x[1])),
        "train_places": list(train_places),
        "test_places": list(test_places),
    }


def predict_best_candidate(model, candidates_features: pd.DataFrame) -> dict:
    """
    Given a set of candidate features for a single place,
    predict the best candidate.

    Returns dict with: lat, lon, source, probability
    """
    X = candidates_features[FEATURE_COLS].fillna(-1)
    probas = model.predict_proba(X)[:, 1]
    best_idx = int(np.argmax(probas))
    best_row = candidates_features.iloc[best_idx]

    return {
        "lat": best_row["candidate_lat"],
        "lon": best_row["candidate_lon"],
        "source": best_row["candidate_source"],
        "probability": float(probas[best_idx]),
    }


def reposition_with_ranker(df: pd.DataFrame, model,
                            geocode_results_map: dict | None = None,
                            building_centroids_map: dict | None = None,
                            ) -> pd.DataFrame:
    """
    Reposition all places using the trained candidate ranking model.

    Adds columns: ranker_lat, ranker_lon, ranker_source, ranker_probability
    """
    from src.features import generate_candidates, extract_candidate_features

    results = []
    for _, row in df.iterrows():
        place_id = row["id"]
        geocode_results = geocode_results_map.get(place_id, []) if geocode_results_map else []
        building_centroids = building_centroids_map.get(place_id, []) if building_centroids_map else []

        candidates = generate_candidates(row, geocode_results, building_centroids)

        # Extract features for all candidates
        candidate_features = []
        for candidate in candidates:
            features = extract_candidate_features(candidate, row, candidates)
            features["candidate_lat"] = candidate["lat"]
            features["candidate_lon"] = candidate["lon"]
            features["candidate_source"] = candidate["source"]
            candidate_features.append(features)

        if not candidate_features:
            results.append({
                "ranker_lat": row["lat"], "ranker_lon": row["lon"],
                "ranker_source": "fallback_current", "ranker_probability": 0.0,
            })
            continue

        cand_df = pd.DataFrame(candidate_features)
        best = predict_best_candidate(model, cand_df)
        results.append({
            "ranker_lat": best["lat"],
            "ranker_lon": best["lon"],
            "ranker_source": best["source"],
            "ranker_probability": best["probability"],
        })

    result_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, result_df], axis=1)
