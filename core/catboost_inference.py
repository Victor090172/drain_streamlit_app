"""
Инференс CatBoost для детекции сливов + теневой режим.
Рассчитывает event-level признаки, делает предсказание,
записывает результат в catboost_shadow_log.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.feedback import _get_connection, detect_last_refuel

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "catboost_model.pkl"

# Параметры расчёта признаков (должны совпадать с prepare_catboost_dataset.py)
REFUEL_LOOKBACK_MIN = 60.0
RECOVERY_LOOKAHEAD_MIN = 60.0
RECENT_REFUEL_THRESHOLD_MIN = 60.0
NO_REFUEL_VALUE = 999.0
MAX_REASONABLE_DROP_L = 1000.0
SPEED_WINDOW_ROWS = 30

_MODEL_CACHE = None


def load_catboost_model():
    """Загружает артефакт CatBoost (кэшировано). None если модели нет."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        logger.info("CatBoost модель не найдена — теневой режим неактивен")
        return None
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        return _MODEL_CACHE
    except Exception as e:
        logger.warning(f"Не удалось загрузить CatBoost модель: {e}")
        return None


def _normalize_event_time(df_feat: pd.DataFrame, event_time: pd.Timestamp) -> pd.Timestamp:
    et = event_time
    if df_feat["datetime"].dt.tz is None and et.tz is not None:
        et = et.tz_localize(None)
    elif df_feat["datetime"].dt.tz is not None and et.tz is None:
        et = et.tz_localize(df_feat["datetime"].dt.tz)
    return et


def compute_event_features(
    df_feat: pd.DataFrame,
    event_time: pd.Timestamp,
    event_idx: int,
    main_window: pd.DataFrame,
    verdict: dict,
    time_since_refuel: float,
) -> dict:
    """Рассчитывает event-level признаки для CatBoost (зеркально prepare_catboost_dataset.py)."""
    et = _normalize_event_time(df_feat, event_time)
    before = df_feat[df_feat["datetime"] <= et]
    after = df_feat[df_feat["datetime"] > et]

    feats = {}

    # Из вердикта
    feats["recovery_ratio"] = float(verdict.get("recovery_ratio") or 0.0)
    feats["ml_score_min"] = float(verdict.get("ml_score_min") or 0.0)
    feats["anomaly_points_count"] = int(verdict.get("anomaly_points_count") or 0)

    # Временные
    feats["hour_of_day"] = int(et.hour)
    feats["day_of_week"] = int(et.dayofweek)

    # Уровень до/после
    if len(main_window) > 0:
        feats["fuel_level_before"] = float(main_window["fuel_lvl"].iloc[0])
        feats["fuel_level_after"] = float(main_window["fuel_lvl"].iloc[-1])
    else:
        feats["fuel_level_before"] = np.nan
        feats["fuel_level_after"] = np.nan

    # Время с заправки
    if time_since_refuel is None or pd.isna(time_since_refuel) or time_since_refuel >= NO_REFUEL_VALUE:
        feats["time_since_last_refuel_min"] = np.nan
        feats["has_recent_refuel"] = 0
    else:
        feats["time_since_last_refuel_min"] = float(time_since_refuel)
        feats["has_recent_refuel"] = 1 if time_since_refuel < RECENT_REFUEL_THRESHOLD_MIN else 0

    # Робастное падение
    if len(before) > 0 and len(after) > 0:
        total_drop = before["fuel_lvl"].max() - after["fuel_lvl"].min()
    elif len(before) > 1:
        total_drop = before["fuel_lvl"].max() - before["fuel_lvl"].iloc[-1]
    else:
        total_drop = 0.0
    feats["total_drop_robust"] = float(np.clip(abs(total_drop), 0, MAX_REASONABLE_DROP_L))

    # Макс. рост до события
    lookback_start = et - pd.Timedelta(minutes=REFUEL_LOOKBACK_MIN)
    before_window = df_feat[(df_feat["datetime"] >= lookback_start) & (df_feat["datetime"] <= et)]
    max_increase = (
        float(before_window["fuel_lvl"].max() - before_window["fuel_lvl"].min())
        if len(before_window) >= 2 else 0.0
    )
    feats["max_fuel_increase_before_drop"] = max_increase
    feats["drop_to_increase_ratio"] = (
        float(abs(total_drop) / max_increase) if max_increase > 0 else np.nan
    )

    # Скорость
    start = max(0, event_idx - SPEED_WINDOW_ROWS)
    end = min(len(df_feat) - 1, event_idx + SPEED_WINDOW_ROWS)
    speed_window = df_feat.iloc[start:end + 1]
    if "speed" in speed_window.columns and len(speed_window) > 0:
        speeds = speed_window["speed"].fillna(0)
        feats["avg_speed"] = float(speeds.mean())
        feats["max_speed"] = float(speeds.max())
        feats["moving_share"] = float((speeds > 2.0).mean())
    else:
        feats["avg_speed"] = feats["max_speed"] = feats["moving_share"] = 0.0

    # Восстановление после события
    lookahead_end = et + pd.Timedelta(minutes=RECOVERY_LOOKAHEAD_MIN)
    after_window = df_feat[(df_feat["datetime"] > et) & (df_feat["datetime"] <= lookahead_end)]
    if len(after_window) >= 2:
        min_after = after_window["fuel_lvl"].min()
        span = after_window["fuel_lvl"].max() - min_after
        recovery = after_window["fuel_lvl"].iloc[-1] - min_after
        feats["recovery_ratio_60min"] = float(recovery / span) if span > 0 else 0.0
        dur_min = (after_window["datetime"].max() - after_window["datetime"].min()).total_seconds() / 60
        feats["recovery_speed"] = float(recovery / dur_min) if dur_min > 0 else 0.0
    else:
        feats["recovery_ratio_60min"] = 0.0
        feats["recovery_speed"] = 0.0

    # Волатильность до события
    feats["volatility_before"] = (
        float(before_window["fuel_lvl"].std()) if len(before_window) >= 2 else 0.0
    )

    # РЭБ и разрывы связи
    feats["gnss_anomaly_count"] = (
        int(speed_window["is_gnss_anomaly"].fillna(0).sum()) if "is_gnss_anomaly" in speed_window.columns else 0
    )
    feats["connection_lost_count"] = (
        int(speed_window["is_connection_lost"].fillna(0).sum()) if "is_connection_lost" in speed_window.columns else 0
    )

    return feats


def predict_catboost(artifact: dict, features: dict) -> dict:
    """Делает предсказание CatBoost для одного события."""
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("threshold", 0.5)

    X = pd.DataFrame([features])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]

    prob = float(model.predict_proba(X)[0][1])
    prediction = "СЛИВ" if prob >= threshold else "ЛОЖНЫЙ"

    return {"probability": prob, "prediction": prediction, "threshold": threshold}


def save_catboost_shadow_log(
    event_id: int,
    object_name: str,
    event_time,
    catboost_prob: float,
    catboost_pred: str,
    threshold: float,
    system_label: str,
    user_verdict: str,
) -> bool:
    """Записывает теневое предсказание CatBoost в БД."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO catboost_shadow_log (
                        event_id, object_name, event_time,
                        catboost_prob, catboost_pred, threshold,
                        system_label, user_verdict
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        catboost_prob = EXCLUDED.catboost_prob,
                        catboost_pred = EXCLUDED.catboost_pred,
                        threshold = EXCLUDED.threshold,
                        system_label = EXCLUDED.system_label,
                        user_verdict = EXCLUDED.user_verdict
                """, (event_id, object_name, event_time, catboost_prob,
                      catboost_pred, threshold, system_label, user_verdict))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Не удалось записать catboost_shadow_log: {e}")
        return False


def predict_and_log_catboost(
    df_feat: pd.DataFrame,
    event_time,
    event_idx: int,
    main_window: pd.DataFrame,
    verdict: dict,
    event_id: int,
    object_name: str,
    user_verdict: str,
    time_since_refuel: float,
) -> dict:
    """
    Полный цикл теневого режима: предсказание CatBoost + запись в БД.
    Возвращает dict с результатом или None если модель недоступна.
    """
    artifact = load_catboost_model()
    if artifact is None:
        return None

    features = compute_event_features(
        df_feat, event_time, event_idx, main_window, verdict, time_since_refuel
    )
    result = predict_catboost(artifact, features)

    save_catboost_shadow_log(
        event_id=event_id,
        object_name=object_name,
        event_time=event_time,
        catboost_prob=result["probability"],
        catboost_pred=result["prediction"],
        threshold=result["threshold"],
        system_label=verdict.get("label", ""),
        user_verdict=user_verdict,
    )
    return result
