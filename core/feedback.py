"""Сбор обратной связи — запись в PostgreSQL."""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "fuel_drop_rate", "fuel_volatility_15min", "drawdown_15min", "drawdown_30min",
    "total_drop_10min", "drop_duration_min", "drop_consistency", "consecutive_drops",
    "is_large_gap", "monotonic_drop_length", "time_diff_min", "is_gnss_anomaly",
    "is_gsm_weak", "is_connection_lost", "is_engine_on", "speed",
]

INSERT_EVENT_SQL = """
    INSERT INTO feedback_events (
        object_name, event_time, address, user_verdict, system_label,
        ml_detected, rule_detected, rule_reason,
        ml_score_min, anomaly_points_count, total_drop, gap_drop
    ) VALUES %s
    RETURNING id
"""

INSERT_POINTS_SQL = """
    INSERT INTO feedback_points (
        event_id, datetime,
        feat_fuel_drop_rate, feat_fuel_volatility_15min, feat_drawdown_15min,
        feat_drawdown_30min, feat_total_drop_10min, feat_drop_duration_min,
        feat_drop_consistency, feat_consecutive_drops, feat_is_large_gap,
        feat_monotonic_drop_length, feat_time_diff_min, feat_is_gnss_anomaly,
        feat_is_gsm_weak, feat_is_connection_lost, feat_is_engine_on, feat_speed,
        is_anomaly_ml, fuel_diff, speed, is_gnss_anomaly_raw,
        point_label
    ) VALUES %s
"""


def _get_connection():
    """Создаёт соединение с PostgreSQL."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        connect_timeout=5,
        options="-c statement_timeout=5000",
    )


def _label_point(row: pd.Series, user_verdict: str) -> int:
    """Размечает одну точку внутри окна."""
    if user_verdict == "false":
        return 0
    
    if row.get("is_anomaly") == -1:
        return 1
    
    if (row.get("fuel_diff", 0) < -0.5 and 
        row.get("speed", 0) == 0 and 
        row.get("is_gnss_anomaly", 1) == 0 and
        row.get("total_drop_10min", 0) > 3.0):
        return 1
    
    if (row.get("time_diff_min", 0) > 60 and 
        row.get("fuel_diff", 0) < -5.0):
        return 1
    
    return 0


def save_feedback(
    event_info: Dict, 
    user_verdict: str, 
    features_row: pd.Series,
    window_df: pd.DataFrame | None = None,
) -> bool:
    """Сохраняет событие + ВСЕ точки окна для обучения CatBoost."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                event_row = (
                    event_info.get("object_name", ""),
                    event_info.get("event_time", ""),
                    event_info.get("address", "") or None,
                    user_verdict,
                    event_info.get("system_label", event_info.get("label", "")),
                    bool(event_info.get("ml_detected", False)),
                    bool(event_info.get("rule_detected", False)),
                    event_info.get("rule_reason", "") or None,
                    float(event_info.get("ml_score_min", 0.0) or 0.0),
                    int(event_info.get("anomaly_points_count", 0) or 0),
                    float(event_info.get("total_drop", 0.0) or 0.0),
                    float(event_info.get("gap_drop", 0.0) or 0.0),
                )
                cur.execute(INSERT_EVENT_SQL, [event_row])
                event_id = cur.fetchone()[0]
                
                if window_df is not None and not window_df.empty:
                    points_rows = []
                    for _, row in window_df.iterrows():
                        label = _label_point(row, user_verdict)
                        
                        feat_vals = []
                        for col in FEATURE_COLUMNS:
                            val = row.get(col, None)
                            if val is None or (isinstance(val, float) and pd.isna(val)):
                                feat_vals.append(None)
                            else:
                                feat_vals.append(float(val))
                        
                        points_rows.append((
                            event_id,
                            row.get("datetime"),
                            *feat_vals,
                            int(row.get("is_anomaly", 1)),
                            float(row.get("fuel_diff", 0) or 0),
                            float(row.get("speed", 0) or 0),
                            int(row.get("is_gnss_anomaly", 0) or 0),
                            label,
                        ))
                    
                    execute_values(cur, INSERT_POINTS_SQL, points_rows, page_size=100)
                
            conn.commit()
        
        n_points = len(window_df) if window_df is not None else 0
        logger.info(f"✅ Feedback saved: event_id={event_id}, {n_points} points, verdict={user_verdict}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка записи feedback: {e}")
        return False


def get_feedback_count() -> int:
    """Возвращает количество записей в таблице."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM feedback_events")
                return int(cur.fetchone()[0])
    except Exception:
        return -1


def load_feedback(limit: int = 100) -> pd.DataFrame:
    """Загружает последние N записей из feedback."""
    try:
        with _get_connection() as conn:
            query = f"""
                SELECT * FROM feedback_events
                ORDER BY feedback_ts DESC
                LIMIT {int(limit)}
            """
            return pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"❌ Ошибка чтения feedback: {e}")
        return pd.DataFrame()