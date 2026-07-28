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
    context_window_hours: float = 2.0,  # НОВОЕ: сохранять ±2 часа контекста
) -> bool:
    """
    Сохраняет событие + расширенный контекст для обучения CatBoost.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                # Рассчитать агрегированные признаки
                event_label = 1 if user_verdict == "real" else 0
                max_drop = abs(window_df["fuel_diff"].min()) if window_df is not None else 0
                drop_duration = _calculate_drop_duration(window_df) if window_df is not None else 0
                recovery_ratio = _calculate_recovery_ratio(window_df) if window_df is not None else 0
# Безопасно парсим время из строки (isoformat) в pd.Timestamp
                event_time_str = event_info.get("event_time")
                try:
                    parsed_time = pd.to_datetime(event_time_str)
                    hour_of_day = parsed_time.hour
                    day_of_week = parsed_time.dayofweek
                except Exception:
                    hour_of_day = 0
                    day_of_week = 0
                fuel_before = window_df["fuel_lvl"].iloc[0] if window_df is not None and len(window_df) > 0 else 0
                fuel_after = window_df["fuel_lvl"].iloc[-1] if window_df is not None and len(window_df) > 0 else 0
                
                # Вставить событие с агрегированными признаками
                cur.execute("""
                    INSERT INTO feedback_events (
                        object_name, event_time, address, user_verdict, system_label,
                        ml_detected, rule_detected, rule_reason,
                        ml_score_min, anomaly_points_count, total_drop, gap_drop,
                        event_label, max_drop_in_window, drop_duration_min,
                        time_since_last_refuel_min, recovery_ratio,
                        hour_of_day, day_of_week,
                        fuel_level_before, fuel_level_after
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_info.get("object_name", ""),
                    event_info.get("event_time", ""),
                    event_info.get("address", "") or None,
                    user_verdict,
                    event_info.get("system_label", ""),
                    bool(event_info.get("ml_detected", False)),
                    bool(event_info.get("rule_detected", False)),
                    event_info.get("rule_reason", "") or None,
                    float(event_info.get("ml_score_min", 0.0) or 0.0),
                    int(event_info.get("anomaly_points_count", 0) or 0),
                    float(event_info.get("total_drop", 0.0) or 0.0),
                    float(event_info.get("gap_drop", 0.0) or 0.0),
                    event_label,
                    max_drop,
                    drop_duration,
                    None,  # time_since_last_refuel — пока не считаем
                    recovery_ratio,
                    hour_of_day,
                    day_of_week,
                    fuel_before,
                    fuel_after,
                ))
                event_id = cur.fetchone()[0]
                
                # Вставить точки с расширенным контекстом
                if window_df is not None and not window_df.empty:
                    # Сохраняем все точки в окне ±2 часа
                    points_rows = []
                    for _, row in window_df.iterrows():
                        label = _label_point(row, user_verdict)
                        feat_vals = [float(row.get(col, 0) or 0) for col in FEATURE_COLUMNS]
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
        return True
    except Exception as e:
        logger.error(f"Ошибка записи feedback: {e}")
        return False


def _calculate_drop_duration(window_df: pd.DataFrame) -> float:
    """Рассчитать длительность падения в минутах."""
    if window_df is None or len(window_df) == 0:
        return 0
    drops = window_df[window_df["fuel_diff"] < -0.5]
    if len(drops) == 0:
        return 0
    return (drops["datetime"].max() - drops["datetime"].min()).total_seconds() / 60


def _calculate_recovery_ratio(window_df: pd.DataFrame) -> float:
    """Рассчитать коэффициент восстановления уровня топлива."""
    if window_df is None or len(window_df) < 10:
        return 0
    min_fuel = window_df["fuel_lvl"].min()
    max_fuel = window_df["fuel_lvl"].max()
    if max_fuel == min_fuel:
        return 0
    # Найти точку минимума и посмотреть, насколько уровень восстановился после неё
    min_idx = window_df["fuel_lvl"].idxmin()
    after_min = window_df.loc[min_idx:]
    if len(after_min) < 2:
        return 0
    final_fuel = after_min["fuel_lvl"].iloc[-1]
    recovery = final_fuel - min_fuel
    total_drop = max_fuel - min_fuel
    return recovery / total_drop if total_drop > 0 else 0


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
