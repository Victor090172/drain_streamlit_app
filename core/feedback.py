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

# Исправлено: количество колонок теперь совпадает с количеством передаваемых значений (21)
INSERT_EVENT_SQL = """
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


def _safe_float(val):
    """Безопасное преобразование в float для psycopg2 (обрабатывает np.nan и None)."""
    if val is None or pd.isna(val):
        return None
    return float(val)


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
    context_window_hours: float = 2.0,
) -> bool:
    """
    Сохраняет событие + расширенный контекст для обучения CatBoost.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                event_label = 1 if user_verdict == "real" else 0
                
                # Безопасный расчет агрегированных признаков
                max_drop = _safe_float(abs(window_df["fuel_diff"].min())) if window_df is not None and "fuel_diff" in window_df else 0.0
                drop_duration = _calculate_drop_duration(window_df) if window_df is not None else 0.0
                recovery_ratio = _calculate_recovery_ratio(window_df) if window_df is not None else 0.0
                
                # Надежное извлечение часа и дня недели (работает и со строкой, и с pd.Timestamp)
                event_time_val = event_info.get("event_time")
                try:
                    parsed_time = pd.to_datetime(event_time_val)
                    hour_of_day = int(parsed_time.hour)
                    day_of_week = int(parsed_time.dayofweek)
                except Exception:
                    hour_of_day = 0
                    day_of_week = 0
                    
                fuel_before = _safe_float(window_df["fuel_lvl"].iloc[0]) if window_df is not None and len(window_df) > 0 else 0.0
                fuel_after = _safe_float(window_df["fuel_lvl"].iloc[-1]) if window_df is not None and len(window_df) > 0 else 0.0
                
                # Вставка события (21 значение)
                cur.execute(INSERT_EVENT_SQL, (
                    event_info.get("object_name", ""),
                    event_info.get("event_time", ""),
                    event_info.get("address", "") or None,
                    user_verdict,
                    event_info.get("system_label", ""),
                    bool(event_info.get("ml_detected", False)),
                    bool(event_info.get("rule_detected", False)),
                    event_info.get("rule_reason", "") or None,
                    _safe_float(event_info.get("ml_score_min")),
                    int(event_info.get("anomaly_points_count", 0) or 0),
                    _safe_float(event_info.get("total_drop")),
                    _safe_float(event_info.get("gap_drop")),
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
                
                # Вставка точек с расширенным контекстом
                if window_df is not None and not window_df.empty:
                    points_rows = []
                    for _, row in window_df.iterrows():
                        label = _label_point(row, user_verdict)
                        
                        # Безопасное извлечение признаков
                        feat_vals = [_safe_float(row.get(col)) for col in FEATURE_COLUMNS]
                        
                        points_rows.append((
                            event_id,
                            row.get("datetime"),
                            *feat_vals,
                            int(row.get("is_anomaly", 1)),
                            _safe_float(row.get("fuel_diff")),
                            _safe_float(row.get("speed")),
                            int(row.get("is_gnss_anomaly", 0)),
                            int(label),
                        ))
                    execute_values(cur, INSERT_POINTS_SQL, points_rows, page_size=100)
                
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка записи feedback: {e}")
        return False


def _calculate_drop_duration(window_df: pd.DataFrame) -> float:
    """Рассчитать длительность падения в минутах."""
    if window_df is None or len(window_df) == 0 or "fuel_diff" not in window_df:
        return 0.0
    drops = window_df[window_df["fuel_diff"] < -0.5]
    if len(drops) == 0:
        return 0.0
    return float((drops["datetime"].max() - drops["datetime"].min()).total_seconds() / 60)


def _calculate_recovery_ratio(window_df: pd.DataFrame) -> float:
    """Рассчитать коэффициент восстановления уровня топлива."""
    if window_df is None or len(window_df) < 10 or "fuel_lvl" not in window_df:
        return 0.0
    
    min_fuel = window_df["fuel_lvl"].min()
    max_fuel = window_df["fuel_lvl"].max()
    
    if pd.isna(min_fuel) or pd.isna(max_fuel) or max_fuel == min_fuel:
        return 0.0
        
    min_idx = window_df["fuel_lvl"].idxmin()
    after_min = window_df.loc[min_idx:]
    
    if len(after_min) < 2:
        return 0.0
        
    final_fuel = after_min["fuel_lvl"].iloc[-1]
    recovery = final_fuel - min_fuel
    total_drop = max_fuel - min_fuel
    
    return float(recovery / total_drop) if total_drop > 0 else 0.0


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
    

def load_training_data(
    min_feedback_date: str = None,
    max_feedback_date: str = None,
    include_user_false: bool = True,
    include_user_real_normal_points: bool = True,
) -> pd.DataFrame:
    """
    Загружает данные для переобучения Isolation Forest из БД.
    """
    try:
        with _get_connection() as conn:
            query = """
                SELECT 
                    p.feat_fuel_drop_rate,
                    p.feat_fuel_volatility_15min,
                    p.feat_drawdown_15min,
                    p.feat_drawdown_30min,
                    p.feat_total_drop_10min,
                    p.feat_drop_duration_min,
                    p.feat_drop_consistency,
                    p.feat_consecutive_drops,
                    p.feat_is_large_gap,
                    p.feat_monotonic_drop_length,
                    p.feat_time_diff_min,
                    p.feat_is_gnss_anomaly,
                    p.feat_is_gsm_weak,
                    p.feat_is_connection_lost,
                    p.feat_is_engine_on,
                    p.feat_speed,
                    p.point_label,
                    e.user_verdict,
                    e.feedback_ts
                FROM feedback_points p
                JOIN feedback_events e ON p.event_id = e.id
                WHERE 1=1
            """
            
            params = []
            if min_feedback_date:
                query += " AND e.feedback_ts >= %s"
                params.append(min_feedback_date)
            if max_feedback_date:
                query += " AND e.feedback_ts <= %s"
                params.append(max_feedback_date)
            
            conditions = []
            if include_user_false:
                conditions.append("e.user_verdict = 'false'")
            if include_user_real_normal_points:
                conditions.append("(e.user_verdict = 'real' AND p.point_label = 0)")
            
            if conditions:
                query += " AND (" + " OR ".join(conditions) + ")"
            
            df = pd.read_sql(query, conn, params=params)
            
            rename_map = {col: col.replace("feat_", "") for col in df.columns if col.startswith("feat_")}
            df = df.rename(columns=rename_map)
            
            logger.info(f"📊 Загружено {len(df)} точек для обучения")
            return df
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных для обучения: {e}")
        return pd.DataFrame()
