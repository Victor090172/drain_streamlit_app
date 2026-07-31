"""Сбор обратной связи — запись в PostgreSQL."""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD,
    # Контекстное окно для CatBoost
    CONTEXT_HOURS_BEFORE, CONTEXT_HOURS_AFTER,
    CONTEXT_ROWS_BEFORE, CONTEXT_ROWS_AFTER,
    # Пороги определения заправки
    REFUEL_THRESHOLD_L, MAX_SPEED_FOR_REFUEL,
    REFUEL_STABILITY_RATIO, REFUEL_STABILITY_WINDOW_MIN,
    NO_REFUEL_VALUE,
)

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
    time_since_last_refuel_min: float | None = None,   # НОВЫЙ параметр
) -> int:
    """
    Сохраняет событие + точки окна.
    Возвращает event_id (или -1 при ошибке).
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                event_label = 1 if user_verdict == "real" else 0

                max_drop = _safe_float(abs(window_df["fuel_diff"].min())) if window_df is not None and "fuel_diff" in window_df else 0.0
                drop_duration = _calculate_drop_duration(window_df) if window_df is not None else 0.0
                recovery_ratio = _calculate_recovery_ratio(window_df) if window_df is not None else 0.0

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
                    _safe_float(time_since_last_refuel_min),   # ← БЫЛО None, ТЕПЕРЬ расчёт
                    recovery_ratio,
                    hour_of_day,
                    day_of_week,
                    fuel_before,
                    fuel_after,
                ))
                event_id = cur.fetchone()[0]

                if window_df is not None and not window_df.empty:
                    points_rows = []
                    for _, row in window_df.iterrows():
                        label = _label_point(row, user_verdict)
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
        return event_id          # ← возвращаем ID
    except Exception as e:
        logger.error(f"Ошибка записи feedback: {e}")
        return -1                # ← маркер ошибки


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

def detect_last_refuel(
    df_telemetry: pd.DataFrame,
    event_time,
    refuel_threshold_l: float = 5.0,        # суммарный рост для заправки
    max_speed_for_refuel: float = 2.0,      # стоянка (отсекает плескание)
    stability_ratio: float = 0.5,           # необратимость (отсекает наклон)
    stability_window_min: float = 30.0,     # окно проверки стабилизации
    growth_window_min: float = 10.0,        # окно, за которое считаем рост
) -> float:
    """
    Определяет минуты с последней заправки до события.
    Заправка = суммарный рост уровня за growth_window_min + стоянка + необратимость.
    Возвращает 999.0, если заправки не было.
    """
    from datetime import timedelta

    if df_telemetry is None or df_telemetry.empty or "fuel_lvl" not in df_telemetry:
        return 999.0

    event_time = pd.to_datetime(event_time)
    history = df_telemetry[df_telemetry["datetime"] < event_time].copy()

    if len(history) < 3:
        return 999.0

    history = history.sort_values("datetime").set_index("datetime")

    # 🔧 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: рост от МИНИМУМА за скользящее окно
    # Для заправки это даст суммарные 40 л, а не 0.9 л на точку
    rolling_min = history["fuel_lvl"].rolling(f"{int(growth_window_min)}min").min()
    history["fuel_growth"] = history["fuel_lvl"] - rolling_min

    refuel_times = []

    for idx, row in history.iterrows():
        # Условие 1: значительный суммарный рост
        growth = row["fuel_growth"]
        if pd.isna(growth) or growth < refuel_threshold_l:
            continue

        # Условие 2: стоянка (отсекает плескание при движении)
        speed = row.get("speed", 0)
        if pd.notna(speed) and speed > max_speed_for_refuel:
            continue

        # Условие 3: необратимость (отсекает наклон)
        future = history[
            (history.index > idx)
            & (history.index <= idx + timedelta(minutes=stability_window_min))
        ]
        if not future.empty:
            min_after = future["fuel_lvl"].min()
            if (row["fuel_lvl"] - min_after) > (growth * stability_ratio):
                continue  # уровень упал обратно → наклон, не заправка

        refuel_times.append(idx)

    if not refuel_times:
        return 999.0

    last_refuel = max(refuel_times)
    return float((event_time - last_refuel).total_seconds() / 60)


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
    
INSERT_CONTEXT_SQL = """
INSERT INTO telemetry_context (
    event_id, datetime, fuel_level, speed, alt, power_out, power_in,
    gsm_lvl, gps_sensor,
    feat_fuel_drop_rate, feat_fuel_volatility_15min, feat_drawdown_15min,
    feat_drawdown_30min, feat_total_drop_10min, feat_drop_duration_min,
    feat_drop_consistency, feat_consecutive_drops, feat_is_large_gap,
    feat_monotonic_drop_length, feat_time_diff_min, feat_is_gnss_anomaly,
    feat_is_gsm_weak, feat_is_connection_lost, feat_is_engine_on,
    is_anomaly_ml, anomaly_score_ml, fuel_diff, time_diff_min
) VALUES %s
"""


def save_telemetry_context(
    event_id: int,
    df_telemetry: pd.DataFrame,
    event_time,
    event_idx: int,
) -> bool:
    """
    Сохраняет полную телеметрию в telemetry_context.
    Гибридное окно: max([-24ч, +3ч], [event_idx-60, event_idx+60]).
    """
    from datetime import timedelta
    from config import (
        CONTEXT_HOURS_BEFORE, CONTEXT_HOURS_AFTER,
        CONTEXT_ROWS_BEFORE, CONTEXT_ROWS_AFTER,
    )

    if event_id < 0 or df_telemetry is None or df_telemetry.empty:
        return False

    try:
        event_time = pd.to_datetime(event_time)

        # Временное окно
        time_start = event_time - timedelta(hours=CONTEXT_HOURS_BEFORE)
        time_end = event_time + timedelta(hours=CONTEXT_HOURS_AFTER)

        # Количественное окно
        row_start = max(0, event_idx - CONTEXT_ROWS_BEFORE)
        row_end = min(len(df_telemetry) - 1, event_idx + CONTEXT_ROWS_AFTER)

        # Гибрид: берём максимум покрытия
        actual_start = min(time_start, df_telemetry.iloc[row_start]["datetime"])
        actual_end = max(time_end, df_telemetry.iloc[row_end]["datetime"])

        ctx = df_telemetry[
            (df_telemetry["datetime"] >= actual_start)
            & (df_telemetry["datetime"] <= actual_end)
        ].copy()

        if ctx.empty:
            return False

        rows = []
        for _, r in ctx.iterrows():
            rows.append((
                event_id,
                r.get("datetime"),
                _safe_float(r.get("fuel_lvl")),
                _safe_float(r.get("speed")),
                _safe_float(r.get("alt")),
                _safe_float(r.get("power_out")),
                _safe_float(r.get("power_in")),
                _safe_float(r.get("gsm_lvl")),
                _to_native_int(r.get("gps_sensor")),
                _safe_float(r.get("fuel_drop_rate")),
                _safe_float(r.get("fuel_volatility_15min")),
                _safe_float(r.get("drawdown_15min")),
                _safe_float(r.get("drawdown_30min")),
                _safe_float(r.get("total_drop_10min")),
                _safe_float(r.get("drop_duration_min")),
                _safe_float(r.get("drop_consistency")),
                _safe_float(r.get("consecutive_drops")),
                _safe_float(r.get("is_large_gap")),
                _safe_float(r.get("monotonic_drop_length")),
                _safe_float(r.get("time_diff_min")),
                _safe_float(r.get("is_gnss_anomaly")),
                _safe_float(r.get("is_gsm_weak")),
                _safe_float(r.get("is_connection_lost")),
                _safe_float(r.get("is_engine_on")),
                _to_native_int(r.get("is_anomaly")),
                _safe_float(r.get("anomaly_score")),
                _safe_float(r.get("fuel_diff")),
                _safe_float(r.get("time_diff_min")),
            ))

        with _get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, INSERT_CONTEXT_SQL, rows, page_size=500)
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка записи telemetry_context: {e}")
        return False


def _to_native_int(val):
    """Безопасное преобразование в int (обрабатывает np.int64, nan, None)."""
    if val is None or pd.isna(val):
        return None
    return int(val)
