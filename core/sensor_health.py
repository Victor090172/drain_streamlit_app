"""
Оценка качества показаний датчика уровня топлива (ДУТ).

Этап 1: информирование пользователя + накопление статистики.
НЕ интегрируется в детекцию сливов — только наблюдение.

Анализирует телеметрию, загруженную для анализа сливов (-24ч/+3ч),
и выдаёт интегральную оценку здоровья датчика + рекомендации.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.feedback import _get_connection

logger = logging.getLogger(__name__)

# Веса аспектов в интегральной оценке (сумма = 1.0)
WEIGHTS = {
    "noise_score": 0.30,        # шум/волатильность
    "continuity_score": 0.25,   # непрерывность данных
    "plausibility_score": 0.20, # физическая правдоподобность
    "stability_score": 0.25,    # отсутствие резких скачков
}


def analyze_sensor_health(df_feat: pd.DataFrame) -> dict:
    """
    Анализирует качество данных ДУТ по загруженной телеметрии.
    Возвращает словарь сырых метрик или None если данных нет.
    """
    if df_feat is None or df_feat.empty:
        return None

    m = {}

    # ---------- 1. Шум / волатильность ----------
    if "fuel_volatility_15min" in df_feat.columns:
        vol = df_feat["fuel_volatility_15min"].replace([np.inf, -np.inf], np.nan).dropna()
        m["avg_volatility"] = float(vol.mean()) if len(vol) > 0 else 0.0
        m["max_volatility"] = float(vol.max()) if len(vol) > 0 else 0.0
    else:
        m["avg_volatility"] = 0.0
        m["max_volatility"] = 0.0

    # ---------- 2. Резкие скачки (статистический подход, устойчив к выбросам) ----------
    if "fuel_diff" in df_feat.columns:
        diffs = df_feat["fuel_diff"].replace([np.inf, -np.inf], np.nan).dropna().abs()
        if len(diffs) > 0:
            median_diff = float(diffs.median())
            mad = float((diffs - median_diff).abs().median())
            # Порог: медиана + 5*MAD, но не меньше 3 л (заправка/слив дают больше)
            spike_threshold = max(median_diff + 5 * mad, 3.0)
            m["spike_count"] = int((diffs > spike_threshold).sum())
            m["spike_threshold"] = round(spike_threshold, 2)
        else:
            m["spike_count"] = 0
            m["spike_threshold"] = 3.0
    else:
        m["spike_count"] = 0
        m["spike_threshold"] = 3.0

    # ---------- 3. Непрерывность данных ----------
    m["connection_lost_count"] = int(df_feat["is_connection_lost"].sum()) \
        if "is_connection_lost" in df_feat.columns else 0
    if "time_diff_min" in df_feat.columns:
        td = df_feat["time_diff_min"].replace([np.inf, -np.inf], np.nan).dropna()
        m["max_time_gap_min"] = float(td.max()) if len(td) > 0 else 0.0
        m["large_gap_count"] = int((td > 30).sum()) if len(td) > 0 else 0
    else:
        m["max_time_gap_min"] = 0.0
        m["large_gap_count"] = 0

    # ---------- 4. Аномалии GPS / РЭБ ----------
    m["gnss_anomaly_count"] = int(df_feat["is_gnss_anomaly"].sum()) \
        if "is_gnss_anomaly" in df_feat.columns else 0

    # ---------- 5. Физическая правдоподобность (эвристика ёмкости) ----------
    if "fuel_lvl" in df_feat.columns:
        fuel = df_feat["fuel_lvl"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(fuel) > 0:
            m["min_fuel_level"] = float(fuel.min())
            m["max_fuel_level"] = float(fuel.max())
            # Эвристика: ёмкость ≈ максимальный наблюдаемый уровень + 5% запас
            m["estimated_capacity"] = round(m["max_fuel_level"] * 1.05, 1)
            m["negative_level_count"] = int((fuel < 0).sum())
            m["over_capacity_count"] = int((fuel > m["estimated_capacity"] * 1.1).sum())
        else:
            m.update({
                "min_fuel_level": 0.0, "max_fuel_level": 0.0,
                "estimated_capacity": 0.0, "negative_level_count": 0,
                "over_capacity_count": 0,
            })
    else:
        m.update({
            "min_fuel_level": 0.0, "max_fuel_level": 0.0,
            "estimated_capacity": 0.0, "negative_level_count": 0,
            "over_capacity_count": 0,
        })

    # ---------- 6. Общий объём данных ----------
    m["data_points"] = len(df_feat)
    if "datetime" in df_feat.columns and len(df_feat) > 1:
        period = (df_feat["datetime"].max() - df_feat["datetime"].min()).total_seconds() / 3600
        m["period_hours"] = round(period, 1)
        m["period_start"] = df_feat["datetime"].min()
        m["period_end"] = df_feat["datetime"].max()
    else:
        m["period_hours"] = 0.0
        m["period_start"] = None
        m["period_end"] = None

    return m


def compute_health_scores(metrics: dict) -> dict:
    """
    Сводит сырые метрики в суб-скоры 0-100 (выше = лучше)
    и интегральную оценку overall_health.
    """
    s = {}

    # Шум: средняя волатильность 5 л и выше → 0 баллов
    avg_vol = metrics.get("avg_volatility", 0.0)
    s["noise_score"] = round(max(0.0, 100.0 - avg_vol * 20.0), 1)

    # Непрерывность: доля потерянных точек, 20% и выше → 0 баллов
    lost = metrics.get("connection_lost_count", 0)
    total = max(metrics.get("data_points", 1), 1)
    lost_ratio = lost / total
    s["continuity_score"] = round(max(0.0, 100.0 - lost_ratio * 500.0), 1)

    # Правдоподобность: штраф за отрицательные уровни и выбросы выше ёмкости
    implausible = metrics.get("negative_level_count", 0) + metrics.get("over_capacity_count", 0)
    s["plausibility_score"] = round(max(0.0, 100.0 - implausible * 10.0), 1)

    # Стабильность: штраф за резкие скачки
    spikes = metrics.get("spike_count", 0)
    s["stability_score"] = round(max(0.0, 100.0 - spikes * 5.0), 1)

    # Интегральная оценка (взвешенная сумма)
    overall = sum(s[k] * w for k, w in WEIGHTS.items())
    s["overall_health"] = round(overall, 1)

    return s


def get_recommendations(metrics: dict, scores: dict) -> list:
    """Формирует текстовые рекомендации на основе метрик."""
    recs = []

    if scores.get("noise_score", 100) < 60:
        recs.append("Высокая волатильность уровня — проверьте настройку фильтрации ДУТ.")

    if scores.get("stability_score", 100) < 60:
        recs.append(
            f"Зафиксировано {metrics.get('spike_count', 0)} резких скачков уровня — "
            f"возможна болтанка или неисправность датчика."
        )

    if scores.get("continuity_score", 100) < 60:
        recs.append("Частые разрывы связи — проверьте GSM-антенну и питание трекера.")

    if metrics.get("gnss_anomaly_count", 0) > 5:
        recs.append(
            f"Много аномалий GPS ({metrics.get('gnss_anomaly_count', 0)}) — "
            f"возможны помехи (РЭБ) или проблемы с антенной."
        )

    if metrics.get("negative_level_count", 0) > 0:
        recs.append("Зафиксированы отрицательные значения уровня — требуется калибровка ДУТ.")

    if metrics.get("max_time_gap_min", 0) > 120:
        recs.append(
            f"Большой разрыв в данных ({metrics.get('max_time_gap_min', 0):.0f} мин) — "
            f"возможна потеря питания или связи."
        )

    if not recs:
        recs.append("Датчик работает стабильно, замечаний нет.")

    return recs


def evaluate_sensor(df_feat: pd.DataFrame) -> dict:
    """
    Полный цикл оценки: метрики → скоры → рекомендации.
    Возвращает единый словарь или None.
    """
    metrics = analyze_sensor_health(df_feat)
    if metrics is None:
        return None
    scores = compute_health_scores(metrics)
    recommendations = get_recommendations(metrics, scores)

    result = {**metrics, **scores, "recommendations": recommendations}
    return result


def save_sensor_health(object_name: str, result: dict) -> bool:
    """Сохраняет результат оценки в БД для накопления статистики."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sensor_health_log (
                        object_name, period_start, period_end,
                        data_points, period_hours,
                        noise_score, continuity_score, plausibility_score,
                        stability_score, overall_health,
                        avg_volatility, max_volatility, spike_count,
                        connection_lost_count, large_gap_count, max_time_gap_min,
                        gnss_anomaly_count, min_fuel_level, max_fuel_level,
                        estimated_capacity, negative_level_count, recommendations
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    object_name,
                    result.get("period_start"), result.get("period_end"),
                    result.get("data_points"), result.get("period_hours"),
                    result.get("noise_score"), result.get("continuity_score"),
                    result.get("plausibility_score"), result.get("stability_score"),
                    result.get("overall_health"),
                    result.get("avg_volatility"), result.get("max_volatility"),
                    result.get("spike_count"), result.get("connection_lost_count"),
                    result.get("large_gap_count"), result.get("max_time_gap_min"),
                    result.get("gnss_anomaly_count"), result.get("min_fuel_level"),
                    result.get("max_fuel_level"), result.get("estimated_capacity"),
                    result.get("negative_level_count"),
                    " | ".join(result.get("recommendations", [])),
                ))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Не удалось сохранить sensor_health_log: {e}")
        return False


def load_health_ranking(limit_objects: int = 50) -> pd.DataFrame:
    """
    Загружает рейтинг объектов по качеству датчиков (агрегация по последним оценкам).
    Возвращает DataFrame, отсортированный по overall_health.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH latest AS (
                        SELECT DISTINCT ON (object_name)
                            object_name, overall_health, created_at,
                            noise_score, continuity_score,
                            plausibility_score, stability_score
                        FROM sensor_health_log
                        ORDER BY object_name, created_at DESC
                    )
                    SELECT * FROM latest
                    ORDER BY overall_health ASC
                    LIMIT %s
                """, (limit_objects,))
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        logger.warning(f"Не удалось загрузить рейтинг здоровья датчиков: {e}")
        return pd.DataFrame()

