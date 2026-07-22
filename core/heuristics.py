"""Эвристические правила + гибридный вердикт.
Воспроизводит логику из ноутбука (cell 201, 224) с адаптивным окном.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from config import (
    SLOW_DRAIN_THRESHOLD_L,
    NIGHT_DRAIN_THRESHOLD_L,
    SUSPICIOUS_TOTAL_DROP_L,
)


def check_slow_drain(window: pd.DataFrame) -> tuple[bool, str, float]:
    """Медленный слив: стоянка + хороший GPS + просадка > порога за 10 мин.
    Как в ноутбуке cell 201.
    
    🔧 ЗАЩИТА ОТ "ЭХА": Проверяем, что падение продолжается в последние 5 точках,
    а не было в прошлом (иначе срабатывает на уже завершившемся сливе).
    """
    mask = (window["speed"] == 0) & (window["is_gnss_anomaly"] == 0)
    subset = window[mask]
    if subset.empty or "total_drop_10min" not in subset.columns:
        return False, "", 0.0
    
    # 🔧 ПРОВЕРКА: Падение продолжается в последние 5 точках
    last_5 = subset.tail(5)
    has_recent_activity = (
        (last_5["fuel_diff"] < 0).any() | 
        (last_5["monotonic_drop_length"] > 0).any()
    )
    
    if not has_recent_activity:
        # Слив уже закончился — не считаем его "медленным сливом"
        return False, "", 0.0
    
    max_drop = float(subset["total_drop_10min"].max())
    if max_drop > SLOW_DRAIN_THRESHOLD_L:
        return True, f"Медленный слив: просадка {max_drop:.1f} л за 10 мин при стоянке", max_drop
    return False, "", max_drop


def check_night_drain(window: pd.DataFrame) -> tuple[bool, str, float]:
    """Ночной слив: большой разрыв во времени + падение уровня.
    Как в ноутбуке cell 224.
    """
    if "gap_drawdown" not in window.columns:
        return False, "", 0.0
    max_gap = float(window["gap_drawdown"].max())
    if max_gap > NIGHT_DRAIN_THRESHOLD_L:
        return True, f"Ночной слив: падение на {max_gap:.1f} л за время простоя", max_gap
    return False, "", max_gap


def make_verdict(
    main_window: pd.DataFrame,
    extended_window: pd.DataFrame | None = None,
) -> Dict:
    """
    Гибридный вердикт. Воспроизводит логику из ноутбука cell 201/224.
    
    Логика:
      1. ML на main_window → если нашёл аномалии → "СЛИВ (ML)"
      2. Если ML молчит → медленный слив на main_window
      3. Если и это молчит → ночной слив на extended_window (gap_drawdown)
      4. Если всё молчит, но total_drop < -3 в main_window → "ПОДОЗРЕНИЕ"
      5. Иначе → "ЛОЖНЫЙ"
    """
    # --- 1. ML на основном окне ---
    anomaly_points = main_window[main_window["is_anomaly"] == -1]
    ml_detected = len(anomaly_points) > 0
    ml_score_min = float(main_window["anomaly_score"].min()) if not main_window.empty else 0.0

    # --- 2. Медленный слив на основном окне ---
    rule_detected = False
    rule_reason = ""

    if not ml_detected:
        slow_detected, slow_reason, slow_drop = check_slow_drain(main_window)
        if slow_detected:
            rule_detected, rule_reason = True, slow_reason

    # --- 3. Ночной слив на расширенном окне ---
    gap_drop = 0.0
    if not ml_detected and not rule_detected and extended_window is not None:
        night_detected, night_reason, gap_drop = check_night_drain(extended_window)
        if night_detected:
            rule_detected, rule_reason = True, night_reason

    # --- Суммарное падение в основном окне ---
    fuel_drops = main_window[main_window["fuel_diff"] < 0]
    total_drop = float(fuel_drops["fuel_diff"].sum()) if not fuel_drops.empty else 0.0


 
    # --- 4-5. Финальная метка ---
    if ml_detected:
        label = "СЛИВ (ML)"
    elif rule_detected:
        label = "ПОДОЗРЕНИЕ НА СЛИВ (медленный слив/ночной слив)"
    elif total_drop < -SUSPICIOUS_TOTAL_DROP_L:
        label = "ПОДОЗРЕНИЕ НА СЛИВ (падение уровня)"
    else:
        label = "ЛОЖНЫЙ СЛИВ"

    return {
        "label": label,
        "ml_detected": ml_detected,
        "rule_detected": rule_detected,
        "rule_reason": rule_reason,
        "ml_score_min": ml_score_min,
        "anomaly_points_count": int(len(anomaly_points)),
        "total_drop": total_drop,
        "gap_drop": gap_drop,
    }