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

def post_process_verdict(
    verdict: dict,
    main_window: pd.DataFrame,
    speed_threshold_kmh: float = 15.0,
) -> dict:
    """
    Пост-обработка вердикта: скоростной фильтр + информативное пояснение.
    Если техника двигалась со скоростью выше порога — помечаем как
    'ВЕРОЯТНО ЛОЖНЫЙ СЛИВ (движение)' с пояснением.
    """
    v = dict(verdict)  # копия, чтобы не менять оригинал

    # ---------- Расчёт характеристик движения в окне ----------
    avg_speed = 0.0
    moving_share = 0.0
    if main_window is not None and "speed" in main_window and len(main_window) > 0:
        speeds = main_window["speed"].fillna(0)
        avg_speed = float(speeds.mean())
        moving_share = float((speeds > 2.0).mean())  # доля времени в движении

    v["avg_speed"] = round(avg_speed, 1)
    v["moving_share"] = round(moving_share, 2)

    # ---------- Сбор дополнительных замечаний (РЭБ, связь) ----------
    extra_notes = []
    if main_window is not None and len(main_window) > 0:
        if "is_gnss_anomaly" in main_window:
            gnss_anomalies = int(main_window["is_gnss_anomaly"].fillna(0).sum())
            if gnss_anomalies > 0:
                extra_notes.append(
                    f"Обнаружены аномалии GPS/ГЛОНАСС ({gnss_anomalies} раз) — "
                    f"возможны искажения от РЭБ."
                )
        if "is_connection_lost" in main_window:
            conn_lost = int(main_window["is_connection_lost"].fillna(0).sum())
            if conn_lost > 0:
                extra_notes.append(f"Зафиксированы разрывы связи ({conn_lost} раз).")

    # ---------- Применение скоростного фильтра ----------
    is_moving = avg_speed > speed_threshold_kmh
    v["speed_filter_applied"] = is_moving

    if is_moving:
        # Меняем вердикт только если система подозревала слив
        original_label = v.get("label", "")
        if "ЛОЖНЫЙ" not in original_label:
            v["label"] = "ВЕРОЯТНО ЛОЖНЫЙ СЛИВ (движение)"
            v["rule_detected"] = True

            reason_parts = [
                f"Техника двигалась со средней скоростью {avg_speed:.1f} км/ч "
                f"(в движении {moving_share*100:.0f}% времени). "
                f"Слив в движении маловероятен — вероятна болтанка топлива в баке."
            ]
            reason_parts.extend(extra_notes)
            v["rule_reason"] = " ".join(reason_parts)
        else:
            # Уже ложный — дополняем пояснение скоростью
            existing = v.get("rule_reason") or ""
            speed_note = f"Техника двигалась ({avg_speed:.1f} км/ч)."
            v["rule_reason"] = (existing + " " + speed_note + " ".join(extra_notes)).strip()
    else:
        # Стоянка: если есть замечания по РЭБ — дополняем пояснение
        if extra_notes and v.get("rule_reason") is not None:
            v["rule_reason"] = (v.get("rule_reason") + " " + " ".join(extra_notes)).strip()

    return v

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
