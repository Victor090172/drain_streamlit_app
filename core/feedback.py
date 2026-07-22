# -*- coding: utf-8 -*-

"""Сбор обратной связи от пользователей для будущего дообучения."""
from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path
from typing import Dict

import pandas as pd

from config import FEEDBACK_PATH

_lock = threading.Lock()


def _ensure_file(path: Path) -> None:
    if not path.exists():
        path.write_text("")


def save_feedback(event_info: Dict, user_verdict: str, features_row: pd.Series) -> None:
    """
    Дописывает одну строку в feedback.csv (lock-safe).
    user_verdict: "real" | "false"
    """
    _ensure_file(FEEDBACK_PATH)

    row = {
        "feedback_ts": dt.datetime.now().isoformat(timespec="seconds"),
        "object_name": event_info.get("object_name", ""),
        "event_time": event_info.get("event_time", ""),
        "address": event_info.get("address", ""),
        "user_verdict": user_verdict,                 # real / false
        "system_label": event_info.get("label", ""),
        "ml_detected": int(event_info.get("ml_detected", False)),
        "rule_detected": int(event_info.get("rule_detected", False)),
        "rule_reason": event_info.get("rule_reason", ""),
        "ml_score_min": event_info.get("ml_score_min", 0.0),
        "anomaly_points_count": event_info.get("anomaly_points_count", 0),
        "total_drop": event_info.get("total_drop", 0.0),
        "gap_drop": event_info.get("gap_drop", 0.0),
    }
    # Разворачиваем признаки в отдельные колонки — удобно для CatBoost
    for feat, val in features_row.items():
        row[f"feat__{feat}"] = val

    new_row_df = pd.DataFrame([row])

    with _lock:
        write_header = not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0
        new_row_df.to_csv(
            FEEDBACK_PATH,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )