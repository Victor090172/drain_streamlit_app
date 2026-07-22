"""Feature engineering — 1:1 из notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Время": "datetime",
        "Внешнее питание (В)": "power_out",
        "Внутреннее питание (В)": "power_in",
        "Высота над уровнем моря (м)": "alt",
        "Датчик GPS/ГЛОНАСС": "gps_sensor",
        "Скорость (км/ч)": "speed",
        "Уровень топлива": "fuel_lvl",
        "Уровень сигнала GSM (%)": "gsm_lvl",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["object_id", "datetime"]).reset_index(drop=True)

    # 1. Базовые дельты
    df["time_diff_min"] = df.groupby("object_id")["datetime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0).clip(lower=0.01)
    df["fuel_diff"] = df.groupby("object_id")["fuel_lvl"].diff()
    df["fuel_drop_rate"] = (np.abs(df["fuel_diff"]) / df["time_diff_min"]).clip(upper=100).fillna(0)

    # 2. Монотонность падения
    df["fuel_decrease"] = (df.groupby("object_id")["fuel_lvl"].diff() < 0).astype(int)
    df["monotonic_group"] = (df["fuel_decrease"] == 0).astype(int).groupby(df["object_id"]).cumsum()

    # 3. Статистика окна
    df["fuel_volatility_15min"] = df.groupby("object_id")["fuel_lvl"].transform(
        lambda x: x.rolling(window=10, min_periods=3).std()
    ).fillna(0)
    df["max_fuel_15min"] = df.groupby("object_id")["fuel_lvl"].transform(
        lambda x: x.rolling(window=10, min_periods=3).max()
    ).fillna(df["fuel_lvl"])
    df["drawdown_15min"] = (df["max_fuel_15min"] - df["fuel_lvl"]).clip(lower=0)

    df["max_fuel_30min"] = df.groupby("object_id")["fuel_lvl"].transform(
        lambda x: x.rolling(window=20, min_periods=3).max()
    ).fillna(df["fuel_lvl"])
    df["drawdown_30min"] = (df["max_fuel_30min"] - df["fuel_lvl"]).clip(lower=0)

    df["total_drop_10min"] = df.groupby("object_id")["fuel_diff"].transform(
        lambda x: x.rolling(window=10, min_periods=3).apply(lambda y: -y[y < 0].sum(), raw=False)
    ).fillna(0)

    df["drop_duration_min"] = df.groupby(["object_id", "monotonic_group"])["time_diff_min"].transform("sum")
    df["drop_duration_min"] = np.where(df["fuel_decrease"] == 1, df["drop_duration_min"], 0)

    df["drop_std"] = df.groupby(["object_id", "monotonic_group"])["fuel_diff"].transform(
        lambda x: x.std() if len(x) > 1 else 0
    )
    df["drop_consistency"] = np.where(df["fuel_decrease"] == 1, df["drop_std"], 999)
    df["monotonic_drop_length"] = df.groupby(["object_id", "monotonic_group"])["fuel_decrease"].transform("cumsum")
    df["monotonic_drop_length"] = np.where(df["fuel_decrease"] == 1, df["monotonic_drop_length"], 0)

    # 4. РЭБ и разрывы
    df["is_large_gap"] = (df["time_diff_min"] > 5.0).astype(int)
    df["drop_group"] = (df["fuel_diff"] >= 0).astype(int).groupby(df["object_id"]).cumsum()
    df["consecutive_drops"] = df.groupby(["object_id", "drop_group"])["fuel_diff"].transform(
        lambda x: (x < 0).cumsum()
    )
    df["consecutive_drops"] = np.where(df["fuel_diff"] < 0, df["consecutive_drops"], 0)
    df["gap_drawdown"] = np.where(df["time_diff_min"] > 60.0, np.abs(df["fuel_diff"]), 0.0)

    # 5. Флаги РЭБ
    df["alt_diff"] = df.groupby("object_id")["alt"].diff().abs().fillna(0)
    df["is_gnss_anomaly"] = ((df["gps_sensor"] < 4) | (df["alt_diff"] > 500)).astype(int)
    df["is_gsm_weak"] = (df["gsm_lvl"] < 20).astype(int)
    df["is_connection_lost"] = (df["time_diff_min"] > 10).astype(int)

    if "ignition" in df.columns:
        df["is_engine_on"] = (df["ignition"] == 1).astype(int)
    else:
        df["is_engine_on"] = 0
        if "power_out" in df.columns:
            df["is_engine_on"] = np.where(df["power_out"] > 12.0, 1, df["is_engine_on"])
        if "speed" in df.columns:
            df["is_engine_on"] = np.where(df["speed"] > 3.0, 1, df["is_engine_on"])
        df["is_engine_on"] = df["is_engine_on"].astype(int)

    df = df.drop(columns=["fuel_decrease", "monotonic_group", "drop_group", "drop_std"], errors="ignore")
    return df

