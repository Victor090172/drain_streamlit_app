"""Парсинг Excel-отчётов о сливах."""
from __future__ import annotations

import io
from typing import Tuple, Union

import pandas as pd


def parse_drain_report(file_input: Union[str, bytes]) -> Tuple[str, pd.DataFrame]:
    """
    Извлекает и нормализует данные о сливах из Excel-отчёта.
    Принимает путь (str) или содержимое файла (bytes) — для совместимости
    с st.file_uploader.
    """
    if isinstance(file_input, (bytes, bytearray)):
        df = pd.read_excel(io.BytesIO(file_input))
    else:
        df = pd.read_excel(file_input)

    object_name = str(df["Отчет по топливу"].values[1]).lower()

    drop_to = max(df.index[df["Отчет по топливу"] == "Время"])
    drop_after = max(df.index[df["Отчет по топливу"] == "Итого"])

    df = df.loc[drop_to + 1: drop_after - 1].copy()
    df.dropna(axis="columns", how="all", inplace=True)
    if "Unnamed: 2" in df.columns:
        df.drop(columns="Unnamed: 2", inplace=True, errors="ignore")
    df.reset_index(drop=True, inplace=True)

    rename_map = {
        "Отчет по топливу": "Время",
        "Unnamed: 1": "Уровень до",
        "Unnamed: 3": "Слив",
        "Unnamed: 4": "Уровень после",
        "Unnamed: 5": "Адрес",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    df["Время"] = pd.to_datetime(df["Время"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["Время"] = df["Время"].dt.tz_localize("Europe/Moscow")

    df.dropna(subset=["Время", "Слив"], inplace=True)
    for col in ["Уровень до", "Уровень после", "Слив"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Уровень после"] != df["Уровень до"]].copy()
    df["Слив"] = df["Слив"].fillna(1000)

    return object_name, df

