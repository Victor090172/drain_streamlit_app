"""Парсинг Excel-отчётов о сливах."""
from __future__ import annotations

import io
import re
from typing import Tuple, Union

import pandas as pd


def _normalize_object_name(name: str) -> str:
    """
    Нормализует имя объекта: убирает невидимые символы, 
    неразрывные пробелы, приводит к нижнему регистру.
    """
    # Заменяем неразрывные пробелы (\xa0) и другие невидимые символы на обычные пробелы
    name = re.sub(r'[\xa0\u2000-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', ' ', name)
    # Убираем множественные пробелы
    name = re.sub(r'\s+', ' ', name).strip()
    # Приводим к нижнему регистру
    return name.lower()


def parse_drain_report(file_input: Union[str, bytes]) -> Tuple[str, pd.DataFrame]:
    """
    Извлекает и нормализует данные о сливах из Excel-отчёта.
    """
    if isinstance(file_input, (bytes, bytearray)):
        df = pd.read_excel(io.BytesIO(file_input))
    else:
        df = pd.read_excel(file_input)

    # 🔧 НОРМАЛИЗАЦИЯ ИМЕНИ ОБЪЕКТА
    raw_name = str(df["Отчет по топливу"].values[1])
    object_name = _normalize_object_name(raw_name)

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