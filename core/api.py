"""API-клиент FortMonitor и функции загрузки телеметрии."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import numpy as np
import streamlit as st
import logging
import urllib.parse

from config import API_BASE_URL, API_USERNAME, API_PASSWORD, TARGET_SENSORS

logger = logging.getLogger(__name__)

class FortMonitorClient:
    def __init__(self, base_url: str, username: str, password: str,
                 api_version: str = "api/integration/v1/"):
        self.base_url = base_url.rstrip("/") + "/"
        self.api_version = api_version
        self.username = username
        self.password = password
        self._client = httpx.Client(timeout=60.0)
        self._session_id: Optional[str] = None
        self._last_auth_time: Optional[dt.datetime] = None

    def _get_headers(self) -> Dict[str, str]:
        return {"SessionId": self._session_id} if self._session_id else {}

    def ensure_session(self) -> None:
        now = dt.datetime.now()
        if self._session_id is None or self._last_auth_time is None:
            self._authenticate()
        elif (now - self._last_auth_time) > dt.timedelta(minutes=25):
            self._authenticate()

    def _authenticate(self) -> None:
        url = f"{self.base_url}{self.api_version}connect"
        data = {"login": self.username, "password": self.password,
                "lang": "ru-ru", "timezone": "0"}
        response = self._client.get(url, params=data)
        response.raise_for_status()
        self._session_id = response.headers.get("SessionId")
        date_str = response.headers.get("date", "")
        if date_str:
            sid_date = dt.datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
            self._last_auth_time = sid_date + dt.timedelta(hours=3)
        else:
            self._last_auth_time = dt.datetime.now()

    def find_object(self, object_name: str) -> Optional[str]:
        """Ищет ID объекта по имени. Возвращает ID или None."""
        self.ensure_session()
        url = f"{self.base_url}{self.api_version}object/find"
        
        #  Явно кодируем имя объекта для URL
        encoded_name = urllib.parse.quote(object_name, safe='')
        
        # 🔧 Логирование для отладки
        logger.info(f"🔍 Поиск объекта: '{object_name}' (encoded: '{encoded_name}')")
        
        response = self._client.post(
            url, 
            headers=self._get_headers(),
            params={"paramName": "name", "paramValue": object_name},
        )
        
        logger.info(f"📡 Ответ API: status={response.status_code}")
        
        data = response.json()
        if data.get("objects"):
            object_id = str(data["objects"][0]["id"])
            logger.info(f"✅ Объект найден: ID={object_id}")
            return object_id
        
        logger.warning(f"❌ Объект '{object_name}' не найден в FortMonitor")
        return None

    def get_sensor_params(self, object_id: str, target_sensors: List[str]) -> Optional[str]:
        self.ensure_session()
        url = f"{self.base_url}{self.api_version}objsensorslist"
        response = self._client.get(url, headers=self._get_headers(),
                                    params={"oid": object_id})
        sensors = response.json().get("obj_sensors", [])
        param_ids = []
        for t_name in target_sensors:
            for sensor in sensors:
                if sensor.get("name") == t_name:
                    sid = sensor.get("sid", 0)
                    pid = sensor.get("pid", 0)
                    param_ids.append(f"s{sid}" if sid > 0 else f"p{pid}")
                    break
        return ",".join(param_ids) if param_ids else None

    def get_telemetry(self, object_id: str, slist: str,
                      date_from: str, date_to: str) -> Dict[str, Any]:
        self.ensure_session()
        url = f"{self.base_url}{self.api_version}objdata"
        params = {"oid": object_id, "slist": slist, "from": date_from, "to": date_to}
        response = self._client.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()


@st.cache_resource
def get_api_client() -> FortMonitorClient:
    """Singleton API-клиента на всё приложение."""
    return FortMonitorClient(API_BASE_URL, API_USERNAME, API_PASSWORD)


def process_telemetry_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Сырой ответ API → чистый DataFrame (1:1 из notebook)."""
    col_names = ["Время"] + raw_data.get("column_names", [])
    records = raw_data.get("obj_data", {}).get("records", [])
    df = pd.DataFrame(records, columns=col_names)

    df["Время"] = pd.to_datetime(df["Время"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["Время"] = df["Время"].dt.tz_localize("UTC").dt.tz_convert("Europe/Moscow")

    num_cols = [c for c in df.columns if c != "Время"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fuel_col = next((c for c in df.columns if "Уровень топлива" in c), None)
    if fuel_col:
        df[fuel_col] = np.ceil(df[fuel_col] * 10) / 10
    return df


@st.cache_data(show_spinner="📡 Загружаем телеметрию через API...")
def fetch_telemetry_for_object(object_name: str,
                               min_time_iso: str,
                               max_time_iso: str) -> pd.DataFrame:
    """
    Тянет телеметрию за окно [min-3h, max+3h] для одного объекта.
    🔧 FIX: Корректно обрабатывает часовые пояса при запросе.
    """
    client = get_api_client()
    object_id = client.find_object(object_name)
    if not object_id:
        raise ValueError(f"Объект '{object_name}' не найден в FortMonitor")

    slist = client.get_sensor_params(object_id, TARGET_SENSORS)
    if not slist:
        raise ValueError(f"Не удалось найти датчики для '{object_name}'")

    min_time = pd.to_datetime(min_time_iso)
    max_time = pd.to_datetime(max_time_iso)

    # 🔧 FIX: Преобразуем в UTC для корректного запроса
    if min_time.tz is not None:
        min_time_utc = min_time.tz_convert('UTC')
    else:
        min_time_utc = min_time.tz_localize('Europe/Moscow').tz_convert('UTC')

    if max_time.tz is not None:
        max_time_utc = max_time.tz_convert('UTC')
    else:
        max_time_utc = max_time.tz_localize('Europe/Moscow').tz_convert('UTC')

    # Теперь запрашиваем в UTC
    expanded_min = (min_time_utc - pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    expanded_max = (max_time_utc + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")

    raw = client.get_telemetry(object_id, slist, expanded_min, expanded_max)
    df = process_telemetry_data(raw)
    df["object_id"] = object_id
    return df

