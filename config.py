from pathlib import Path
import streamlit as st
import os

# ============================================================
# ПУТИ
# ============================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "isolation_forest_model.pkl"
FEATURES_PATH = MODELS_DIR / "features_list.pkl"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

# ============================================================
# API FortMonitor
# ============================================================
API_BASE_URL = "https://glonassagro.com/"

# Читаем из secrets.toml (облако/локалка) или из переменных окружения
API_USERNAME = st.secrets.get("API_USERNAME", os.getenv("API_USERNAME", ""))
API_PASSWORD = st.secrets.get("API_PASSWORD", os.getenv("API_PASSWORD", ""))

if not API_USERNAME or not API_PASSWORD:
    st.error("❌ API credentials not configured! Please set secrets.")
    st.stop()


# ============================================================
# PostgreSQL (для фидбека)
# ============================================================
PG_HOST = st.secrets.get("PG_HOST", os.getenv("PG_HOST", "localhost"))
PG_PORT = int(st.secrets.get("PG_PORT", os.getenv("PG_PORT", "5432")))
PG_DB = st.secrets.get("PG_DB", os.getenv("PG_DB", "drain_feedback"))
PG_USER = st.secrets.get("PG_USER", os.getenv("PG_USER", "drain_app"))
PG_PASSWORD = st.secrets.get("PG_PASSWORD", os.getenv("PG_PASSWORD", ""))

if not PG_PASSWORD:
    st.error("❌ PostgreSQL credentials not configured! Please set secrets.")
    st.stop()


# Проверка на случай, если секреты не настроены
if not API_USERNAME or not API_PASSWORD:
    st.error(" API credentials not configured! Please set secrets.")
    st.stop()

# ============================================================
# МАППИНГ ДАТЧИКОВ (fallback-списки названий)
# ============================================================
# Для каждого нужного признака — список возможных названий в API FortMonitor.
# Поиск идёт по порядку: первое найденное название будет использовано.
# Если ни одно не найдено — датчик считается отсутствующим.

SENSOR_MAPPING = {
    # КРИТИЧНЫЕ ДАТЧИКИ (без них анализ невозможен)
    "fuel_lvl": [
        "Уровень топлива",
        "ДУТ",
        "Сумматор",
        "Сумматор уровня топлива",
        "Сумматор ДУТ",
        "Fuel Level",
        "Топливо",
        "Сумматор датчиков уровня топлива",
    ],
    "speed": [
        "Скорость",
        "Скорость (км/ч)",
        "Speed",
    ],
    
    # ВАЖНЫЕ, НО НЕ КРИТИЧНЫЕ (без них используем дефолты)
    "gps_sensor": [
        "Датчик GPS/ГЛОНАСС",
        "GPS/ГЛОНАСС",
        "Количество спутников",
        "GPS",
        "ГЛОНАСС",
    ],
    "gsm_lvl": [
        "Уровень сигнала GSM (%)",
        "Уровень GSM",
        "GSM",
        "Сигнал GSM",
    ],
    "alt": [
        "Высота над уровнем моря",
        "Высота над уровнем моря (м)",
        "Высота",
        "Altitude",
    ],
    "power_out": [
        "Внешнее питание",
        "Внешнее питание (В)",
        "Борт",
        "Напряжение борта",
    ],
    "power_in": [
        "Внутреннее питание",
        "Внутреннее питание (В)",
        "Внутреннее",
        "Напряжение внутреннее",
    ],
}

# Критичные датчики — без них анализ невозможен
CRITICAL_SENSORS = ["fuel_lvl", "speed"]

# Опциональные датчики — без них используем дефолты
OPTIONAL_SENSORS = ["gps_sensor", "gsm_lvl", "alt", "power_out", "power_in"]

# Все датчики для запроса к API
ALL_SENSORS = CRITICAL_SENSORS + OPTIONAL_SENSORS

# ============================================================
# ОКНА АНАЛИЗА (как в ноутбуке)
# ============================================================
# Временное окно: [-10 мин, +20 мин] от события (cell 124 в ноутбуке)
WINDOW_BEFORE_MIN = 10
WINDOW_AFTER_MIN = 20

# Минимальное количество точек во временном окне
# Если меньше — переключаемся на строковое окно (для ночных сливов)
MIN_POINTS_IN_TIME_WINDOW = 10

# Строковое окно: ±30 строк от события (cell 224 в ноутбуке)
ROW_WINDOW_SIZE = 30

# Расширенное строковое окно: для ночного слива (gap_drawdown)
EXTENDED_WINDOW_ROWS = 60

# Параметры телеметрии
EXPAND_HOURS_BEFORE = 3
EXPAND_HOURS_AFTER = 3

# ============================================================
# ПОРОГИ ЭВРИСТИК (как в ноутбуке, cell 201/224)
# ============================================================
SLOW_DRAIN_THRESHOLD_L = 3.5
NIGHT_DRAIN_THRESHOLD_L = 5.0
SUSPICIOUS_TOTAL_DROP_L = 3.0