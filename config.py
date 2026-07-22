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
API_USERNAME = st.secrets.get("API_USERNAME") or os.getenv("API_USERNAME")
API_PASSWORD = st.secrets.get("API_PASSWORD") or os.getenv("API_PASSWORD")

# Проверка на случай, если секреты не настроены
if not API_USERNAME or not API_PASSWORD:
    st.error(" API credentials not configured! Please set secrets.")
    st.stop()

TARGET_SENSORS = [
    "Внешнее питание", "Внутреннее питание", "Высота над уровнем моря",
    "Датчик GPS/ГЛОНАСС", "Скорость", "Уровень сигнала GSM", "Уровень топлива",
]

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