"""Streamlit-приложение для детекции сливов топлива.
Воспроизводит логику из ноутбука с адаптивным окном для ночных сливов.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    WINDOW_BEFORE_MIN, WINDOW_AFTER_MIN,
    MIN_POINTS_IN_TIME_WINDOW, ROW_WINDOW_SIZE,
    EXTENDED_WINDOW_ROWS, FEEDBACK_PATH,
)
from core.api import fetch_telemetry_for_object
from core.telemetry import parse_drain_report
from core.model_router import get_model_router
from core.heuristics import make_verdict
from core.feedback import save_feedback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Детектор сливов топлива", layout="wide")


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _find_event_idx(df_feat: pd.DataFrame, event_time: pd.Timestamp) -> int:
    """Находит индекс строки, ближайшей к event_time."""
    if df_feat["datetime"].dt.tz is None and event_time.tz is not None:
        event_time = event_time.tz_localize(None)
    elif df_feat["datetime"].dt.tz is not None and event_time.tz is None:
        event_time = event_time.tz_localize(df_feat["datetime"].dt.tz)
    diffs = (df_feat["datetime"] - event_time).abs()
    return int(diffs.idxmin())


def _build_adaptive_window(df_feat: pd.DataFrame, event_time: pd.Timestamp, event_idx: int) -> tuple[pd.DataFrame, str]:
    """
    Адаптивное окно: как в ноутбуке.
    
    По умолчанию: временное окно [-10, +20] мин (cell 199 в ноутбуке).
    Если в нём мало точек (< 10): переключаемся на строковое окно ±30 строк (cell 224).
    
    Возвращает: (window, window_type) где window_type = 'time' или 'row'
    """
    # 1. Пытаемся построить временное окно
    if df_feat["datetime"].dt.tz is None and event_time.tz is not None:
        event_time_naive = event_time.tz_localize(None)
    elif df_feat["datetime"].dt.tz is not None and event_time.tz is None:
        event_time_naive = event_time.tz_localize(df_feat["datetime"].dt.tz)
    else:
        event_time_naive = event_time

    t_start = event_time_naive - pd.Timedelta(minutes=WINDOW_BEFORE_MIN)
    t_end = event_time_naive + pd.Timedelta(minutes=WINDOW_AFTER_MIN)
    mask = (df_feat["datetime"] >= t_start) & (df_feat["datetime"] <= t_end)
    time_window = df_feat[mask].copy()

    # 2. Проверяем, достаточно ли точек
    if len(time_window) >= MIN_POINTS_IN_TIME_WINDOW:
        return time_window, "time"
    
    # 3. Мало точек — переключаемся на строковое окно (для ночных сливов)
    start = max(0, event_idx - ROW_WINDOW_SIZE)
    end = min(len(df_feat) - 1, event_idx + ROW_WINDOW_SIZE)
    row_window = df_feat.iloc[start:end + 1].copy()
    return row_window, "row"


def _build_extended_window(df_feat: pd.DataFrame, event_idx: int) -> pd.DataFrame:
    """Расширенное окно: ±60 строк от события (для ночных сливов с большим gap)."""
    start = max(0, event_idx - EXTENDED_WINDOW_ROWS)
    end = min(len(df_feat) - 1, event_idx + EXTENDED_WINDOW_ROWS)
    return df_feat.iloc[start:end + 1].copy()


def _plot_event(window: pd.DataFrame, event_time: pd.Timestamp, window_type: str) -> go.Figure:
    """График окна анализа с обозначением времени события."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window["datetime"], y=window["fuel_lvl"],
        mode="lines+markers", name="Уровень топлива",
        line=dict(color="#1f77b4"), marker=dict(size=4),
    ))
    anomalies = window[window["is_anomaly"] == -1]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["datetime"], y=anomalies["fuel_lvl"],
            mode="markers", name="Аномалия (ML)",
            marker=dict(color="red", size=10, symbol="x"),
        ))

    x_line = event_time.to_pydatetime()
    if hasattr(x_line, "tzinfo") and x_line.tzinfo is not None:
        x_line = x_line.replace(tzinfo=None)

    fig.add_vline(x=x_line, line_dash="dash", line_color="orange")
    fig.add_annotation(
        x=x_line, y=window["fuel_lvl"].max(),
        text="Событие", showarrow=False, yanchor="bottom",
        font=dict(color="orange", size=12),
    )

    title_suffix = " (строковое окно)" if window_type == "row" else " (временное окно)"
    fig.update_layout(
        title=f"Окно анализа: {len(window)} точек {title_suffix}",
        xaxis_title="Время", yaxis_title="Уровень, л",
        height=420, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_full_telemetry(df_full: pd.DataFrame, event_time: pd.Timestamp) -> go.Figure:
    """График всей телеметрии с обозначением времени события."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_full["datetime"], y=df_full["fuel_lvl"],
        mode="lines", name="Уровень топлива",
        line=dict(color="#1f77b4", width=1.5),
    ))
    
    anomalies = df_full[df_full["is_anomaly"] == -1]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["datetime"], y=anomalies["fuel_lvl"],
            mode="markers", name="Аномалии (ML)",
            marker=dict(color="red", size=6, symbol="x", opacity=0.6),
        ))
    
    x_line = event_time.to_pydatetime()
    if hasattr(x_line, "tzinfo") and x_line.tzinfo is not None:
        x_line = x_line.replace(tzinfo=None)
    
    fig.add_vline(x=x_line, line_dash="dash", line_color="orange", line_width=2)
    fig.add_annotation(
        x=x_line, y=df_full["fuel_lvl"].max(),
        text="Событие", showarrow=True, arrowhead=2,
        arrowcolor="orange", font=dict(color="orange", size=12),
        yanchor="bottom",
    )
    
    fig.update_layout(
        title=f"Полная телеметрия: {len(df_full)} точек | Событие в {event_time.strftime('%d.%m.%Y %H:%M')}",
        xaxis_title="Время", yaxis_title="Уровень, л",
        height=500, margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        ),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛢️ Детектор сливов")
    uploaded = st.file_uploader("Excel-отчёт о сливах", type=["xlsx", "xls"])
    st.caption(
        f"Окно: ⏱️ −{WINDOW_BEFORE_MIN}/+{WINDOW_AFTER_MIN} мин | "
        f"Мин. точек: {MIN_POINTS_IN_TIME_WINDOW} | "
        f"Строковое: ±{ROW_WINDOW_SIZE}"
    )
    if FEEDBACK_PATH.exists():
        try:
            n_fb = sum(1 for _ in open(FEEDBACK_PATH, encoding="utf-8-sig")) - 1
            st.caption(f"📝 Собрано отзывов: **{max(n_fb, 0)}**")
        except Exception:
            pass


# ============================================================
# ОСНОВНАЯ ЛОГИКА
# ============================================================

if uploaded is None:
    st.info("Загрузите Excel-отчёт, чтобы начать анализ.")
    st.stop()

file_bytes = uploaded.read()

# 1. Парсим отчёт
try:
    object_name, df_drains = parse_drain_report(file_bytes)
except Exception as e:
    st.error(f"Не удалось разобрать отчёт: {e}")
    st.stop()

st.subheader(f"Объект: **{object_name}**")
st.caption(f"Найдено событий в отчёте: **{len(df_drains)}**")

# 2. Тянем телеметрию (с кэшем)
try:
    logger.info(f"🔍 Запрашиваем телеметрию для объекта: '{object_name}'")
    df_telemetry = fetch_telemetry_for_object(
        object_name,
        df_drains["Время"].min().isoformat(),
        df_drains["Время"].max().isoformat(),
    )
    if len(df_telemetry)< MIN_POINTS_IN_TIME_WINDOW:
        min_time = df_drains["Время"].min() - pd.Timedelta(hours=6)
        max_time = df_drains["Время"].max() + pd.Timedelta(hours=6)
        
        logger.info(f"🔍 Запрашиваем телеметрию для объекта: '{object_name}'")
        df_telemetry = fetch_telemetry_for_object(
            object_name,
            min_time.isoformat(),
            max_time.isoformat(),
        )
except Exception as e:
    st.error(f"Ошибка при запросе телеметрии: {e}")
    st.stop()

if df_telemetry.empty:
    st.error("Телеметрия пуста. Проверьте имя объекта и даты.")
    st.stop()

# 3. Считаем признаки + предсказания
router = get_model_router()
df_feat = router.predict(df_telemetry)

# 4. Считаем вердикты по каждому событию (адаптивное окно)
verdicts = []
for _, row in df_drains.iterrows():
    event_time = pd.to_datetime(row["Время"])
    idx = _find_event_idx(df_feat, event_time)

    # Адаптивное окно: временное или строковое
    main_window, window_type = _build_adaptive_window(df_feat, event_time, idx)

    # Расширенное окно: для ночного слива
    extended_window = _build_extended_window(df_feat, idx)

    v = make_verdict(main_window, extended_window)
    v["event_idx"] = idx
    v["event_time"] = event_time
    v["main_window_size"] = len(main_window)
    v["window_type"] = window_type
    v["extended_window_size"] = len(extended_window)
    v["report_level_before"] = row.get("Уровень до")
    v["report_level_after"] = row.get("Уровень после")
    v["report_drain"] = row.get("Слив")
    v["address"] = row.get("Адрес", "")
    verdicts.append(v)

df_verdicts = pd.DataFrame(verdicts)


# 5. Таблица событий
label_colors = {
    "СЛИВ (ML)": "🔴",
    "ПОДОЗРЕНИЕ НА СЛИВ (медленный слив/ночной слив)": "🟠",
    "ПОДОЗРЕНИЕ НА СЛИВ (падение уровня)": "🟡",
    "ЛОЖНЫЙ СЛИВ": "🟢",
}

df_show = df_verdicts.copy()
df_show["Вердикт"] = df_show["label"].map(lambda x: f"{label_colors.get(x, '')} {x}")
df_show["Время события"] = df_show["event_time"].dt.strftime("%d.%m.%Y %H:%M")
df_show["Адрес"] = df_show["address"]
df_show["Отчёт: слив, л"] = df_show["report_drain"]
df_show["ML score (min)"] = df_show["ml_score_min"].round(4)
df_show["Просадка в окне, л"] = df_show["total_drop"].round(1)
df_show["Точек в окне"] = df_show["main_window_size"]
df_show["Тип окна"] = df_show["window_type"].map({"time": "⏱️ время", "row": "📊 строки"})

st.dataframe(
    df_show[["Вердикт", "Время события", "Адрес", "Отчёт: слив, л",
             "ML score (min)", "Просадка в окне, л", "Точек в окне", "Тип окна"]],
    use_container_width=True, hide_index=True,
)



# ============================================================
# ДЕТАЛЬНЫЙ ПРОСМОТР
# ============================================================

# ---------- Детальный просмотр ----------

st.markdown("---")
st.subheader("🔎 Детальный разбор события")

options = [
    f"{i+1}. {v['event_time'].strftime('%d.%m %H:%M')} — {v['label']} — {v['address']}"
    for i, v in enumerate(verdicts)
]
sel = st.selectbox("Выберите событие", range(len(options)), format_func=lambda i: options[i])
v = verdicts[sel]

# Для графика используем адаптивное окно
main_window, window_type = _build_adaptive_window(df_feat, v["event_time"], v["event_idx"])
event_time = v["event_time"]

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(_plot_event(main_window, event_time, window_type), use_container_width=True)
with col2:
    st.metric("Вердикт системы", v["label"])
    st.metric("Аномальных точек (ML)", v["anomaly_points_count"])
    st.metric("ML score (min)", f"{v['ml_score_min']:.4f}")
    st.metric("Просадка в окне, л", f"{v['total_drop']:.1f}")
    st.metric("Gap drawdown, л", f"{v['gap_drop']:.1f}")
    st.caption(f"Окно: {v['main_window_size']} точек ({'⏱️ время' if v['window_type'] == 'time' else '📊 строки'}) | Gap: {v['extended_window_size']} точек")
    if v["rule_reason"]:
        st.info(f"**Правило:** {v['rule_reason']}")


# График полной телеметрии
st.markdown("#### 📊 Полная телеметрия (с возможностью масштабирования)")
st.caption("Используйте инструменты выше графика для приближения/удаления. Двойной клик — сброс масштаба.")
st.plotly_chart(_plot_full_telemetry(df_feat, event_time), use_container_width=True)

# ============================================================
# КНОПКИ ОБРАТНОЙ СВЯЗИ
# ============================================================

st.markdown("---")
st.markdown("#### Ваш вердикт по этому событию")
b1, b2, _ = st.columns([1, 1, 4])
with b1:
    clicked_real = st.button("✅ Реальный слив", key=f"real_{sel}", use_container_width=True)
with b2:
    clicked_false = st.button("❌ Ложное срабатывание", key=f"false_{sel}", use_container_width=True)

features_row = df_feat.loc[v["event_idx"], router.features]

event_info = {
    "object_name": object_name,
    "event_time": event_time.isoformat(),
    "address": v["address"],
    **{k: v[k] for k in ["label", "ml_detected", "rule_detected", "rule_reason",
                         "ml_score_min", "anomaly_points_count", "total_drop", "gap_drop"]},
}

if clicked_real:
    save_feedback(event_info, "real", features_row)
    st.success("✅ Спасибо! Записано как «Реальный слив».")
if clicked_false:
    save_feedback(event_info, "false", features_row)
    st.success("❌ Спасибо! Записано как «Ложное срабатывание».")


# ============================================================
# ИСТОРИЯ ФИДБЕКА
# ============================================================

with st.expander("📜 История собранных отзывов"):
    if FEEDBACK_PATH.exists():
        try:
            df_fb = pd.read_csv(FEEDBACK_PATH, encoding="utf-8-sig")
            st.dataframe(df_fb.tail(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Не удалось прочитать feedback.csv: {e}")
    else:
        st.info("Пока нет собранных отзывов.")