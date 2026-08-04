"""Streamlit-приложение для детекции сливов топлива.
Воспроизводит логику из ноутбука с адаптивным окном для ночных сливов.
"""
from __future__ import annotations
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.feedback import (
    save_feedback,
    save_telemetry_context,
    detect_last_refuel,
    load_feedback,
    get_feedback_count,
)
from config import (
    WINDOW_BEFORE_MIN, WINDOW_AFTER_MIN,
    MIN_POINTS_IN_TIME_WINDOW, ROW_WINDOW_SIZE,
    EXTENDED_WINDOW_ROWS, SENSOR_MAPPING,
    BASE_DIR, SPEED_FILTER_THRESHOLD_KMH,
)
from core.api import fetch_telemetry_for_object
from core.telemetry import parse_drain_report
from core.model_router import get_model_router
from core.heuristics import make_verdict, post_process_verdict

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


st.logo("Агропилот.png")

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
    # Счётчик отзывов из БД
    try:
      
        n_fb = get_feedback_count()
        if n_fb >= 0:
            st.caption(f"📝 Собрано отзывов в БД: **{n_fb}**")
        else:
            st.caption("📝 БД недоступна")
    except Exception:
        st.caption("📝 Ошибка подключения к БД")

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

# 2. Тянем телеметрию (с расширенным окном для контекста CatBoost)
try:
    # Расширяем окно: -24ч до первого события (для поиска заправки)
    # и +3ч после последнего (для восстановления уровня)
    min_time = df_drains["Время"].min() - pd.Timedelta(hours=24)
    max_time = df_drains["Время"].max() + pd.Timedelta(hours=3)

    logger.info(f"🔍 Запрашиваем телеметрию для объекта: '{object_name}'")
    logger.info(f"🔍 Окно: {min_time} — {max_time}")

    df_telemetry = fetch_telemetry_for_object(
        object_name,
        min_time.isoformat(),
        max_time.isoformat(),
    )
except Exception as e:
    st.error(f"Ошибка при запросе телеметрии: {e}")
    st.stop()

# ⚠️ Эту проверку обязательно вернуть!
if df_telemetry.empty:
    st.error("Телеметрия пуста. Проверьте имя объекта и даты.")
    st.stop()
    
# Показываем, какие датчики были найдены
available_cols = [c for c in df_telemetry.columns if c in SENSOR_MAPPING.keys()]
missing_cols = [c for c in SENSOR_MAPPING.keys() if c not in df_telemetry.columns]

if missing_cols:
    st.warning(f"⚠️ У объекта отсутствуют датчики: {', '.join(missing_cols)}. "
              f"Для них будут использованы дефолтные значения.")
else:
    st.success(f"✅ Все датчики найдены: {', '.join(available_cols)}")

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
    v = post_process_verdict(v, main_window, speed_threshold_kmh=SPEED_FILTER_THRESHOLD_KMH)
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
        "ВЕРОЯТНО ЛОЖНЫЙ СЛИВ (движение)": "🟢",
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

# 🔧 ПЕРЕСЧИТЫВАЕМ ОКНО ДЛЯ ВЫБРАННОГО СОБЫТИЯ
main_window, window_type = _build_adaptive_window(df_feat, v["event_time"], v["event_idx"])
event_time = v["event_time"]

# ============================================================
# ОБЪЯСНЕНИЕ РЕШЕНИЯ МОДЕЛИ (XAI) - ОБНОВЛЯЕТСЯ ПРИ СМЕНЕ СОБЫТИЯ
# ============================================================
st.markdown("---")
st.markdown("### 🧠 Почему система приняла такое решение?")
st.caption("Анализ агрегированных признаков телеметрии во всем окне анализа (не только в момент события)")

# 🔧 ИСПОЛЬЗУЕМ main_window, который пересчитан для выбранного события
window = main_window 

# Рассчитываем агрегированные метрики по всему окну
max_drop_10min = float(window['total_drop_10min'].max()) if 'total_drop_10min' in window.columns else 0.0
max_drop_rate = float(window['fuel_drop_rate'].max()) if 'fuel_drop_rate' in window.columns else 0.0
max_consecutive = int(window['consecutive_drops'].max()) if 'consecutive_drops' in window.columns else 0
was_stationary = (window['speed'] == 0).mean() > 0.8  # Если >80% времени техника стояла
gnss_anomalies_count = int(window['is_gnss_anomaly'].sum())
conn_lost_count = int(window['is_connection_lost'].sum())

# 🚗 НОВЫЕ МЕТРИКИ ДВИЖЕНИЯ
avg_speed = float(window['speed'].mean()) if 'speed' in window.columns else 0.0
max_speed = float(window['speed'].max()) if 'speed' in window.columns else 0.0
moving_share = float((window['speed'] > 2.0).mean()) if 'speed' in window.columns else 0.0  # доля времени в движении (>2 км/ч)
is_moving = avg_speed > 15.0  # порог скоростного фильтра (болтанка)

# Считаем общую просадку топлива в окне (только отрицательные значения)
total_fuel_loss = float(window[window['fuel_diff'] < 0]['fuel_diff'].sum()) if len(window[window['fuel_diff'] < 0]) > 0 else 0.0

# Создаем 4 колонки для компактного отображения метрик окна
col1, col2, col3, col4 = st.columns(4)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Макс. просадка за 10 мин",
        f"{max_drop_10min:.1f} л",
        delta="Подозрительно ⚠️" if max_drop_10min > 3.0 else "Норма",
        delta_color="inverse" if max_drop_10min > 3.0 else "off"
    )
    st.metric(
        "Макс. скорость падения",
        f"{max_drop_rate:.2f} л/мин",
        delta="Высокая ⚠️" if max_drop_rate > 1.0 else "Норма",
        delta_color="inverse" if max_drop_rate > 1.0 else "off"
    )
with col2:
    st.metric(
        "Макс. монолитность падения",
        f"{max_consecutive} точек",
        help="Сколько точек подряд показывают строгое снижение уровня топлива"
    )
    st.metric(
        "Общая просадка в окне",
        f"{abs(total_fuel_loss):.1f} л"
    )
with col3:
    st.metric(
        "Режим техники",
        "Стоянка ✅" if was_stationary else "В движении 🚗",
        delta_color="normal" if was_stationary else "inverse"
    )
    # 🚗 НОВАЯ МЕТРИКА: средняя скорость
    st.metric(
        "Средняя скорость",
        f"{avg_speed:.1f} км/ч",
        delta="Болтанка ⚠️" if is_moving else "Стоянка",
        delta_color="inverse" if is_moving else "normal",
        help="При средней скорости > 15 км/ч падение уровня скорее всего вызвано болтанкой топлива, а не сливом"
    )
with col4:
    st.metric(
        "Аномалии GPS (РЭБ)",
        f"⚠️ {gnss_anomalies_count} раз" if gnss_anomalies_count > 0 else "✅ Нет",
        delta_color="inverse" if gnss_anomalies_count > 0 else "off"
    )
    st.metric(
        "Разрывы связи",
        f"⚠️ {conn_lost_count} раз" if conn_lost_count > 0 else "✅ Нет",
        delta_color="inverse" if conn_lost_count > 0 else "off"
    )

# Генерация текстовой расшифровки на основе агрегированных признаков окна
st.markdown("#### 📝 Расшифровка решения:")
explanation = []

# 🚗 0. Проверка на движение (болтанка топлива)
if is_moving:
    explanation.append(
        f"🔵 Техника двигалась со средней скоростью **{avg_speed:.1f} км/ч** "
        f"(в движении {moving_share*100:.0f}% времени, макс. {max_speed:.0f} км/ч). "
        f"При движении падение уровня топлива с высокой вероятностью вызвано **болтанкой** "
        f"(плескание топлива в баке), а не сливом. Слив в движении крайне маловероятен."
    )
elif not was_stationary:
    explanation.append(
        f"🟡 Техника двигалась с невысокой скоростью "
        f"(средняя {avg_speed:.1f} км/ч, {moving_share*100:.0f}% времени в движении). "
        f"Возможна как болтанка, так и слив при кратковременной остановке — требуется внимание."
    )

# 1. Проверка на классический слив при стоянке (по максимуму в окне)
if was_stationary and max_drop_10min > 3.0:
    explanation.append(f"🔴 **Техника преимущественно стояла**, но в окне анализа зафиксирована просадка топлива **{max_drop_10min:.1f} л** за 10 минут. Это основной триггер для подозрения на слив.")

# 2. Проверка на монотонность (ключевой признак реального слива, а не шума)
if max_consecutive >= 3:
    explanation.append(f" Зафиксировано монотонное (непрерывное) падение уровня в течение **{max_consecutive} точек** подряд, что характерно для реального слива, а не для шума датчика.")

# 3. Проверка на РЭБ / помехи
if gnss_anomalies_count > 0:
    explanation.append(f"🟡 **Обнаружены аномалии GPS/ГЛОНАСС** ({gnss_anomalies_count} раз в окне). Данные об уровне топлива в этот период могут быть искажены. Вердикт модели требует повышенной внимательности.")

# 4. Проверка на ночной слив / разрыв связи
if conn_lost_count > 0 and total_fuel_loss < -3.0:
    explanation.append(f"🟡 Был разрыв связи. После восстановления уровень топлива оказался ниже на **{abs(total_fuel_loss):.1f} л**, что характерно для 'ночного слива'.")

# 5. Если все чисто
if not explanation:
    explanation.append("🟢 **Признаки соответствуют нормальной эксплуатации**: в окне анализа не зафиксировано резких необъяснимых просадок уровня топлива при стоянке. Модель классифицировала это как норму.")

# Выводим расшифровку списком
for exp in explanation:
    st.markdown(f"- {exp}")


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
st.caption(
    "Отзыв будет записан в PostgreSQL: событие + точки окна + "
    "полный контекст телеметрии (−24ч/+3ч) для обучения CatBoost."
)

b1, b2, _ = st.columns([1, 1, 4])
with b1:
    clicked_real = st.button("✅ Реальный слив", key=f"real_{sel}", use_container_width=True)
with b2:
    clicked_false = st.button("❌ Ложное срабатывание", key=f"false_{sel}", use_container_width=True)

# Признаки в момент события (для обратной совместимости)
features_row = df_feat.loc[v["event_idx"], router.features]

# Всё окно анализа (адаптивное) — для записи точек в БД
main_window, _ = _build_adaptive_window(df_feat, v["event_time"], v["event_idx"])

event_time = v["event_time"]
event_idx = v["event_idx"]

# 🔧 Расчёт времени с последней заправки по ПОЛНОЙ телеметрии
time_since_refuel = detect_last_refuel(df_feat, event_time)

event_info = {
    "object_name": object_name,
    "event_time": event_time.isoformat(),
    "address": v["address"],
    "system_label": v["label"],
    **{k: v[k] for k in [
        "ml_detected", "rule_detected", "rule_reason",
        "ml_score_min", "anomaly_points_count", "total_drop", "gap_drop",
    ]},
}


def _process_feedback(verdict: str) -> None:
    """Единая обработка сохранения feedback + контекста телеметрии."""
    # 1. Сохраняем событие + точки окна (возвращает event_id)
    event_id = save_feedback(
        event_info=event_info,
        user_verdict=verdict,
        features_row=features_row,
        window_df=main_window,
        time_since_last_refuel_min=time_since_refuel,
    )

    if event_id < 0:
        st.error("⚠️ Не удалось сохранить в БД. Проверьте подключение к PostgreSQL.")
        return

    # 2. Сохраняем полную телеметрию (контекст для CatBoost)
    ctx_ok = save_telemetry_context(event_id, df_feat, event_time, event_idx)

    if ctx_ok:
        st.success(
            f"✅ Записано в БД: событие + {len(main_window)} точек окна "
            f"+ контекст телеметрии. Время с заправки: {time_since_refuel:.0f} мин."
        )
    else:
        st.warning(
            f"⚠️ Событие сохранено (id={event_id}), "
            f"но контекст телеметрии не записан. Проверьте таблицу telemetry_context."
        )


if clicked_real:
    _process_feedback("real")

if clicked_false:
    _process_feedback("false")


# ============================================================
# ИСТОРИЯ ФИДБЕКА (теперь из PostgreSQL)
# ============================================================

from core.feedback import load_feedback, get_feedback_count

with st.expander("📜 История собранных отзывов (из БД)"):
    count = get_feedback_count()
    if count > 0:
        st.caption(f"Всего записей в базе данных: **{count}**")
        
        # Загружаем последние 50 записей
        df_fb = load_feedback(limit=50)
        
        if not df_fb.empty:
            # Показываем только ключевые колонки для удобства чтения оператором
            # (признаки модели скрыты, они нужны только для обучения CatBoost)
            display_cols = [
                "feedback_ts", "object_name", "event_time", "user_verdict",
                "system_label", "rule_reason", "total_drop", "ml_score_min"
            ]
            
            # Фильтруем только те колонки, которые реально существуют в DataFrame
            cols_to_show = [c for c in display_cols if c in df_fb.columns]
            
            # Переименуем колонки для красивого отображения на русском
            rename_map = {
                "feedback_ts": "Время отзыва",
                "object_name": "Объект",
                "event_time": "Время события",
                "user_verdict": "Вердикт пользователя",
                "system_label": "Вердикт системы",
                "rule_reason": "Причина (правило)",
                "total_drop": "Просадка (л)",
                "ml_score_min": "ML score (min)"
            }
            df_display = df_fb[cols_to_show].rename(columns=rename_map)
            
            # Форматируем вердикт пользователя для наглядности
            df_display["Вердикт пользователя"] = df_display["Вердикт пользователя"].map({
                "real": "✅ Реальный",
                "false": "❌ Ложный"
            })
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("Не удалось загрузить данные из БД.")
    else:
        st.info("Пока нет собранных отзывов в базе данных.")
