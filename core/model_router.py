# -*- coding: utf-8 -*-
"""
Единая точка входа для предсказаний.
Сейчас: IsolationForest (novelty detection).
В будущем: сюда добавится CatBoost (supervised), и predict() будет агрегировать обе модели.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import MODEL_PATH, FEATURES_PATH
from core.features import create_features


@st.cache_resource
def _load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def _load_features() -> List[str]:
    return joblib.load(FEATURES_PATH)


class ModelRouter:
    """Роутер моделей. Инкапсулирует всё, что касается предсказаний."""

    def __init__(self):
        self.if_model = _load_model()
        self.features: List[str] = list(_load_features())
        # Заглушка под будущее:
        # self.cb_model = _load_catboost() if CATBOOST_PATH.exists() else None

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Принимает сырую телеметрию (с object_id, Время, fuel_lvl, ...).
        Возвращает DataFrame с добавленными колонками:
          - все признаки из self.features
          - is_anomaly   (1 = норма, -1 = аномалия — формат IF)
          - anomaly_score
        """
        df = create_features(df_raw)
        X = df[self.features].copy()

        # Защита от inf/NaN
        for col in self.features:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

        df["is_anomaly"] = self.if_model.predict(X)
        df["anomaly_score"] = self.if_model.decision_function(X)

        # В будущем здесь будет агрегация с CatBoost:
        # if self.cb_model is not None:
        #     df["cb_proba"] = self.cb_model.predict_proba(X)[:, 1]
        #     df["final_verdict"] = self._aggregate(df["is_anomaly"], df["cb_proba"])

        return df


@st.cache_resource
def get_model_router() -> ModelRouter:
    return ModelRouter()
