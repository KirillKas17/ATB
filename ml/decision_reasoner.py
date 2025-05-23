import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import talib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


@dataclass
class DecisionConfig:
    """Конфигурация принятия решений"""

    min_confidence: float = 0.7
    max_risk: float = 0.3
    min_samples: int = 100
    max_samples: int = 1000
    update_interval: int = 1  # часов
    cache_size: int = 1000
    compression: bool = True
    metrics_window: int = 24  # часов
    ensemble_size: int = 3
    feature_importance_threshold: float = 0.1
    decision_timeout: int = 5  # секунд


@dataclass
class DecisionMetrics:
    """Метрики принятия решений"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confidence: float
    risk_score: float
    last_update: datetime
    samples_count: int
    decision_count: int
    error_count: int
    feature_importance: Dict[str, float]


@dataclass
class DecisionReport:
    """Отчет о принятом решении"""

    signal_type: str  # тип сигнала (long/short)
    confidence: float  # уверенность в решении
    timestamp: datetime  # время принятия решения
    features_importance: Dict[str, float]  # важность признаков
    technical_indicators: Dict[str, float]  # значения индикаторов
    market_context: Dict[str, Any]  # контекст рынка
    explanation: str  # текстовое объяснение
    visualization_path: str  # путь к визуализации


@dataclass
class TradeDecision:
    """Торговое решение"""

    symbol: str
    action: str  # open/close/hold
    direction: str  # long/short
    volume: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    metadata: Dict


class DecisionReasoner:
    """Module for making trading decisions with confidence estimation."""

    def __init__(
        self,
        min_confidence=0.7,
        max_uncertainty=0.3,
        config: Optional[DecisionConfig] = None,
        log_dir: str = "logs",
    ):
        """
        Инициализация анализатора решений.

        Args:
            min_confidence: Минимальная уверенность
            max_uncertainty: Максимальная неопределенность
            config (Optional[DecisionConfig]): Конфигурация параметров
            log_dir: Директория для логов
        """
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
        self.config = config or DecisionConfig()
        self.decision_dir = Path("decisions")
        self.decision_dir.mkdir(parents=True, exist_ok=True)

        # Создание директорий
        os.makedirs(self.decision_dir / "reports", exist_ok=True)
        os.makedirs(self.decision_dir / "visualizations", exist_ok=True)

        # Инициализация компонентов
        self.scaler = StandardScaler()
        self.explainer = None
        self.lime_explainer = None

        # Хранение отчетов
        self.reports: Dict[str, List[DecisionReport]] = {}

        self.bayes_model = GaussianNB()
        self.logistic_model = LogisticRegression()
        self.trade_history = []
        self.model_trained = False

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логгера
        logger.add(
            self.log_dir / "decision_reason.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

        # Данные
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []

        # Модели
        self.models = {}
        self.scalers = {}
        self.metrics = {}

        # Кэш
        self._decision_cache = {}
        self._feature_cache = {}

        # Загрузка состояния
        self._load_state()

    def _load_state(self):
        """Загрузка состояния"""
        try:
            state_file = self.decision_dir / "state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.metrics_history = state.get("metrics_history", [])
                    self.metrics = state.get("metrics", {})
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        try:
            state_file = self.decision_dir / "state.json"
            state = {"metrics_history": self.metrics_history, "metrics": self.metrics}
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков"""
        try:
            features = pd.DataFrame()

            # Ценовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log1p(features["returns"])
            features["volatility"] = features["returns"].rolling(20).std()
            # Технические индикаторы
            features["rsi"] = pd.Series(
                ta.RSI(df["close"].values, timeperiod=14), 
                index=df.index
            )
            macd, macd_signal, _ = ta.MACD(
                df["close"].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            features["macd"] = pd.Series(macd, index=df.index)
            features["macd_signal"] = pd.Series(macd_signal, index=df.index)

            bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"].values)
            features["bb_upper"] = pd.Series(bb_upper, index=df.index)
            features["bb_middle"] = pd.Series(bb_middle, index=df.index)
            features["bb_lower"] = pd.Series(bb_lower, index=df.index)

            features["atr"] = ta.ATR(
                df["high"].values, df["low"].values, df["close"].values
            )
            features["adx"] = ta.ADX(
                df["high"].values, df["low"].values, df["close"].values
            )

            # Объемные признаки
            features["volume_ma"] = df["volume"].rolling(20).mean()
            features["volume_std"] = df["volume"].rolling(20).std()
            features["volume_ratio"] = df["volume"] / features["volume_ma"]

            return features.fillna(0)
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return pd.DataFrame()

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Расчет экспоненциальной скользящей средней"""
        try:
            return pd.Series(
                ta.EMA(data.values, timeperiod=period), index=data.index
            )
        except Exception as e:
            logger.error(f"Ошибка расчета EMA: {e}")
            return pd.Series(index=data.index)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Расчет метрик"""
        try:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted")),
                "recall": float(recall_score(y_true, y_pred, average="weighted")),
                "f1": float(f1_score(y_true, y_pred, average="weighted")),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            return {}

    def _calculate_risk_score(
        self, features: pd.DataFrame, predictions: np.ndarray
    ) -> float:
        """Расчет оценки риска"""
        try:
            # Волатильность
            volatility = features["volatility"].iloc[-1]

            # Тренд
            trend = features["trend"].iloc[-1]

            # Объем
            volume = features["volume_ratio"].iloc[-1]

            # Моментум
            momentum = features["momentum"].iloc[-1]

            # Расчет риска
            risk = np.mean(
                [volatility, abs(trend), 1 / volume if volume > 0 else 1, abs(momentum)]
            )

            return float(risk)

        except Exception as e:
            logger.error(f"Ошибка расчета риска: {e}")
            return 1.0

    def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Получение важности признаков"""
        try:
            if model_id not in self.models:
                return {}

            model = self.models[model_id]
            if not hasattr(model, "feature_importances_"):
                return {}

            features = self._extract_features(self.data_buffer)
            importance = dict(zip(features.columns, model.feature_importances_))

            # Нормализация
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            # Фильтрация по порогу
            importance = {
                k: v
                for k, v in importance.items()
                if v >= self.config.feature_importance_threshold
            }

            return importance

        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return {}

    def update(self, df: pd.DataFrame, model_id: str, model: Any):
        """Обновление модели"""
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, df]).tail(
                self.config.max_samples
            )

            if len(self.data_buffer) < self.config.min_samples:
                return

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)

            # Подготовка данных
            X = features.values
            y = (
                self.data_buffer["close"].shift(-1) > self.data_buffer["close"]
            ).values[:-1]
            X = X[
                :-1
            ]  # Убираем последнюю строку, так как для нее нет целевой переменной

            # Нормализация
            if model_id not in self.scalers:
                self.scalers[model_id] = StandardScaler()
            X_scaled = self.scalers[model_id].fit_transform(X)

            # Обучение модели
            model.fit(X_scaled, y)
            self.models[model_id] = model

            # Расчет метрик
            y_pred = model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)

            # Обновление истории метрик
            self.metrics_history.append(
                {"timestamp": datetime.now(), "model_id": model_id, **metrics}
            )

            # Ограничение истории
            self.metrics_history = self.metrics_history[-self.config.metrics_window :]

            # Расчет риска
            risk_score = self._calculate_risk_score(features, y_pred)

            # Получение важности признаков
            feature_importance = self._get_feature_importance(model_id)

            # Обновление метрик
            self.metrics[model_id] = DecisionMetrics(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                confidence=1.0 - risk_score,
                risk_score=risk_score,
                last_update=datetime.now(),
                samples_count=len(self.data_buffer),
                decision_count=self.metrics.get(model_id, {}).get("decision_count", 0)
                + 1,
                error_count=self.metrics.get(model_id, {}).get("error_count", 0),
                feature_importance=feature_importance,
            ).__dict__

            # Сохранение состояния
            self._save_state()

            logger.info(f"Модель {model_id} обновлена. Метрики: {metrics}")

        except Exception as e:
            logger.error(f"Ошибка обновления модели: {e}")
            if model_id in self.metrics:
                self.metrics[model_id]["error_count"] += 1
            raise

    def make_decision(
        self, df: pd.DataFrame, model_id: str
    ) -> Tuple[np.ndarray, float, float]:
        """Принятие решения"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Модель {model_id} не найдена")

            # Извлечение признаков
            features = self._extract_features(df)

            # Нормализация
            X_scaled = self.scalers[model_id].transform(features.values)

            # Предсказание
            predictions = self.models[model_id].predict(X_scaled)
            probabilities = self.models[model_id].predict_proba(X_scaled)

            # Расчет уверенности
            confidence = float(np.mean(np.max(probabilities, axis=1)))

            # Расчет риска
            risk_score = self._calculate_risk_score(features, predictions)

            # Проверка порогов
            if confidence < self.min_confidence:
                logger.warning(f"Низкая уверенность: {confidence}")
                return np.zeros_like(predictions), 0.0, 1.0

            if risk_score > self.config.max_risk:
                logger.warning(f"Высокий риск: {risk_score}")
                return np.zeros_like(predictions), 0.0, 1.0

            return predictions, confidence, risk_score

        except Exception as e:
            logger.error(f"Ошибка принятия решения: {e}")
            raise

    def get_metrics(self, model_id: Optional[str] = None) -> Dict:
        """Получение метрик"""
        if model_id:
            return self.metrics.get(model_id, {})
        return self.metrics

    def get_history(self, model_id: Optional[str] = None) -> List[Dict]:
        """Получение истории"""
        if model_id:
            return [m for m in self.metrics_history if m["model_id"] == model_id]
        return self.metrics_history

    def reset(self):
        """Сброс состояния"""
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self._decision_cache.clear()
        self._feature_cache.clear()

    def explain_decision(
        self, pair: str, timeframe: str, data: pd.DataFrame, model: Any, signal: Dict
    ) -> DecisionReport:
        """
        Объяснение принятого решения.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            data: Данные
            model: Модель
            signal: Сигнал

        Returns:
            DecisionReport: Отчет о решении
        """
        try:
            # Подготовка данных
            features = self._prepare_features(data)

            # Инициализация объяснителей
            if self.explainer is None:
                self._init_explainers(features, model)

            # Получение важности признаков
            feature_importance = self._get_feature_importance(model)

            # Получение значений индикаторов
            indicators = self._get_technical_indicators(data)

            # Получение контекста рынка
            market_context = self._get_market_context(data)

            # Генерация объяснения
            explanation = self._generate_explanation(
                signal, feature_importance, indicators, market_context
            )

            # Создание визуализации
            viz_path = self._create_visualization(
                pair, timeframe, feature_importance, indicators
            )

            # Создание отчета
            report = DecisionReport(
                signal_type=signal["type"],
                confidence=signal["confidence"],
                timestamp=datetime.now(),
                features_importance=feature_importance,
                technical_indicators=indicators,
                market_context=market_context,
                explanation=explanation,
                visualization_path=viz_path,
            )

            # Сохранение отчета
            self._save_report(pair, timeframe, report)

            return report

        except Exception as e:
            logger.error(f"Error explaining decision: {str(e)}")
            return None

    def _init_explainers(self, features: pd.DataFrame, model: Any):
        """Инициализация объяснителей"""
        try:
            # SHAP объяснитель
            self.explainer = shap.KernelExplainer(
                model.predict,
                features.sample(n=min(self.config.max_samples, len(features))),
            )

            # LIME объяснитель
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                features.values,
                feature_names=features.columns,
                class_names=self.config.class_names,
                mode="classification",
            )

        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")

    def _get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Получение значений индикаторов"""
        try:
            indicators = {}

            # RSI
            indicators["rsi"] = talib.RSI(data["close"]).iloc[-1]

            # MACD
            macd, signal, _ = talib.MACD(data["close"])
            indicators["macd"] = macd.iloc[-1]
            indicators["macd_signal"] = signal.iloc[-1]

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data["close"])
            indicators["bb_upper"] = bb_upper.iloc[-1]
            indicators["bb_middle"] = bb_middle.iloc[-1]
            indicators["bb_lower"] = bb_lower.iloc[-1]

            # ATR
            indicators["atr"] = talib.ATR(
                data["high"], data["low"], data["close"]
            ).iloc[-1]

            return indicators

        except Exception as e:
            logger.error(f"Error getting technical indicators: {str(e)}")
            return {}

    def _get_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расширенный анализ контекста рынка.

        Args:
            data: DataFrame с рыночными данными

        Returns:
            Dict[str, Any]: Контекст рынка
        """
        try:
            context = {}

            # Анализ тренда
            ema_20 = self.calculate_ema(data["close"], 20)
            ema_50 = self.calculate_ema(data["close"], 50)
            ema_200 = self.calculate_ema(data["close"], 200)

            context["trend"] = {
                "direction": "up" if ema_20.iloc[-1] > ema_200.iloc[-1] else "down",
                "strength": abs(ema_20.iloc[-1] - ema_200.iloc[-1]) / ema_200.iloc[-1],
                "acceleration": (ema_20.pct_change(5) - ema_50.pct_change(5)).iloc[-1],
                "consistency": self._calculate_trend_consistency(data),
            }

            # Анализ волатильности
            atr = talib.ATR(
                data["high"].values, data["low"].values, data["close"].values
            )
            volatility = data["close"].pct_change().rolling(20).std()

            context["volatility"] = {
                "current": volatility.iloc[-1],
                "historical": volatility.mean(),
                "atr": atr.iloc[-1],
                "volatility_ratio": atr.iloc[-1] / data["close"].iloc[-1],
                "regime": "high" if volatility.iloc[-1] > 0.02 else "low",
            }

            # Анализ объема
            volume_ma = talib.SMA(data["volume"], timeperiod=20)
            volume_trend = data["volume"].pct_change(20)

            context["volume"] = {
                "current": data["volume"].iloc[-1],
                "trend": volume_trend.iloc[-1],
                "relative": data["volume"].iloc[-1] / volume_ma.iloc[-1],
                "distribution": self._calculate_volume_profile(data),
            }

            # Анализ импульса
            rsi = talib.RSI(data["close"].values)
            macd, signal, hist = talib.MACD(data["close"].values)

            context["momentum"] = {
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "macd_hist": hist.iloc[-1],
                "price_momentum": data["close"].pct_change(10).iloc[-1],
            }

            # Анализ структуры рынка
            support_resistance = self._calculate_market_structure(data)
            liquidity_zones = self._calculate_liquidity_zones(data)

            context["market_structure"] = {
                "support_resistance": support_resistance,
                "liquidity_zones": liquidity_zones,
                "price_position": self._calculate_price_position(
                    data, support_resistance
                ),
                "fractals": self._calculate_fractals(data),
            }

            # Анализ манипуляций
            imbalance = self._calculate_imbalance(data)
            fakeouts = self._identify_fakeouts(data)

            context["manipulation"] = {
                "imbalance": imbalance,
                "fakeouts": fakeouts,
                "stop_hunts": self._identify_stop_hunts(data),
                "liquidity_hunts": self._identify_liquidity_hunts(data),
            }

            # Анализ корреляций
            context["correlations"] = self._calculate_correlations(data)

            # Анализ временных паттернов
            context["time_patterns"] = self._analyze_time_patterns(data)

            return context

        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {}

    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Расчет согласованности тренда."""
        try:
            # Расчет направления свечей
            candle_direction = (data["close"] > data["open"]).astype(int)

            # Расчет согласованности
            consistency = candle_direction.rolling(20).mean().iloc[-1]

            return consistency

        except Exception as e:
            logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.5

    def _calculate_price_position(
        self, data: pd.DataFrame, levels: List[float]
    ) -> float:
        """Расчет позиции цены относительно уровней."""
        try:
            current_price = data["close"].iloc[-1]
            min_level = min(levels)
            max_level = max(levels)

            return (current_price - min_level) / (max_level - min_level)

        except Exception as e:
            logger.error(f"Error calculating price position: {str(e)}")
            return 0.5

    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict:
        """Анализ временных паттернов."""
        try:
            # Преобразование временных меток
            data["hour"] = pd.to_datetime(data.index).hour
            data["day_of_week"] = pd.to_datetime(data.index).dayofweek

            # Анализ по часам
            hourly_volatility = data.groupby("hour")["close"].pct_change().std()
            hourly_volume = data.groupby("hour")["volume"].mean()

            # Анализ по дням недели
            daily_volatility = data.groupby("day_of_week")["close"].pct_change().std()
            daily_volume = data.groupby("day_of_week")["volume"].mean()

            return {
                "hourly": {
                    "volatility": hourly_volatility.to_dict(),
                    "volume": hourly_volume.to_dict(),
                },
                "daily": {
                    "volatility": daily_volatility.to_dict(),
                    "volume": daily_volume.to_dict(),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing time patterns: {str(e)}")
            return {}

    def _get_candle_patterns(self, data: pd.DataFrame) -> List[str]:
        """Расширенное определение свечных паттернов.

        Args:
            data: DataFrame с данными свечей

        Returns:
            List[str]: Список обнаруженных паттернов
        """
        try:
            patterns = []

            # Базовые паттерны
            if talib.CDLDOJI(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("doji")

            if talib.CDLENGULFING(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("engulfing")

            if talib.CDLHAMMER(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("hammer")

            # Расширенные паттерны
            if talib.CDLMORNINGSTAR(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("morning_star")

            if talib.CDLEVENINGSTAR(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("evening_star")

            if talib.CDLHARAMI(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("harami")

            if talib.CDLPIERCING(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("piercing")

            if talib.CDLDARKCLOUDCOVER(
                data["open"], data["high"], data["low"], data["close"]
            ).iloc[-1]:
                patterns.append("dark_cloud_cover")

            # Кастомные паттерны
            if self._is_three_white_soldiers(data):
                patterns.append("three_white_soldiers")

            if self._is_three_black_crows(data):
                patterns.append("three_black_crows")

            if self._is_three_line_strike(data):
                patterns.append("three_line_strike")

            if self._is_doji_star(data):
                patterns.append("doji_star")

            if self._is_abandoned_baby(data):
                patterns.append("abandoned_baby")

            if self._is_gravestone_doji(data):
                patterns.append("gravestone_doji")

            if self._is_dragonfly_doji(data):
                patterns.append("dragonfly_doji")

            if self._is_marubozu(data):
                patterns.append("marubozu")

            if self._is_spinning_top(data):
                patterns.append("spinning_top")

            if self._is_belt_hold(data):
                patterns.append("belt_hold")

            if self._is_kicking(data):
                patterns.append("kicking")

            if self._is_breakaway(data):
                patterns.append("breakaway")

            if self._is_meeting_lines(data):
                patterns.append("meeting_lines")

            if self._is_in_neck(data):
                patterns.append("in_neck")

            if self._is_on_neck(data):
                patterns.append("on_neck")

            if self._is_thrusting(data):
                patterns.append("thrusting")

            if self._is_upside_gap_two_crows(data):
                patterns.append("upside_gap_two_crows")

            if self._is_three_inside_up(data):
                patterns.append("three_inside_up")

            if self._is_three_inside_down(data):
                patterns.append("three_inside_down")

            if self._is_three_outside_up(data):
                patterns.append("three_outside_up")

            if self._is_three_outside_down(data):
                patterns.append("three_outside_down")

            return patterns

        except Exception as e:
            logger.error(f"Error getting candle patterns: {str(e)}")
            return []

    def _generate_explanation(
        self,
        signal: Dict,
        feature_importance: Dict[str, float],
        indicators: Dict[str, float],
        market_context: Dict[str, Any],
    ) -> str:
        """Генерация текстового объяснения"""
        try:
            explanation = []

            # Основное решение
            explanation.append(
                f"Signal: {signal['type'].upper()} "
                f"(confidence: {signal['confidence']:.2f})"
            )

            # Важные признаки
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:3]

            explanation.append("\nKey factors:")
            for feature, importance in top_features:
                explanation.append(f"- {feature}: {importance:.2f}")

            # Технические индикаторы
            explanation.append("\nTechnical indicators:")
            for indicator, value in indicators.items():
                explanation.append(f"- {indicator}: {value:.2f}")

            # Контекст рынка
            explanation.append("\nMarket context:")
            explanation.append(f"- Trend: {market_context['trend']}")
            explanation.append(f"- Volatility: {market_context['volatility']:.2f}")
            if market_context["candle_patterns"]:
                explanation.append(
                    f"- Patterns: {', '.join(market_context['candle_patterns'])}"
                )

            return "\n".join(explanation)

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Error generating explanation"

    def _create_visualization(
        self,
        pair: str,
        timeframe: str,
        feature_importance: Dict[str, float],
        indicators: Dict[str, float],
    ) -> str:
        """Создание визуализации"""
        try:
            # Создание фигуры
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            # График важности признаков
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())

            sns.barplot(x=importance, y=features, ax=ax1)
            ax1.set_title("Feature Importance")

            # График индикаторов
            indicators_names = list(indicators.keys())
            indicator_values = list(indicators.values())

            sns.barplot(x=indicator_values, y=indicators_names, ax=ax2)
            ax2.set_title("Technical Indicators")

            # Сохранение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pair}_{timeframe}_{timestamp}.png"
            path = os.path.join(self.decision_dir / "visualizations", filename)

            plt.tight_layout()
            plt.savefig(path)
            plt.close()

            return path

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return ""

    def _save_report(self, pair: str, timeframe: str, report: DecisionReport):
        """Сохранение отчета"""
        try:
            # Создание директории
            pair_dir = self.decision_dir / pair
            pair_dir.mkdir(parents=True, exist_ok=True)

            # Сохранение отчета
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{timeframe}_{timestamp}.json"
            path = pair_dir / filename

            with open(path, "w") as f:
                json.dump(
                    {
                        "signal_type": report.signal_type,
                        "confidence": report.confidence,
                        "timestamp": report.timestamp.isoformat(),
                        "features_importance": report.features_importance,
                        "technical_indicators": report.technical_indicators,
                        "market_context": report.market_context,
                        "explanation": report.explanation,
                        "visualization_path": report.visualization_path,
                    },
                    f,
                    indent=4,
                )

            # Сохранение в памяти
            if pair not in self.reports:
                self.reports[pair] = []
            self.reports[pair].append(report)

        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

    def get_reports(
        self, pair: str, timeframe: Optional[str] = None
    ) -> List[DecisionReport]:
        """Получение отчетов"""
        try:
            reports = self.reports.get(pair, [])

            if timeframe:
                reports = [r for r in reports if r.timeframe == timeframe]

            return reports

        except Exception as e:
            logger.error(f"Error getting reports: {str(e)}")
            return []

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков"""
        try:
            features = pd.DataFrame()

            # Технические индикаторы
            features["rsi"] = talib.RSI(data["close"])
            features["macd"], features["macd_signal"], _ = talib.MACD(data["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                talib.BBANDS(data["close"])
            )
            features["atr"] = talib.ATR(data["high"], data["low"], data["close"])

            # Свечные характеристики
            features["body_size"] = abs(data["close"] - data["open"])
            features["upper_shadow"] = data["high"] - data[["open", "close"]].max(
                axis=1
            )
            features["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]
            features["is_bullish"] = (data["close"] > data["open"]).astype(int)

            # Объемные характеристики
            features["volume_ma"] = talib.SMA(data["volume"], timeperiod=20)
            features["volume_ratio"] = data["volume"] / features["volume_ma"]

            # Волатильность
            features["volatility"] = data["close"].pct_change().rolling(window=20).std()

            # Тренд
            features["trend"] = talib.ADX(data["high"], data["low"], data["close"])

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def predict_with_confidence(self, features: Dict) -> Tuple[str, float]:
        """Predict trade direction with confidence score.

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (direction, confidence)
        """
        try:
            if not self.model_trained:
                logger.warning("Model not trained yet. Using default confidence.")
                return "neutral", 0.5

            # Convert features to array
            feature_array = np.array([list(features.values())])
            feature_array = self.scaler.transform(feature_array)

            # Get predictions from both models
            bayes_pred = self.bayes_model.predict_proba(feature_array)[0]
            logistic_pred = self.logistic_model.predict_proba(feature_array)[0]

            # Combine predictions
            combined_probs = (bayes_pred + logistic_pred) / 2

            # Get direction and confidence
            direction = "buy" if combined_probs[1] > 0.5 else "sell"
            confidence = max(combined_probs)

            return direction, confidence

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return "neutral", 0.5

    def adjust_strategy_signal(
        self, signal: Dict, confidence: float, market_regime: str
    ) -> Dict:
        """Adjust strategy signal based on confidence and market regime.

        Args:
            signal: Original strategy signal
            confidence: Prediction confidence
            market_regime: Current market regime

        Returns:
            Adjusted signal dictionary
        """
        try:
            adjusted_signal = signal.copy()

            # Adjust position size based on confidence
            if "position_size" in adjusted_signal:
                adjusted_signal["position_size"] *= confidence

            # Adjust stop loss and take profit based on regime
            if market_regime == "trend":
                # Wider stops in trend
                if "stop_loss" in adjusted_signal:
                    adjusted_signal["stop_loss"] *= 1.2
                if "take_profit" in adjusted_signal:
                    adjusted_signal["take_profit"] *= 1.2
            elif market_regime == "volatility":
                # Tighter stops in volatility
                if "stop_loss" in adjusted_signal:
                    adjusted_signal["stop_loss"] *= 0.8
                if "take_profit" in adjusted_signal:
                    adjusted_signal["take_profit"] *= 0.8

            # Add confidence to signal
            adjusted_signal["confidence"] = confidence

            return adjusted_signal

        except Exception as e:
            logger.error(f"Error adjusting signal: {str(e)}")
            return signal

    def train_models(
        self, features: pd.DataFrame, targets: pd.Series, trade_results: List[Dict]
    ) -> None:
        """Train prediction models on historical data.

        Args:
            features: Feature DataFrame
            targets: Target series (1 for profitable trades, 0 for losses)
            trade_results: List of trade result dictionaries
        """
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Train models
            self.bayes_model.fit(scaled_features, targets)
            self.logistic_model.fit(scaled_features, targets)

            # Store trade history
            self.trade_history = trade_results

            # Update training status
            self.model_trained = True

            logger.info("Models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")

    def evaluate_performance(self) -> Dict:
        """Evaluate model performance on recent trades.

        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.trade_history:
                return {}

            # Calculate metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t["pnl"] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate average confidence for winning vs losing trades
            winning_confidences = [
                t["confidence"] for t in self.trade_history if t["pnl"] > 0
            ]
            losing_confidences = [
                t["confidence"] for t in self.trade_history if t["pnl"] <= 0
            ]

            avg_winning_confidence = (
                np.mean(winning_confidences) if winning_confidences else 0
            )
            avg_losing_confidence = (
                np.mean(losing_confidences) if losing_confidences else 0
            )

            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_winning_confidence": avg_winning_confidence,
                "avg_losing_confidence": avg_losing_confidence,
            }

        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            return {}

    def should_retrain(self) -> bool:
        """Check if models need retraining.

        Returns:
            Boolean indicating if retraining is needed
        """
        try:
            if not self.model_trained:
                return True

            # Get recent performance
            recent_trades = self.trade_history[-100:]  # Last 100 trades
            if len(recent_trades) < 100:
                return False

            # Calculate recent win rate
            recent_wins = len([t for t in recent_trades if t["pnl"] > 0])
            recent_win_rate = recent_wins / len(recent_trades)

            # Check if performance has degraded
            return recent_win_rate < 0.5

        except Exception as e:
            logger.error(f"Error checking retraining need: {str(e)}")
            return False

    def explain(self, decision: TradeDecision, data: Dict) -> str:
        """
        Формирование объяснения торгового решения.

        Args:
            decision: Торговое решение
            data: Дополнительные данные (сигналы, индикаторы и т.д.)

        Returns:
            str: Объяснение решения
        """
        try:
            # Формирование объяснения
            explanation = self._generate_explanation(decision, data)

            # Логирование
            self._log_explanation(decision, explanation, data)

            return explanation

        except Exception as e:
            logger.error(f"Error explaining decision: {str(e)}")
            return "Ошибка при формировании объяснения"

    def _generate_explanation(self, decision: TradeDecision, data: Dict) -> str:
        """Генерация объяснения решения"""
        parts = []

        # Основное решение
        parts.append(self._explain_action(decision))

        # Стратегия и режим
        if "strategy" in data and "regime" in data:
            parts.append(self._explain_strategy(data["strategy"], data["regime"]))

        # Индикаторы
        if "indicators" in data:
            parts.append(self._explain_indicators(data["indicators"]))

        # Активность китов
        if "whale_activity" in data:
            parts.append(self._explain_whale_activity(data["whale_activity"]))

        # Объемы
        if "volume_data" in data:
            parts.append(self._explain_volume(data["volume_data"]))

        return " ".join(parts)

    def _explain_action(self, decision: TradeDecision) -> str:
        """Объяснение действия"""
        if decision.action == "hold":
            return f"Принято решение воздержаться от торговли {decision.symbol} из-за низкой уверенности ({decision.confidence:.2%})"

        direction = "длинную" if decision.direction == "long" else "короткую"
        return (
            f"Открыта {direction} позиция по {decision.symbol} "
            f"с объемом {decision.volume:.4f} "
            f"и уверенностью {decision.confidence:.2%}. "
            f"Стоп-лосс: {decision.stop_loss:.2f}, "
            f"тейк-профит: {decision.take_profit:.2f}"
        )

    def _explain_strategy(self, strategy: Dict, regime: str) -> str:
        """Объяснение выбора стратегии"""
        return (
            f"Была выбрана стратегия {strategy['name']} "
            f"из-за текущего режима рынка '{regime}' "
            f"и подтверждения от {strategy['confirmations']} индикаторов"
        )

    def _explain_indicators(self, indicators: Dict) -> str:
        """Объяснение сигналов индикаторов"""
        signals = []
        for name, signal in indicators.items():
            if signal["value"] > 0:
                signals.append(f"{name} ({signal['value']:.2f})")

        if signals:
            return f"Подтверждающие сигналы: {', '.join(signals)}"
        return "Нет подтверждающих сигналов от индикаторов"

    def _explain_whale_activity(self, whale_data: Dict) -> str:
        """Объяснение активности китов"""
        if whale_data["active"]:
            return (
                f"Обнаружена активность китов: "
                f"{whale_data['volume']:.2f} {whale_data['side']} "
                f"на уровне {whale_data['price']:.2f}"
            )
        return "Активность китов не обнаружена"

    def _explain_volume(self, volume_data: Dict) -> str:
        """Объяснение объемов"""
        return (
            f"Объемы: {volume_data['current']:.2f} "
            f"({volume_data['change']:+.2%} к среднему)"
        )

    def _log_explanation(self, decision: TradeDecision, explanation: str, data: Dict):
        """Логирование объяснения"""
        log_entry = {
            "timestamp": decision.timestamp.isoformat(),
            "symbol": decision.symbol,
            "decision": {
                "action": decision.action,
                "direction": decision.direction,
                "volume": decision.volume,
                "confidence": decision.confidence,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
            },
            "explanation": explanation,
            "data": data,
        }

        logger.info(json.dumps(log_entry))

    def _is_three_white_soldiers(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'три белых солдата'."""
        try:
            if len(data) < 3:
                return False

            # Проверка трех последовательных бычьих свечей
            three_bullish = all(
                data["close"].iloc[-i] > data["open"].iloc[-i] for i in range(1, 4)
            )

            # Проверка роста закрытия
            rising_closes = all(
                data["close"].iloc[-i] > data["close"].iloc[-i - 1] for i in range(1, 3)
            )

            # Проверка размера тел свечей
            body_sizes = [
                abs(data["close"].iloc[-i] - data["open"].iloc[-i]) for i in range(1, 4)
            ]
            increasing_bodies = all(
                body_sizes[i] > body_sizes[i + 1] for i in range(len(body_sizes) - 1)
            )

            return three_bullish and rising_closes and increasing_bodies

        except Exception as e:
            logger.error(f"Error checking three white soldiers: {str(e)}")
            return False

    def _is_three_black_crows(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'три черные вороны'."""
        try:
            if len(data) < 3:
                return False

            # Проверка трех последовательных медвежьих свечей
            three_bearish = all(
                data["close"].iloc[-i] < data["open"].iloc[-i] for i in range(1, 4)
            )

            # Проверка падения закрытия
            falling_closes = all(
                data["close"].iloc[-i] < data["close"].iloc[-i - 1] for i in range(1, 3)
            )

            # Проверка размера тел свечей
            body_sizes = [
                abs(data["close"].iloc[-i] - data["open"].iloc[-i]) for i in range(1, 4)
            ]
            increasing_bodies = all(
                body_sizes[i] > body_sizes[i + 1] for i in range(len(body_sizes) - 1)
            )

            return three_bearish and falling_closes and increasing_bodies

        except Exception as e:
            logger.error(f"Error checking three black crows: {str(e)}")
            return False

    def _is_three_line_strike(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'трехлинейный удар'."""
        try:
            if len(data) < 4:
                return False

            # Проверка трех свечей в одном направлении
            first_three_same = all(
                (data["close"].iloc[-i] > data["open"].iloc[-i])
                == (data["close"].iloc[-1] > data["open"].iloc[-1])
                for i in range(2, 4)
            )

            # Проверка четвертой свечи
            fourth_opposite = (data["close"].iloc[-4] > data["open"].iloc[-4]) != (
                data["close"].iloc[-1] > data["open"].iloc[-1]
            )

            # Проверка поглощения
            engulfing = abs(data["close"].iloc[-4] - data["open"].iloc[-4]) > sum(
                abs(data["close"].iloc[-i] - data["open"].iloc[-i]) for i in range(1, 4)
            )

            return first_three_same and fourth_opposite and engulfing

        except Exception as e:
            logger.error(f"Error checking three line strike: {str(e)}")
            return False

    def _is_doji_star(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'звезда доджи'."""
        try:
            if len(data) < 3:
                return False

            # Проверка первой свечи
            first_candle = (
                abs(data["close"].iloc[-3] - data["open"].iloc[-3])
                > data["atr"].iloc[-3]
            )

            # Проверка доджи
            doji = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                < data["atr"].iloc[-2] * 0.1
            )

            # Проверка гэпа
            gap_up = data["low"].iloc[-2] > data["high"].iloc[-3]
            gap_down = data["high"].iloc[-2] < data["low"].iloc[-3]

            # Проверка третьей свечи
            third_candle = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1]
            )

            return first_candle and doji and (gap_up or gap_down) and third_candle

        except Exception as e:
            logger.error(f"Error checking doji star: {str(e)}")
            return False

    def _is_abandoned_baby(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'брошенное дитя'."""
        try:
            if len(data) < 3:
                return False

            # Проверка первой свечи
            first_candle = (
                abs(data["close"].iloc[-3] - data["open"].iloc[-3])
                > data["atr"].iloc[-3]
            )

            # Проверка доджи
            doji = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                < data["atr"].iloc[-2] * 0.1
            )

            # Проверка гэпа
            gap_up = data["low"].iloc[-2] > data["high"].iloc[-3]
            gap_down = data["high"].iloc[-2] < data["low"].iloc[-3]

            # Проверка третьей свечи
            third_candle = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1]
            )

            # Проверка противоположного направления
            opposite_direction = (data["close"].iloc[-3] > data["open"].iloc[-3]) != (
                data["close"].iloc[-1] > data["open"].iloc[-1]
            )

            return (
                first_candle
                and doji
                and (gap_up or gap_down)
                and third_candle
                and opposite_direction
            )

        except Exception as e:
            logger.error(f"Error checking abandoned baby: {str(e)}")
            return False

    def _is_gravestone_doji(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'надгробная плита'."""
        try:
            if len(data) < 1:
                return False

            # Проверка доджи
            doji = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                < data["atr"].iloc[-1] * 0.1
            )

            # Проверка верхней тени
            upper_shadow = data["high"].iloc[-1] - max(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )
            lower_shadow = (
                min(data["open"].iloc[-1], data["close"].iloc[-1])
                - data["low"].iloc[-1]
            )

            return doji and upper_shadow > lower_shadow * 3

        except Exception as e:
            logger.error(f"Error checking gravestone doji: {str(e)}")
            return False

    def _is_dragonfly_doji(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'стрекоза'."""
        try:
            if len(data) < 1:
                return False

            # Проверка доджи
            doji = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                < data["atr"].iloc[-1] * 0.1
            )

            # Проверка нижней тени
            upper_shadow = data["high"].iloc[-1] - max(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )
            lower_shadow = (
                min(data["open"].iloc[-1], data["close"].iloc[-1])
                - data["low"].iloc[-1]
            )

            return doji and lower_shadow > upper_shadow * 3

        except Exception as e:
            logger.error(f"Error checking dragonfly doji: {str(e)}")
            return False

    def _is_marubozu(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'марубозу'."""
        try:
            if len(data) < 1:
                return False

            # Проверка отсутствия теней
            no_upper_shadow = data["high"].iloc[-1] == max(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )
            no_lower_shadow = data["low"].iloc[-1] == min(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )

            return no_upper_shadow and no_lower_shadow

        except Exception as e:
            logger.error(f"Error checking marubozu: {str(e)}")
            return False

    def _is_spinning_top(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'волчок'."""
        try:
            if len(data) < 1:
                return False

            # Проверка маленького тела
            body_size = abs(data["close"].iloc[-1] - data["open"].iloc[-1])
            total_range = data["high"].iloc[-1] - data["low"].iloc[-1]

            small_body = body_size < total_range * 0.3

            # Проверка длинных теней
            upper_shadow = data["high"].iloc[-1] - max(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )
            lower_shadow = (
                min(data["open"].iloc[-1], data["close"].iloc[-1])
                - data["low"].iloc[-1]
            )

            long_shadows = upper_shadow > body_size and lower_shadow > body_size

            return small_body and long_shadows

        except Exception as e:
            logger.error(f"Error checking spinning top: {str(e)}")
            return False

    def _is_belt_hold(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'пояс'."""
        try:
            if len(data) < 1:
                return False

            # Проверка длинного тела
            body_size = abs(data["close"].iloc[-1] - data["open"].iloc[-1])
            total_range = data["high"].iloc[-1] - data["low"].iloc[-1]

            long_body = body_size > total_range * 0.7

            # Проверка отсутствия одной из теней
            upper_shadow = data["high"].iloc[-1] - max(
                data["open"].iloc[-1], data["close"].iloc[-1]
            )
            lower_shadow = (
                min(data["open"].iloc[-1], data["close"].iloc[-1])
                - data["low"].iloc[-1]
            )

            no_upper = upper_shadow < body_size * 0.1
            no_lower = lower_shadow < body_size * 0.1

            return long_body and (no_upper or no_lower)

        except Exception as e:
            logger.error(f"Error checking belt hold: {str(e)}")
            return False

    def _is_kicking(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'удар'."""
        try:
            if len(data) < 2:
                return False

            # Проверка марубозу
            first_marubozu = self._is_marubozu(data.iloc[-2:-1])
            second_marubozu = self._is_marubozu(data.iloc[-1:])

            # Проверка противоположного направления
            opposite_direction = (data["close"].iloc[-2] > data["open"].iloc[-2]) != (
                data["close"].iloc[-1] > data["open"].iloc[-1]
            )

            # Проверка гэпа
            gap = (data["low"].iloc[-1] > data["high"].iloc[-2]) or (
                data["high"].iloc[-1] < data["low"].iloc[-2]
            )

            return first_marubozu and second_marubozu and opposite_direction and gap

        except Exception as e:
            logger.error(f"Error checking kicking: {str(e)}")
            return False

    def _is_breakaway(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'разрыв'."""
        try:
            if len(data) < 5:
                return False

            # Проверка первой свечи
            first_candle = (
                abs(data["close"].iloc[-5] - data["open"].iloc[-5])
                > data["atr"].iloc[-5]
            )

            # Проверка второй свечи
            second_candle = (
                abs(data["close"].iloc[-4] - data["open"].iloc[-4])
                > data["atr"].iloc[-4] * 0.5
            )

            # Проверка третьей свечи
            third_candle = (
                abs(data["close"].iloc[-3] - data["open"].iloc[-3])
                > data["atr"].iloc[-3] * 0.3
            )

            # Проверка четвертой свечи
            fourth_candle = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                > data["atr"].iloc[-2] * 0.2
            )

            # Проверка пятой свечи
            fifth_candle = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1]
            )

            # Проверка направления
            same_direction = all(
                (data["close"].iloc[-i] > data["open"].iloc[-i])
                == (data["close"].iloc[-1] > data["open"].iloc[-1])
                for i in range(2, 6)
            )

            return (
                first_candle
                and second_candle
                and third_candle
                and fourth_candle
                and fifth_candle
                and same_direction
            )

        except Exception as e:
            logger.error(f"Error checking breakaway: {str(e)}")
            return False

    def _is_meeting_lines(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'встречные линии'."""
        try:
            if len(data) < 2:
                return False

            # Проверка длинных свечей
            first_long = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                > data["atr"].iloc[-2]
            )
            second_long = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1]
            )

            # Проверка противоположного направления
            opposite_direction = (data["close"].iloc[-2] > data["open"].iloc[-2]) != (
                data["close"].iloc[-1] > data["open"].iloc[-1]
            )

            # Проверка закрытия на одном уровне
            same_close = (
                abs(data["close"].iloc[-2] - data["close"].iloc[-1])
                < data["atr"].iloc[-1] * 0.1
            )

            return first_long and second_long and opposite_direction and same_close

        except Exception as e:
            logger.error(f"Error checking meeting lines: {str(e)}")
            return False

    def _is_in_neck(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'в шее'."""
        try:
            if len(data) < 2:
                return False

            # Проверка первой свечи
            first_candle = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                > data["atr"].iloc[-2]
            )

            # Проверка второй свечи
            second_candle = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1] * 0.5
            )

            # Проверка направления
            bearish = data["close"].iloc[-2] < data["open"].iloc[-2]

            # Проверка закрытия
            close_near_low = (
                abs(data["close"].iloc[-1] - data["low"].iloc[-2])
                < data["atr"].iloc[-1] * 0.1
            )

            return first_candle and second_candle and bearish and close_near_low

        except Exception as e:
            logger.error(f"Error checking in neck: {str(e)}")
            return False

    def _is_on_neck(self, data: pd.DataFrame) -> bool:
        """Проверка паттерна 'на шее'."""
        try:
            if len(data) < 2:
                return False

            # Проверка первой свечи
            first_candle = (
                abs(data["close"].iloc[-2] - data["open"].iloc[-2])
                > data["atr"].iloc[-2]
            )

            # Проверка второй свечи
            second_candle = (
                abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                > data["atr"].iloc[-1] * 0.5
            )

            # Проверка направления
            bearish = data["close"].iloc[-2] < data["open"].iloc[-2]

            # Проверка закрытия
            close_near_low = (
                abs(data["close"].iloc[-1] - data["low"].iloc[-2])
                < data["atr"].iloc[-1] * 0.1
            )

            return first_candle and second_candle and bearish and close_near_low

        except Exception as e:
            logger.error(f"Error checking on neck: {str(e)}")
            return False

    def _identify_fakeouts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Идентификация фейковых пробоев"""
        try:
            fakeouts = []

            # Расчет уровней поддержки и сопротивления
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data["close"].values)

            # Поиск пробоев
            for i in range(1, len(data)):
                # Пробой верхней границы
                if (
                    data["close"].iloc[i - 1] < bb_upper[i - 1]
                    and data["close"].iloc[i] > bb_upper[i]
                ):
                    # Проверка на возврат
                    if data["close"].iloc[i + 1 : i + 5].min() < bb_middle[i]:
                        fakeouts.append(
                            {
                                "type": "bullish",
                                "index": i,
                                "price": data["close"].iloc[i],
                                "strength": (data["close"].iloc[i] - bb_upper[i])
                                / bb_upper[i],
                            }
                        )

                # Пробой нижней границы
                if (
                    data["close"].iloc[i - 1] > bb_lower[i - 1]
                    and data["close"].iloc[i] < bb_lower[i]
                ):
                    # Проверка на возврат
                    if data["close"].iloc[i + 1 : i + 5].max() > bb_middle[i]:
                        fakeouts.append(
                            {
                                "type": "bearish",
                                "index": i,
                                "price": data["close"].iloc[i],
                                "strength": (bb_lower[i] - data["close"].iloc[i])
                                / bb_lower[i],
                            }
                        )

            return fakeouts
        except Exception as e:
            logger.error(f"Ошибка идентификации фейковых пробоев: {e}")
            return []

    def _calculate_market_structure(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Расчет уровней поддержки и сопротивления"""
        try:
            # Используем полосы Боллинджера для определения уровней
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data["close"].values)

            # Находим локальные максимумы и минимумы
            peaks, _ = talib.MAX(data["high"].values, timeperiod=20)
            troughs, _ = talib.MIN(data["low"].values, timeperiod=20)

            # Группируем близкие уровни
            support_levels = self._group_price_levels(troughs)
            resistance_levels = self._group_price_levels(peaks)

            return {
                "support": support_levels,
                "resistance": resistance_levels,
                "bb_upper": float(bb_upper[-1]),
                "bb_middle": float(bb_middle[-1]),
                "bb_lower": float(bb_lower[-1]),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета рыночной структуры: {e}")
            return {
                "support": [],
                "resistance": [],
                "bb_upper": 0.0,
                "bb_middle": 0.0,
                "bb_lower": 0.0,
            }

    def _calculate_liquidity_zones(
        self, data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, float]]]:
        """Расчет зон ликвидности"""
        try:
            zones = []

            # Анализ объемов на уровнях цены
            price_levels = np.linspace(data["low"].min(), data["high"].max(), 50)
            volume_profile = self._calculate_volume_profile(data)

            # Находим зоны с высокой ликвидностью
            for i in range(len(price_levels) - 1):
                level_volume = volume_profile[i]
                if level_volume > np.mean(volume_profile) * 1.5:
                    zones.append(
                        {
                            "price_level": float(price_levels[i]),
                            "volume": float(level_volume),
                            "strength": float(level_volume / np.mean(volume_profile)),
                        }
                    )

            return {"zones": zones}
        except Exception as e:
            logger.error(f"Ошибка расчета зон ликвидности: {e}")
            return {"zones": []}

    def _calculate_volume_profile(self, data: pd.DataFrame) -> np.ndarray:
        """Расчет профиля объема"""
        try:
            # Создаем гистограмму объемов по уровням цены
            price_levels = np.linspace(data["low"].min(), data["high"].max(), 50)
            volume_profile = np.zeros_like(price_levels)

            for i in range(len(data)):
                price = data["close"].iloc[i]
                volume = data["volume"].iloc[i]

                # Находим ближайший уровень цены
                level_idx = np.abs(price_levels - price).argmin()
                volume_profile[level_idx] += volume

            return volume_profile
        except Exception as e:
            logger.error(f"Ошибка расчета профиля объема: {e}")
            return np.zeros(50)

    def _calculate_fractals(
        self, data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Расчет фракталов"""
        try:
            fractals = []

            # Поиск фракталов на 5 свечах
            for i in range(2, len(data) - 2):
                # Бычий фрактал
                if (
                    data["low"].iloc[i - 2] > data["low"].iloc[i]
                    and data["low"].iloc[i - 1] > data["low"].iloc[i]
                    and data["low"].iloc[i + 1] > data["low"].iloc[i]
                    and data["low"].iloc[i + 2] > data["low"].iloc[i]
                ):
                    fractals.append(
                        {
                            "type": "bullish",
                            "index": i,
                            "price": float(data["low"].iloc[i]),
                            "strength": float(
                                (
                                    data["low"].iloc[i - 2 : i + 3].max()
                                    - data["low"].iloc[i]
                                )
                                / data["low"].iloc[i]
                            ),
                        }
                    )

                # Медвежий фрактал
                if (
                    data["high"].iloc[i - 2] < data["high"].iloc[i]
                    and data["high"].iloc[i - 1] < data["high"].iloc[i]
                    and data["high"].iloc[i + 1] < data["high"].iloc[i]
                    and data["high"].iloc[i + 2] < data["high"].iloc[i]
                ):
                    fractals.append(
                        {
                            "type": "bearish",
                            "index": i,
                            "price": float(data["high"].iloc[i]),
                            "strength": float(
                                (
                                    data["high"].iloc[i]
                                    - data["high"].iloc[i - 2 : i + 3].min()
                                )
                                / data["high"].iloc[i]
                            ),
                        }
                    )

            return {"fractals": fractals}
        except Exception as e:
            logger.error(f"Ошибка расчета фракталов: {e}")
            return {"fractals": []}

    def _calculate_imbalance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Расчет дисбаланса в стакане"""
        try:
            # Используем объемы для оценки дисбаланса
            buy_volume = data[data["close"] > data["open"]]["volume"].sum()
            sell_volume = data[data["close"] < data["open"]]["volume"].sum()

            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return {"imbalance": 0.0}

            imbalance = (buy_volume - sell_volume) / total_volume
            return {"imbalance": float(imbalance)}
        except Exception as e:
            logger.error(f"Ошибка расчета дисбаланса: {e}")
            return {"imbalance": 0.0}

    def _group_price_levels(
        self, levels: np.ndarray, threshold: float = 0.01
    ) -> List[float]:
        """Группировка близких ценовых уровней"""
        try:
            if len(levels) == 0:
                return []

            # Сортируем уровни
            sorted_levels = np.sort(levels)
            grouped_levels = []
            current_group = [sorted_levels[0]]

            for level in sorted_levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] <= threshold:
                    current_group.append(level)
                else:
                    grouped_levels.append(float(np.mean(current_group)))
                    current_group = [level]

            if current_group:
                grouped_levels.append(float(np.mean(current_group)))

            return grouped_levels
        except Exception as e:
            logger.error(f"Ошибка группировки ценовых уровней: {e}")
            return []
