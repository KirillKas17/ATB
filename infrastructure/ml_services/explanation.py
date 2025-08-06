"""
Модуль для объяснения решений и интерпретации результатов.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame, Series

# Настройка логирования
logger = logging.getLogger(__name__)


class DecisionExplainer:
    """Класс для объяснения торговых решений"""

    def __init__(self) -> None:
        self.feature_importance: Dict[str, float] = {}
        self.decision_history: List[Dict] = []
        self.explanation_templates = {
            "buy": {
                "strong": "Сильный сигнал на покупку: {reason}",
                "moderate": "Умеренный сигнал на покупку: {reason}",
                "weak": "Слабый сигнал на покупку: {reason}",
            },
            "sell": {
                "strong": "Сильный сигнал на продажу: {reason}",
                "moderate": "Умеренный сигнал на продажу: {reason}",
                "weak": "Слабый сигнал на продажу: {reason}",
            },
            "hold": {
                "strong": "Рекомендуется удержание позиции: {reason}",
                "moderate": "Нейтральная позиция: {reason}",
                "weak": "Неопределенность в направлении: {reason}",
            },
        }

    def explain_decision(
        self,
        decision: Dict[str, Any],
        market_data: DataFrame,
        indicators: Dict[str, float],
    ) -> str:
        """Объяснение торгового решения"""
        try:
            action = decision.get("action", "hold")
            confidence = decision.get("confidence", 0.0)
            # Определение силы сигнала
            if confidence > 0.7:
                strength = "strong"
            elif confidence > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            # Генерация объяснения
            reason = self._generate_reason(action, market_data, indicators, decision)
            # Использование шаблона
            template = self.explanation_templates[action][strength]
            explanation = template.format(reason=reason)
            # Добавление деталей
            details = self._add_technical_details(indicators, decision)
            if details:
                explanation += f" {details}"
            return explanation
        except Exception as e:
            logger.error(f"Ошибка при объяснении решения: {e}")
            return "Не удалось сгенерировать объяснение"

    def _generate_reason(
        self,
        action: str,
        market_data: DataFrame,
        indicators: Dict[str, float],
        decision: Dict[str, Any],
    ) -> str:
        """Генерация причины решения"""
        reasons = []
        if action == "buy":
            reasons.extend(self._get_buy_reasons(indicators, market_data))
        elif action == "sell":
            reasons.extend(self._get_sell_reasons(indicators, market_data))
        else:  # hold
            reasons.extend(self._get_hold_reasons(indicators, market_data))
        # Добавление рыночного режима
        regime = decision.get("market_regime", "unknown")
        if regime != "unknown":
            reasons.append(f"рыночный режим: {regime}")
        # Добавление паттернов
        patterns = decision.get("patterns", [])
        if patterns:
            pattern_str = ", ".join(patterns[:3])  # Максимум 3 паттерна
            reasons.append(f"паттерны: {pattern_str}")
        return "; ".join(reasons) if reasons else "технический анализ"

    def _get_buy_reasons(
        self, indicators: Dict[str, float], market_data: DataFrame
    ) -> List[str]:
        """Получение причин для покупки"""
        reasons = []
        # RSI
        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            reasons.append("RSI показывает перепроданность")
        elif rsi < 50:
            reasons.append("RSI ниже нейтрального уровня")
        # MACD
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:
            reasons.append("MACD выше сигнальной линии")
        # Bollinger Bands
        close = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
        bb_lower = indicators.get("bb_lower", 0)
        if close <= bb_lower:
            reasons.append("цена у нижней полосы Боллинджера")
        # Moving Averages
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        if sma_20 > sma_50:
            reasons.append("краткосрочная MA выше долгосрочной")
        # Volume
        volume_ratio = indicators.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            reasons.append("повышенный объем торгов")
        return reasons

    def _get_sell_reasons(
        self, indicators: Dict[str, float], market_data: DataFrame
    ) -> List[str]:
        """Получение причин для продажи"""
        reasons = []
        # RSI
        rsi = indicators.get("rsi", 50)
        if rsi > 70:
            reasons.append("RSI показывает перекупленность")
        elif rsi > 50:
            reasons.append("RSI выше нейтрального уровня")
        # MACD
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd < macd_signal:
            reasons.append("MACD ниже сигнальной линии")
        # Bollinger Bands
        close = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
        bb_upper = indicators.get("bb_upper", 0)
        if close >= bb_upper:
            reasons.append("цена у верхней полосы Боллинджера")
        # Moving Averages
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        if sma_20 < sma_50:
            reasons.append("краткосрочная MA ниже долгосрочной")
        # Volume
        volume_ratio = indicators.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            reasons.append("повышенный объем торгов")
        return reasons

    def _get_hold_reasons(
        self, indicators: Dict[str, float], market_data: DataFrame
    ) -> List[str]:
        """Получение причин для удержания"""
        reasons = []
        # RSI
        rsi = indicators.get("rsi", 50)
        if 40 <= rsi <= 60:
            reasons.append("RSI в нейтральной зоне")
        # MACD
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if abs(macd - macd_signal) < 0.001:
            reasons.append("MACD близок к сигнальной линии")
        # Bollinger Bands
        close = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
        bb_middle = indicators.get("bb_middle", 0)
        if abs(close - bb_middle) / bb_middle < 0.02:
            reasons.append("цена близка к средней полосе Боллинджера")
        # Moving Averages
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        if abs(sma_20 - sma_50) / sma_50 < 0.01:
            reasons.append("MA близки друг к другу")
        # Volume
        volume_ratio = indicators.get("volume_ratio", 1.0)
        if 0.8 <= volume_ratio <= 1.2:
            reasons.append("нормальный объем торгов")
        return reasons

    def _add_technical_details(
        self, indicators: Dict[str, float], decision: Dict[str, Any]
    ) -> str:
        """Добавление технических деталей"""
        details = []
        # Уровень уверенности
        confidence = decision.get("confidence", 0.0)
        if confidence > 0.8:
            details.append("высокая уверенность")
        elif confidence > 0.6:
            details.append("средняя уверенность")
        else:
            details.append("низкая уверенность")
        # Риск
        risk_score = decision.get("risk_score", 1.0)
        if risk_score < 0.3:
            details.append("низкий риск")
        elif risk_score < 0.7:
            details.append("средний риск")
        else:
            details.append("высокий риск")
        # Количество источников
        sources = decision.get("sources", [])
        if sources:
            details.append(f"{len(sources)} источников сигналов")
        return f"({', '.join(details)})"

    def explain_feature_importance(self, feature_importance: Dict[str, float]) -> str:
        """Объяснение важности признаков"""
        try:
            if not feature_importance:
                return "Нет данных о важности признаков"
            # Сортировка по важности
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            # Топ-5 признаков
            top_features = sorted_features[:5]
            explanation = "Наиболее важные признаки: "
            feature_descriptions = []
            for feature, importance in top_features:
                description = self._get_feature_description(feature, importance)
                feature_descriptions.append(description)
            explanation += "; ".join(feature_descriptions)
            return explanation
        except Exception as e:
            logger.error(f"Ошибка при объяснении важности признаков: {e}")
            return "Не удалось объяснить важность признаков"

    def _get_feature_description(self, feature: str, importance: float) -> str:
        """Получение описания признака"""
        feature_descriptions = {
            "rsi": "RSI (относительная сила)",
            "macd": "MACD (схождение/расхождение)",
            "bb_position": "позиция относительно полос Боллинджера",
            "volume_ratio": "соотношение объемов",
            "atr": "средний истинный диапазон",
            "sma_20": "скользящая средняя 20 периодов",
            "sma_50": "скользящая средняя 50 периодов",
            "ema_12": "экспоненциальная MA 12 периодов",
            "ema_26": "экспоненциальная MA 26 периодов",
            "stoch_k": "стохастический осциллятор %K",
            "stoch_d": "стохастический осциллятор %D",
            "williams_r": "Williams %R",
            "cci": "индекс товарного канала",
            "adx": "индекс направленного движения",
            "plus_di": "положительный DI",
            "minus_di": "отрицательный DI",
        }
        description = feature_descriptions.get(feature, feature)
        return f"{description} ({importance:.3f})"

    def explain_market_regime(self, regime: str, confidence: float) -> str:
        """Объяснение рыночного режима"""
        regime_descriptions = {
            "trend": "трендовый режим с четким направлением",
            "sideways": "боковой режим без четкого направления",
            "volatile": "волатильный режим с резкими движениями",
            "unknown": "неопределенный режим",
        }
        description = regime_descriptions.get(regime, regime)
        if confidence > 0.8:
            confidence_level = "высокая уверенность"
        elif confidence > 0.6:
            confidence_level = "средняя уверенность"
        else:
            confidence_level = "низкая уверенность"
        return f"Рыночный режим: {description} ({confidence_level})"

    def explain_patterns(self, patterns: List[str]) -> str:
        """Объяснение паттернов свечей"""
        if not patterns:
            return "Паттерны свечей не обнаружены"
        pattern_descriptions = {
            "three_white_soldiers": "три белых солдата (бычий)",
            "three_black_crows": "три черных ворона (медвежий)",
            "doji_star": "звезда доджи (разворот)",
            "gravestone_doji": "надгробный доджи (медвежий)",
            "dragonfly_doji": "доджи стрекоза (бычий)",
            "marubozu": "марубозу (сильный сигнал)",
            "spinning_top": "волчок (неопределенность)",
            "belt_hold": "пояс (сильный сигнал)",
            "kicking": "пинок (разворот)",
            "breakaway": "разрыв (продолжение)",
            "meeting_lines": "встречные линии (разворот)",
            "in_neck": "в шее (медвежий)",
            "on_neck": "на шее (медвежий)",
        }
        descriptions = []
        for pattern in patterns[:3]:  # Максимум 3 паттерна
            desc = pattern_descriptions.get(pattern, pattern)
            descriptions.append(desc)
        return f"Обнаружены паттерны: {', '.join(descriptions)}"

    def create_summary_report(
        self,
        decision: Dict[str, Any],
        market_data: DataFrame,
        indicators: Dict[str, float],
    ) -> Dict[str, str]:
        """Создание сводного отчета"""
        try:
            report = {}
            # Основное объяснение
            report["main_explanation"] = self.explain_decision(
                decision, market_data, indicators
            )
            # Рыночный режим
            regime = decision.get("market_regime", "unknown")
            regime_confidence = decision.get("regime_confidence", 0.0)
            report["market_regime"] = self.explain_market_regime(
                regime, regime_confidence
            )
            # Паттерны
            patterns = decision.get("patterns", [])
            report["patterns"] = self.explain_patterns(patterns)
            # Важность признаков
            if self.feature_importance:
                report["feature_importance"] = self.explain_feature_importance(
                    self.feature_importance
                )
            # Технические детали
            report["technical_details"] = self._add_technical_details(
                indicators, decision
            )
            return report
        except Exception as e:
            logger.error(f"Ошибка при создании сводного отчета: {e}")
            return {"error": "Не удалось создать отчет"}
