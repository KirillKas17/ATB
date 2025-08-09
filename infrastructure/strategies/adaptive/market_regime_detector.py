"""
Детектор рыночных режимов для адаптивных стратегий
"""

from typing import Any, Dict, Optional

from loguru import logger
import pandas as pd

from domain.type_definitions.strategy_types import MarketRegime
# Убираем неправильный импорт - MarketRegimeAgent не существует в этом модуле
# from infrastructure.agents.market_regime import MarketRegimeAgent


class MarketRegimeDetector:
    """Детектор рыночных режимов"""

    def __init__(self, market_regime_agent: Optional[Any] = None) -> None:
        self.market_regime_agent = market_regime_agent

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Определение рыночного режима"""
        try:
            if self.market_regime_agent:
                regime_prediction = self.market_regime_agent.predict_regime(data)
                return MarketRegime(regime_prediction)
            else:
                return self._fallback_regime_detection(data)
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _fallback_regime_detection(self, data: pd.DataFrame) -> MarketRegime:
        """Улучшенное определение режима с адаптивными порогами"""
        try:
            # Безопасная проверка данных
            if len(data) < 50:
                return MarketRegime.SIDEWAYS
                
            # Получаем среднюю цену для нормализации порогов
            avg_price = data["close"].mean()
            price_level_factor = max(0.1, min(10.0, avg_price / 100.0))  # Нормализация к $100
            
            # Анализ тренда с множественными подтверждениями
            trend_strength = 0.0
            trend_direction = 0  # -1: down, 0: sideways, 1: up
            ema_20_val = 0.0
            ema_50_val = 0.0
            
            try:
                ema_20 = data["close"].ewm(span=20).mean()
                ema_50 = data["close"].ewm(span=50).mean()
                sma_200 = data["close"].rolling(window=200).mean()
                
                if len(ema_20) > 0 and len(ema_50) > 0:
                    ema_20_val = ema_20.iloc[-1]
                    ema_50_val = ema_50.iloc[-1]
                    
                    if ema_50_val > 0:
                        # Адаптивный порог силы тренда
                        trend_strength = abs(ema_20_val - ema_50_val) / ema_50_val
                        
                        # Определяем направление тренда
                        if ema_20_val > ema_50_val:
                            trend_direction = 1
                        elif ema_20_val < ema_50_val:
                            trend_direction = -1
                            
                        # Дополнительное подтверждение долгосрочным трендом
                        if len(sma_200) > 0 and not pd.isna(sma_200.iloc[-1]):
                            sma_200_val = sma_200.iloc[-1]
                            # Усиливаем сигнал если краткосрочный тренд совпадает с долгосрочным
                            if (trend_direction > 0 and ema_20_val > sma_200_val) or \
                               (trend_direction < 0 and ema_20_val < sma_200_val):
                                trend_strength *= 1.5
                                
            except (IndexError, TypeError, ZeroDivisionError):
                pass
                
            # Анализ волатильности с адаптивным порогом
            volatility = 0.0
            volatility_percentile = 0.0
            try:
                if len(data) >= 20:
                    pct_change = data["close"].pct_change()
                    if len(pct_change) >= 20:
                        rolling_std = pct_change.rolling(20).std()
                        if len(rolling_std) > 0:
                            volatility = rolling_std.iloc[-1]
                            # Сравниваем с исторической волатильностью
                            if len(rolling_std) >= 100:
                                volatility_percentile = (rolling_std.iloc[-1] > rolling_std.rolling(100).quantile(0.8)).sum()
            except (IndexError, TypeError):
                pass
                
            # Анализ объема с улучшенной логикой
            volume_ratio = 1.0
            volume_trend = 0  # -1: decreasing, 0: stable, 1: increasing
            try:
                if len(data) >= 20 and "volume" in data.columns:
                    current_volume = data["volume"].iloc[-1]
                    avg_volume_20 = data["volume"].rolling(20).mean().iloc[-1]
                    avg_volume_5 = data["volume"].rolling(5).mean().iloc[-1]
                    
                    if avg_volume_20 > 0:
                        volume_ratio = current_volume / avg_volume_20
                        # Определяем тренд объема
                        if avg_volume_5 > avg_volume_20 * 1.2:
                            volume_trend = 1
                        elif avg_volume_5 < avg_volume_20 * 0.8:
                            volume_trend = -1
            except (IndexError, TypeError, ZeroDivisionError):
                pass
            
            # Адаптивные пороги на основе характеристик актива
            trend_threshold = 0.02 * price_level_factor  # Базовый 2%, адаптируется к цене
            volatility_threshold = 0.02 * price_level_factor  # Базовый 2%
            volume_threshold = 1.5  # Менее волатильный порог
            
            # Анализ последовательности свечей для подтверждения тренда
            consecutive_moves = 0
            if len(data) >= 5:
                closes = data["close"].tail(5)
                if trend_direction > 0:
                    # Проверяем восходящий тренд
                    consecutive_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] > closes.iloc[i-1])
                elif trend_direction < 0:
                    # Проверяем нисходящий тренд  
                    consecutive_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] < closes.iloc[i-1])
            
            # Принятие решения с множественными критериями
            if trend_strength > trend_threshold and consecutive_moves >= 3:
                # Сильный тренд с подтверждением
                return (
                    MarketRegime.TRENDING_UP
                    if trend_direction > 0
                    else MarketRegime.TRENDING_DOWN
                )
            elif volatility > volatility_threshold or volatility_percentile > 0:
                # Высокая волатильность
                return MarketRegime.VOLATILE
            elif volume_ratio > volume_threshold and volume_trend > 0:
                # Прорыв с ростом объема
                return MarketRegime.BREAKOUT
            else:
                # Боковое движение
                return MarketRegime.SIDEWAYS
        except Exception as e:
            logger.error(f"Error in fallback regime detection: {str(e)}")
            return MarketRegime.SIDEWAYS

    def analyze_market_context(
        self, data: pd.DataFrame, regime: MarketRegime
    ) -> Dict[str, Any]:
        """Анализ рыночного контекста"""
        try:
            # Безопасная проверка данных
            if len(data) < 50:
                return {
                    "regime": regime,
                    "volatility": 0.0,
                    "trend_strength": 0.0,
                    "volume_profile": {"current_volume": 0.0, "avg_volume": 0.0, "volume_trend": "neutral"},
                    "liquidity_conditions": {"spread": 0.001, "depth": 1.0, "liquidity_score": 0.8},
                    "market_sentiment": 0.5,
                }
                
            # Безопасный расчет волатильности
            volatility = 0.0
            try:
                if len(data) >= 20:
                    pct_change = data["close"].pct_change()
                    if len(pct_change) >= 20:
                        rolling_std = pct_change.rolling(20).std()
                        if len(rolling_std) > 0:
                            volatility = rolling_std.iloc[-1]
            except (IndexError, TypeError):
                pass
                
            # Безопасный расчет силы тренда
            trend_strength = 0.0
            try:
                ema_20 = data["close"].ewm(span=20).mean()
                ema_50 = data["close"].ewm(span=50).mean()
                if len(ema_20) > 0 and len(ema_50) > 0:
                    ema_20_val = ema_20.iloc[-1]
                    ema_50_val = ema_50.iloc[-1]
                    if ema_50_val > 0:
                        trend_strength = abs(ema_20_val - ema_50_val) / ema_50_val
            except (IndexError, TypeError, ZeroDivisionError):
                pass
                
            # Безопасный анализ объема
            current_volume = 0.0
            avg_volume = 0.0
            volume_trend = "neutral"
            
            try:
                if len(data) >= 20 and "volume" in data.columns:
                    current_volume = float(data["volume"].iloc[-1])
                    volume_rolling = data["volume"].rolling(20).mean()
                    if len(volume_rolling) > 0:
                        avg_volume = float(volume_rolling.iloc[-1])
                        volume_trend = "increasing" if current_volume > avg_volume else "decreasing"
            except (IndexError, TypeError):
                pass
                
            volume_profile = {
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_trend": volume_trend,
            }
            liquidity_conditions = {
                "spread": 0.001,  # Placeholder
                "depth": 1.0,  # Placeholder
                "liquidity_score": 0.8,  # Placeholder
            }
            market_sentiment = 0.5  # Placeholder
            return {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "volume_profile": volume_profile,
                "liquidity_conditions": liquidity_conditions,
                "market_sentiment": market_sentiment,
            }
        except Exception as e:
            logger.error(f"Error analyzing market context: {str(e)}")
            return {
                "regime": regime,
                "volatility": 0.02,
                "trend_strength": 0.0,
                "volume_profile": {},
                "liquidity_conditions": {},
                "market_sentiment": 0.5,
            }
