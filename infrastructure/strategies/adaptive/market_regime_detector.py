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

    def __init__(self, market_regime_agent: Optional[Any] = None):
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
        """Резервное определение режима"""
        try:
            # Безопасная проверка данных
            if len(data) < 50:
                return MarketRegime.SIDEWAYS
                
            # Анализ тренда
            trend_strength = 0.0
            ema_20_val = 0.0
            ema_50_val = 0.0
            
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
                
            # Анализ волатильности
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
                
            # Анализ объема
            volume_ratio = 1.0
            try:
                if len(data) >= 20 and "volume" in data.columns:
                    current_volume = data["volume"].iloc[-1]
                    avg_volume = data["volume"].rolling(20).mean().iloc[-1]
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
            except (IndexError, TypeError, ZeroDivisionError):
                pass
                
            if trend_strength > 0.05:
                return (
                    MarketRegime.TRENDING_UP
                    if ema_20_val > ema_50_val
                    else MarketRegime.TRENDING_DOWN
                )
            elif volatility > 0.03:
                return MarketRegime.VOLATILE
            elif volume_ratio > 1.5:
                return MarketRegime.BREAKOUT
            else:
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
