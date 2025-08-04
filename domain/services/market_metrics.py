"""
Доменный сервис для расчета метрик рынка.
"""

import threading
from datetime import datetime
from typing import Any, Dict, Optional, Union, cast

from shared.numpy_utils import np
import pandas as pd
from pandas import Series

from domain.types.market_metrics_types import (
    LiquidityMetrics,
    MarketMetricsResult,
    MarketStressMetrics,
    MomentumMetrics,
    TrendDirection,
    TrendMetrics,
    VolatilityMetrics,
    VolatilityTrend,
    VolumeMetrics,
    VolumeTrend,
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _safe_series(series: Any) -> pd.Series:
    """Безопасно преобразует данные в pandas Series."""
    if isinstance(series, Series):
        return series
    elif isinstance(series, (list, tuple)):
        return pd.Series(series)
    else:
        return pd.Series(dtype=float)


class MarketMetricsService:
    """Промышленный сервис для расчета рыночных метрик."""

    DEFAULT_CONFIG = {
        "volatility_window": 20,
        "trend_window": 30,
        "volume_window": 20,
        "correlation_window": 30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self._lock = threading.Lock()

    def calculate_volatility_metrics(
        self, data: Union[pd.DataFrame, None]
    ) -> VolatilityMetrics:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return VolatilityMetrics(
                current_volatility=0.0,
                historical_volatility=0.0,
                volatility_percentile=0.0,
                volatility_trend=VolatilityTrend.STABLE,
            )
        window = self.config["volatility_window"]
        closes_result = _safe_series(data.loc[:, "close"] if "close" in data.columns else pd.Series(dtype=float))
        # Безопасно приводим к числовому типу
        if hasattr(closes_result, 'dtype') and str(closes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes = pd.to_numeric(closes_result, errors="coerce").fillna(0.0)
            else:
                closes = closes_result
        else:
            closes = closes_result
        returns = closes.pct_change().dropna()
        if len(returns) >= window:
            current_vol = returns.iloc[-window:].std() * np.sqrt(window)
        else:
            current_vol = 0.0
        hist_vol = returns.std() * np.sqrt(len(returns)) if len(returns) > 0 else 0.0
        percentile = np.clip(
            (current_vol / hist_vol) if hist_vol > 0 else 0.0, 0.0, 1.0
        )
        trend = VolatilityTrend.STABLE
        if len(returns) >= 2 * window:
            prev_vol = returns.iloc[-2 * window : -window].std() * np.sqrt(window)
            if current_vol > prev_vol * 1.05:
                trend = VolatilityTrend.INCREASING
            elif current_vol < prev_vol * 0.95:
                trend = VolatilityTrend.DECREASING
        return VolatilityMetrics(
            current_volatility=float(current_vol),
            historical_volatility=float(hist_vol),
            volatility_percentile=float(percentile),
            volatility_trend=trend,
        )

    def calculate_trend_metrics(self, data: Union[pd.DataFrame, None]) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                'trend_direction': TrendDirection.SIDEWAYS,
                'trend_strength': 0.0,
            }
        window = self.config["trend_window"]
        closes_result = _safe_series(data.loc[:, "close"] if "close" in data.columns else pd.Series(dtype=float))
        if hasattr(closes_result, 'dtype') and str(closes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes = pd.to_numeric(closes_result, errors="coerce").fillna(0.0)
            else:
                closes = closes_result
        else:
            closes = closes_result
        if len(closes) < window:
            return {
                'trend_direction': TrendDirection.SIDEWAYS,
                'trend_strength': 0.0,
            }
        ma_short = closes.rolling(window=window // 2).mean()
        ma_long = closes.rolling(window=window).mean()
        
        # Безопасное получение последних значений
        ma_short_last = 0.0
        ma_long_last = 0.0
        if len(ma_short) > 0:
            ma_short_last = float(ma_short.iloc[-1])
        if len(ma_long) > 0:
            ma_long_last = float(ma_long.iloc[-1])
        
        trend_strength = float(
            np.tanh(
                abs(ma_short_last - ma_long_last) / (ma_long_last + 1e-8)
            )
        )
        if ma_short_last > ma_long_last:
            direction = TrendDirection.UP
        elif ma_short_last < ma_long_last:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.SIDEWAYS
        result = {
            'trend_direction': direction,
            'trend_strength': trend_strength,
        }
        
        # Безопасное вычисление продолжительности тренда
        trend_duration = 0
        if len(closes) > 0 and ma_short_last != ma_long_last:
            trend_sign = np.sign(ma_short_last - ma_long_last)
            diff_signs = closes.diff().apply(lambda x: np.sign(x) if x != 0 else 0)
            trend_duration = int((diff_signs == trend_sign).sum())
        
        if trend_duration:
            result["trend_duration"] = trend_duration
        
        # Безопасное получение поддержки и сопротивления
        support = closes.min()
        resistance = closes.max()
        if not np.isnan(support) and not np.isnan(resistance):
            result["support_resistance"] = {
                "support": float(support),
                "resistance": float(resistance),
            }
        return result

    def calculate_volume_metrics(
        self, data: Union[pd.DataFrame, None]
    ) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                'volume_ma': 0.0,
                'volume_ratio': 0.0,
                'volume_trend': VolumeTrend.STABLE,
                'volume_profile': {},
            }
        window = self.config["volume_window"]
        volumes_result: pd.Series = _safe_series(data.loc[:, "volume"] if "volume" in data.columns else pd.Series(dtype=float))
        if hasattr(volumes_result, 'dtype') and str(volumes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                volumes = pd.to_numeric(volumes_result, errors="coerce").fillna(0.0)
            else:
                volumes = volumes_result
        else:
            volumes = volumes_result
        volume_ma = (
            float(volumes.rolling(window=window).mean().iloc[-1])
            if len(volumes) >= window
            else 0.0
        )
        volume_ratio = (
            float(volumes.iloc[-1] / (volume_ma + 1e-8)) if volume_ma > 0 else 0.0
        )
        if len(volumes) >= 2 * window:
            prev_ma = float(volumes.rolling(window=window).mean().iloc[-window - 1])
            if volume_ma > prev_ma * 1.05:
                trend = VolumeTrend.INCREASING
            elif volume_ma < prev_ma * 0.95:
                trend = VolumeTrend.DECREASING
            else:
                trend = VolumeTrend.STABLE
        else:
            trend = VolumeTrend.STABLE
        return {
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'volume_trend': trend,
            'volume_profile': {},
        }

    def calculate_correlation_metrics(
        self, data1: Union[pd.DataFrame, None], data2: Union[pd.DataFrame, None]
    ) -> Dict[str, Any]:
        if (
            data1 is None
            or data2 is None
            or not isinstance(data1, pd.DataFrame)
            or not isinstance(data2, pd.DataFrame)
            or data1.empty
            or data2.empty
        ):
            return {
                'correlation': 0.0,
                'correlation_strength': 'weak',
                'correlation_trend': 'stable',
            }
        window = self.config["correlation_window"]
        closes1_result: pd.Series = _safe_series(data1.loc[:, "close"] if "close" in data1.columns else pd.Series(dtype=float))
        closes2_result: pd.Series = _safe_series(data2.loc[:, "close"] if "close" in data2.columns else pd.Series(dtype=float))
        if hasattr(closes1_result, 'dtype') and str(closes1_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes1 = pd.to_numeric(closes1_result, errors="coerce").fillna(0.0)
            else:
                closes1 = closes1_result
        else:
            closes1 = closes1_result
        if hasattr(closes2_result, 'dtype') and str(closes2_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes2 = pd.to_numeric(closes2_result, errors="coerce").fillna(0.0)
            else:
                closes2 = closes2_result
        else:
            closes2 = closes2_result
        if len(closes1) < window or len(closes2) < window:
            return {
                'correlation': 0.0,
                'correlation_strength': 'weak',
                'correlation_trend': 'stable',
            }
        # Рассчитываем корреляцию
        correlation = float(closes1.corr(closes2))
        if pd.isna(correlation):
            correlation = 0.0
        # Определяем силу корреляции
        strength = 'weak'  # значение по умолчанию
        if abs(correlation) > 0.7:
            strength = 'strong'
        elif abs(correlation) > 0.3 and abs(correlation) <= 0.7:
            strength = 'moderate'
        # Определяем тренд корреляции
        trend = 'stable'  # значение по умолчанию
        if len(closes1) >= 2 * window and len(closes2) >= 2 * window:
            recent_corr = float(closes1.iloc[-window:].corr(closes2.iloc[-window:]))
            prev_corr = float(closes1.iloc[-2 * window : -window].corr(closes2.iloc[-2 * window : -window]))
            if pd.isna(recent_corr):
                recent_corr = 0.0
            if pd.isna(prev_corr):
                prev_corr = 0.0
            if recent_corr > prev_corr * 1.1:
                trend = 'increasing'
            elif recent_corr < prev_corr * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            # Если недостаточно данных для расчета тренда, используем стабильный тренд
            trend = 'stable'
        return {
            'correlation': correlation,
            'correlation_strength': strength,
            'correlation_trend': trend,
        }

    def calculate_market_efficiency_metrics(
        self, data: Union[pd.DataFrame, None]
    ) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                'efficiency_ratio': 0.0,
                'hurst_exponent': 0.5,
                'market_efficiency': 'inefficient',
            }
        closes_result: pd.Series = _safe_series(data.loc[:, "close"] if "close" in data.columns else pd.Series(dtype=float))
        if hasattr(closes_result, 'dtype') and str(closes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes = pd.to_numeric(closes_result, errors="coerce").fillna(0.0)
            else:
                closes = closes_result
        else:
            closes = closes_result
        if len(closes) < 20:
            return {
                'efficiency_ratio': 0.0,
                'hurst_exponent': 0.5,
                'market_efficiency': 'inefficient',
            }
        # Рассчитываем эффективность рынка
        returns = closes.pct_change().dropna()
        if len(returns) < 10:
            return {
                'efficiency_ratio': 0.0,
                'hurst_exponent': 0.5,
                'market_efficiency': 'inefficient',
            }
        # Эффективность на основе волатильности
        volatility = returns.std()
        if volatility == 0:
            efficiency_ratio = 0.0
        else:
            efficiency_ratio = float(abs(returns.mean()) / volatility)
        # Экспонент Херста (упрощенная версия)
        if len(returns) >= 20:
            # Простая оценка на основе автокорреляции
            autocorr = returns.autocorr()
            if pd.isna(autocorr):
                autocorr = 0.0
            hurst_exponent = 0.5 + 0.5 * float(autocorr)
        else:
            hurst_exponent = 0.5
        # Определяем эффективность рынка
        market_efficiency = 'inefficient'  # значение по умолчанию
        if efficiency_ratio > 0.1:
            market_efficiency = 'efficient'
        elif efficiency_ratio > 0.05 and efficiency_ratio <= 0.1:
            market_efficiency = 'moderate'
        # else: market_efficiency остается 'inefficient'
        return {
            'efficiency_ratio': efficiency_ratio,
            'hurst_exponent': hurst_exponent,
            'market_efficiency': market_efficiency,
        }

    def calculate_liquidity_metrics(
        self, data: Union[pd.DataFrame, None], order_book: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                'bid_ask_spread': 0.0,
                'market_depth': 0.0,
                'liquidity_score': 0.0,
            }
        # Базовые метрики ликвидности
        bid_ask_spread = 0.0
        market_depth = 0.0
        liquidity_score = 0.0
        # Если есть данные стакана, рассчитываем более точные метрики
        if order_book and isinstance(order_book, dict):
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            if bids and asks:
                best_bid = float(bids[0][0]) if isinstance(bids[0], (list, tuple)) else 0.0
                best_ask = float(asks[0][0]) if isinstance(asks[0], (list, tuple)) else 0.0
                if best_bid > 0 and best_ask > 0:
                    bid_ask_spread = (best_ask - best_bid) / best_bid
                    # Глубина рынка
                    bid_volume = sum(float(bid[1]) for bid in bids[:5] if isinstance(bid, (list, tuple)))
                    ask_volume = sum(float(ask[1]) for ask in asks[:5] if isinstance(ask, (list, tuple)))
                    market_depth = bid_volume + ask_volume
                    # Оценка ликвидности
                    if bid_ask_spread < 0.001:
                        liquidity_score = 1.0
                    elif bid_ask_spread < 0.01:
                        liquidity_score = 0.7
                    elif bid_ask_spread < 0.05:
                        liquidity_score = 0.4
                    else:
                        liquidity_score = 0.1
        return {
            'bid_ask_spread': bid_ask_spread,
            'market_depth': market_depth,
            'liquidity_score': liquidity_score,
        }

    def calculate_momentum_metrics(
        self, data: Union[pd.DataFrame, None]
    ) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                'momentum_score': 0.0,
                'momentum_strength': 0.0,
                'momentum_trend': 'neutral',
            }
        closes_result: pd.Series = _safe_series(data.loc[:, "close"] if "close" in data.columns else pd.Series(dtype=float))
        if hasattr(closes_result, 'dtype') and str(closes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes = pd.to_numeric(closes_result, errors="coerce").fillna(0.0)
            else:
                closes = closes_result
        else:
            closes = closes_result
        if len(closes) < 14:
            return {
                'momentum_score': 0.0,
                'momentum_strength': 0.0,
                'momentum_trend': 'neutral',
            }
        # Рассчитываем RSI
        delta = closes.diff().astype(float)
        gain = (delta.where(delta > 0.0, 0.0)).rolling(window=14).mean()
        loss = (delta.where(delta < 0.0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        if pd.isna(current_rsi):
            current_rsi = 50.0
        # Нормализуем RSI к диапазону -1 до 1
        momentum_score = (current_rsi - 50) / 50
        # Сила импульса
        if len(closes) >= 20:
            recent_momentum = float(closes.iloc[-1] / closes.iloc[-20] - 1)
            momentum_strength = abs(recent_momentum)
        else:
            momentum_strength = 0.0
        # Тренд импульса
        momentum_trend = 'neutral'  # значение по умолчанию
        if momentum_score > 0.2:
            momentum_trend = 'positive'
        elif momentum_score < -0.2:
            momentum_trend = 'negative'
        else:
            momentum_trend = 'neutral'
        return {
            'momentum_score': momentum_score,
            'momentum_strength': momentum_strength,
            'momentum_trend': momentum_trend,
        }

    def calculate_market_stress_metrics(
        self, data: Union[pd.DataFrame, None]
    ) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {
                "stress_level": 0.0,
                "stress_trend": "stable",
                "volatility_regime": "normal",
            }
        closes_result: pd.Series = _safe_series(data.loc[:, "close"] if "close" in data.columns else pd.Series(dtype=float))
        volumes_result: pd.Series = _safe_series(data.loc[:, "volume"] if "volume" in data.columns else pd.Series(dtype=float))
        if hasattr(closes_result, 'dtype') and str(closes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                closes = pd.to_numeric(closes_result, errors="coerce").fillna(0.0)
            else:
                closes = closes_result
        else:
            closes = closes_result
        if hasattr(volumes_result, 'dtype') and str(volumes_result.dtype) == "object":
            if hasattr(pd, 'to_numeric'):
                volumes = pd.to_numeric(volumes_result, errors="coerce").fillna(0.0)
            else:
                volumes = volumes_result
        else:
            volumes = volumes_result
        
        if len(closes) < 20:
            return {
                "stress_level": 0.0,
                "stress_trend": "stable",
                "volatility_regime": "normal",
            }
        
        # Безопасное вычисление метрик стресса
        returns = closes.pct_change().dropna()
        if len(returns) < 2:
            return {
                "stress_level": 0.0,
                "stress_trend": "stable",
                "volatility_regime": "normal",
            }
        
        # Stress Level
        stress_level = 0.0
        if len(returns) >= 20:
            recent_volatility = returns.iloc[-20:].std()
            historical_volatility = returns.std()
            if historical_volatility > 0:
                stress_level = float(recent_volatility / historical_volatility)
        
        # Stress Trend
        stress_trend = "stable"
        if len(returns) >= 40:
            recent_stress = returns.iloc[-20:].std()
            prev_stress = returns.iloc[-40:-20].std()
            if recent_stress > prev_stress * 1.2:
                stress_trend = "increasing"
            elif recent_stress < prev_stress * 0.8:
                stress_trend = "decreasing"
        
        # Volatility Regime
        volatility_regime = "normal"
        if stress_level > 2.0:
            volatility_regime = "high"
        elif stress_level < 0.5:
            volatility_regime = "low"
        
        return {
            "stress_level": stress_level,
            "stress_trend": stress_trend,
            "volatility_regime": volatility_regime,
        }

    def get_comprehensive_metrics(
        self, data: Union[pd.DataFrame, None], order_book: Optional[Dict[str, Any]]
    ) -> MarketMetricsResult:
        """Получить комплексные метрики рынка."""
        with self._lock:
            volatility_metrics = self.calculate_volatility_metrics(data)
            trend_metrics = self.calculate_trend_metrics(data)
            volume_metrics = self.calculate_volume_metrics(data)
            liquidity_metrics = self.calculate_liquidity_metrics(data, order_book)
            momentum_metrics = self.calculate_momentum_metrics(data)
            stress_metrics = self.calculate_market_stress_metrics(data)
            efficiency_metrics = self.calculate_market_efficiency_metrics(data)
            
            # Создаем объекты метрик с правильными полями
            trend_metrics_obj = TrendMetrics(
                trend_direction=trend_metrics.get('trend_direction', TrendDirection.SIDEWAYS),
                trend_strength=trend_metrics.get('trend_strength', 0.0),
                trend_confidence=trend_metrics.get('trend_confidence', 0.0),
                support_level=trend_metrics.get('support_level'),
                resistance_level=trend_metrics.get('resistance_level')
            )
            
            volume_metrics_obj = VolumeMetrics(
                current_volume=volume_metrics.get('current_volume', 0.0),
                average_volume=volume_metrics.get('average_volume', 0.0),
                volume_trend=volume_metrics.get('volume_trend', VolumeTrend.STABLE),
                volume_ratio=volume_metrics.get('volume_ratio', 0.0),
                unusual_volume=volume_metrics.get('unusual_volume', False)
            )
            
            liquidity_metrics_obj = LiquidityMetrics(
                bid_ask_spread=liquidity_metrics.get('bid_ask_spread', 0.0),
                market_depth=liquidity_metrics.get('market_depth', 0.0),
                order_book_imbalance=liquidity_metrics.get('order_book_imbalance', 0.0),
                liquidity_score=liquidity_metrics.get('liquidity_score', 0.0)
            )
            
            momentum_metrics_obj = MomentumMetrics(
                rsi=momentum_metrics.get('momentum_score', 0.0),
                macd=momentum_metrics.get('momentum_score', 0.0),
                macd_signal=momentum_metrics.get('momentum_score', 0.0),
                macd_histogram=momentum_metrics.get('momentum_score', 0.0),
                momentum_score=momentum_metrics.get('momentum_strength', 0.0)
            )
            
            stress_metrics_obj = MarketStressMetrics(
                stress_index=stress_metrics.get('stress_level', 0.0),
                fear_greed_index=stress_metrics.get('stress_level', 0.0),
                market_regime=stress_metrics.get('volatility_regime', 'normal'),
                stress_level=stress_metrics.get('stress_trend', 'stable')
            )
            
            return MarketMetricsResult(
                volatility=volatility_metrics,
                trend=trend_metrics_obj,
                volume=volume_metrics_obj,
                liquidity=liquidity_metrics_obj,
                momentum=momentum_metrics_obj,
                stress=stress_metrics_obj,
                timestamp=datetime.now().isoformat(),
            )


# Экспорт класса для обратной совместимости
MarketMetrics = MarketMetricsService
IMarketMetrics = MarketMetricsService
