import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger

from domain.types.strategy_types import (
    MarketRegime,
    Signal,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.core.technical_analysis import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
)

from infrastructure.strategies.base_strategy import BaseStrategy, Signal
from domain.types.strategy_types import StrategyDirection, MarketRegime


class SidewaysStrategy(BaseStrategy):
    """Базовый класс для стратегий флэта"""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
        obv_threshold: float = 1.5,
    ):
        """
        Инициализация стратегии.
        Args:
            bb_period: Период Bollinger Bands
            bb_std: Стандартное отклонение для BB
            rsi_period: Период RSI
            stoch_k: Период %K для Stochastic
            stoch_d: Период %D для Stochastic
            obv_threshold: Порог для OBV
        """
        super().__init__()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.obv_threshold = obv_threshold
        # Добавляем технический анализ
        self.technical_analysis = None  # Будет инициализирован при необходимости

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Валидация входных данных.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Tuple[bool, Optional[str]]: (валидность, сообщение об ошибке)
        """
        try:
            if data is None or data.empty:
                return False, "Data is None or empty"
            
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            
            min_periods = max(self.bb_period, self.rsi_period, self.stoch_k, 50)
            if len(data) < min_periods:
                return False, f"Insufficient data: {len(data)} < {min_periods}"
            
            # Проверка на NaN значения
            for col in required_columns:
                if data[col].isna().any():
                    return False, f"NaN values found in column: {col}"
            
            # Проверка на отрицательные цены
            for col in ["open", "high", "low", "close"]:
                if (data[col] <= 0).any():
                    return False, f"Non-positive values found in column: {col}"
            
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных для определения бокового движения.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            # Валидация данных
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error_msg}")
            # Расчет индикаторов
            df = self._calculate_indicators(data.copy())
            # Определение режима рынка
            market_regime = self._detect_market_regime(df)
            # Анализ волатильности
            volatility = df["close"].pct_change().rolling(20).std().iloc[-1]
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0 
            # Анализ диапазона
            bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            avg_bb_width = bb_width.rolling(20).mean().iloc[-1]
            avg_bb_width = float(avg_bb_width) if avg_bb_width is not None and not pd.isna(avg_bb_width) else 0.0 
            # Анализ RSI
            rsi = df["rsi"].iloc[-1]
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
            rsi_trend = "neutral"
            if rsi < 30:
                rsi_trend = "oversold"
            elif rsi > 70:
                rsi_trend = "overbought"
            # Анализ Stochastic
            stoch_k = df["stoch_k"].iloc[-1]
            stoch_k = float(stoch_k) if stoch_k is not None and not pd.isna(stoch_k) else 50.0 
            stoch_d = df["stoch_d"].iloc[-1]
            stoch_d = float(stoch_d) if stoch_d is not None and not pd.isna(stoch_d) else 50.0 
            # Анализ объема
            volume_analysis = self._analyze_volume(df)
            # Генерация сигналов
            signals = self._generate_signals_from_analysis(df)
            # Расчет метрик
            metrics = {
                "volatility": volatility,
                "bb_width": avg_bb_width,
                "rsi_trend": rsi_trend,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "volume_ratio": volume_analysis["volume_ratio"],
            }
            # Оценка риска
            risk_assessment = {
                "volatility_risk": min(1.0, volatility * 10),
                "range_risk": min(1.0, avg_bb_width * 2),
                "momentum_risk": abs(rsi - 50) / 50,
                "volume_risk": 1.0 - volume_analysis["volume_ratio"],
            }
            # Рекомендации
            recommendations = self._generate_recommendations(df, market_regime)
            return {
                "strategy_id": f"sideways_{id(self)}",
                "timestamp": datetime.now(),
                "market_data": df,
                "indicators": {
                    "bb_upper": df["bb_upper"],
                    "bb_lower": df["bb_lower"],
                    "rsi": df["rsi"],
                    "stoch_k": df["stoch_k"],
                    "stoch_d": df["stoch_d"],
                    "obv": df["obv"],
                },
                "signals": signals,
                "metrics": metrics,
                "market_regime": market_regime,
                "confidence": self._calculate_confidence(df),
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error(f"Error in sideways analysis: {str(e)}")
            return {}

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала на основе боковой стратегии.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[Signal]: Торговый сигнал или None
        """
        try:
            df = self._calculate_indicators(data.copy())
            if df.shape[0] < max(self.bb_period, self.rsi_period):
                return None
            close = df["close"].iloc[-1]
            close = float(close) if close is not None and not pd.isna(close) else 0.0 
            if close <= 0:
                return None
                
            bb_upper = df["bb_upper"].iloc[-1]
            bb_upper = float(bb_upper) if bb_upper is not None and not pd.isna(bb_upper) else 0.0 
            bb_lower = df["bb_lower"].iloc[-1]
            bb_lower = float(bb_lower) if bb_lower is not None and not pd.isna(bb_lower) else 0.0 
            rsi = df["rsi"].iloc[-1]
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
            stoch_k = df["stoch_k"].iloc[-1]
            stoch_k = float(stoch_k) if stoch_k is not None and not pd.isna(stoch_k) else 50.0 
            stoch_d = df["stoch_d"].iloc[-1]
            stoch_d = float(stoch_d) if stoch_d is not None and not pd.isna(stoch_d) else 50.0 
            # Проверка объема
            volume_ok = self._check_volume(df)
            # Сигнал на покупку (отскок от нижней полосы Боллинджера)
            if close <= bb_lower * 1.01 and rsi < 40 and stoch_k < 20 and volume_ok:
                entry_price = close
                stop_loss = self._calculate_stop_loss(df, entry_price, "long")
                take_profit = self._calculate_take_profit(
                    df, entry_price, stop_loss, "long"
                )
                confidence = min(1.0, (40 - rsi) / 40 * 0.8 + 0.2)
                return Signal(
                    direction="long",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                )
            # Сигнал на продажу (отскок от верхней полосы Боллинджера)
            elif close >= bb_upper * 0.99 and rsi > 60 and stoch_k > 80 and volume_ok:
                entry_price = close
                stop_loss = self._calculate_stop_loss(df, entry_price, "short")
                take_profit = self._calculate_take_profit(
                    df, entry_price, stop_loss, "short"
                )
                confidence = min(1.0, (rsi - 60) / 40 * 0.8 + 0.2)
                return Signal(
                    direction="short",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                )
            return None
        except Exception as e:
            logger.error(f"Error generating sideways signal: {str(e)}")
            return None

    def _detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Определение рыночного режима"""
        try:
            # Анализ тренда
            ema_20 = df["close"].ewm(span=20).mean()
            ema_50 = df["close"].ewm(span=50).mean()
            # Если EMA близки друг к другу - боковик
            ema_20_val = float(ema_20.iloc[-1]) if ema_20.iloc[-1] is not None and not pd.isna(ema_20.iloc[-1]) else 0.0 
            ema_50_val = float(ema_50.iloc[-1]) if ema_50.iloc[-1] is not None and not pd.isna(ema_50.iloc[-1]) else 0.0 
            if ema_50_val != 0 and abs(ema_20_val - ema_50_val) / ema_50_val < 0.02:
                return MarketRegime.SIDEWAYS
            # Анализ волатильности
            volatility = df["close"].pct_change().rolling(20).std().iloc[-1]
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0 
            if volatility > 0.03:  # Высокая волатильность
                return MarketRegime.VOLATILE
            return MarketRegime.SIDEWAYS
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, float]:
        """Анализ объема"""
        try:
            volume_ma = df["volume"].rolling(20).mean()
            current_volume = df["volume"].iloc[-1]
            current_volume = float(current_volume) if current_volume is not None and not pd.isna(current_volume) else 0.0 
            volume_ma_val = volume_ma.iloc[-1]
            volume_ma_val = float(volume_ma_val) if volume_ma_val is not None and not pd.isna(volume_ma_val) else 1.0 
            volume_ratio = current_volume / volume_ma_val if volume_ma_val != 0 else 1.0
            return {
                "volume_ratio": volume_ratio,
                "volume_trend_score": (
                    1.2 if volume_ratio > 1.2 else 0.8 if volume_ratio < 0.8 else 1.0
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return {"volume_ratio": 1.0, "volume_trend_score": 1.0}

    def _generate_signals_from_analysis(self, df: pd.DataFrame) -> List[Signal]:
        """Генерация сигналов на основе анализа"""
        signals = []
        try:
            # Проверяем условия для сигналов
            close = df["close"].iloc[-1]
            close = float(close) if close is not None and not pd.isna(close) else 0.0 
            bb_upper = df["bb_upper"].iloc[-1]
            bb_upper = float(bb_upper) if bb_upper is not None and not pd.isna(bb_upper) else 0.0 
            bb_lower = df["bb_lower"].iloc[-1]
            bb_lower = float(bb_lower) if bb_lower is not None and not pd.isna(bb_lower) else 0.0 
            rsi = df["rsi"].iloc[-1]
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
            
            # Сигнал на покупку
            if close <= bb_lower * 1.01 and rsi < 40:
                signal = Signal(
                    direction=StrategyDirection.LONG,
                    entry_price=close,
                    confidence=0.7,
                    strategy_type=StrategyType.SIDEWAYS,
                    market_regime=MarketRegime.SIDEWAYS,
                )
                signals.append(signal)
            # Сигнал на продажу
            elif close >= bb_upper * 0.99 and rsi > 60:
                signal = Signal(
                    direction=StrategyDirection.SHORT,
                    entry_price=close,
                    confidence=0.7,
                    strategy_type=StrategyType.SIDEWAYS,
                    market_regime=MarketRegime.SIDEWAYS,
                )
                signals.append(signal)
        except Exception as e:
            logger.error(f"Error generating signals from analysis: {str(e)}")
        return signals

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Расчет уверенности в анализе"""
        try:
            # Факторы уверенности
            bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            avg_bb_width = bb_width.rolling(20).mean().iloc[-1]
            avg_bb_width = float(avg_bb_width) if avg_bb_width is not None and not pd.isna(avg_bb_width) else 0.0 
            
            rsi = df["rsi"].iloc[-1]
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
            rsi_confidence = 1.0 - abs(rsi - 50) / 50
            
            current_volume = df["volume"].iloc[-1]
            current_volume = float(current_volume) if current_volume is not None and not pd.isna(current_volume) else 0.0 
            volume_ma = df["volume"].rolling(20).mean().iloc[-1]
            volume_ma = float(volume_ma) if volume_ma is not None and not pd.isna(volume_ma) else 1.0 
            volume_ratio = current_volume / volume_ma if volume_ma != 0 else 1.0
            volume_confidence = min(1.0, volume_ratio)
            
            # Средневзвешенная уверенность
            confidence = (
                (1.0 - avg_bb_width) * 0.4
                + rsi_confidence * 0.3
                + volume_confidence * 0.3
            )
            return max(0.1, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _generate_recommendations(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        try:
            rsi = df["rsi"].iloc[-1]
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
            
            bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            avg_bb_width = bb_width.rolling(20).mean().iloc[-1]
            avg_bb_width = float(avg_bb_width) if avg_bb_width is not None and not pd.isna(avg_bb_width) else 0.0 
            
            if rsi < 30:
                recommendations.append(
                    "RSI показывает перепроданность - возможен отскок"
                )
            elif rsi > 70:
                recommendations.append(
                    "RSI показывает перекупленность - возможен откат"
                )
            if avg_bb_width < 0.05:
                recommendations.append(
                    "Узкие полосы Боллинджера указывают на низкую волатильность"
                )
            elif avg_bb_width > 0.15:
                recommendations.append(
                    "Широкие полосы Боллинджера указывают на высокую волатильность"
                )
            if regime == MarketRegime.SIDEWAYS:
                recommendations.append(
                    "Рынок находится в боковом движении - используйте стратегии range trading"
                )
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        return recommendations

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
        try:
            df = data.copy()
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                df["close"], self.bb_period, self.bb_std
            )
            df["bb_upper"] = bb_upper
            df["bb_middle"] = bb_middle
            df["bb_lower"] = bb_lower
            # RSI
            df["rsi"] = calculate_rsi(df["close"], self.rsi_period)
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(
                df["high"], df["low"], df["close"], self.stoch_k, self.stoch_d
            )
            df["stoch_k"] = stoch_k
            df["stoch_d"] = stoch_d
            # OBV
            df["obv"] = self._calculate_obv(df["close"], df["volume"])
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return data

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """Расчет Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Error calculating stochastic: {str(e)}")
            return pd.Series([50.0] * len(close)), pd.Series([50.0] * len(close))

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Расчет On-Balance Volume"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0] 
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]: 
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i] 
                elif close.iloc[i] < close.iloc[i - 1]: 
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i] 
                else:
                    obv.iloc[i] = obv.iloc[i - 1] 
            return obv
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series([0.0] * len(close))

    def _check_volume(self, data: pd.DataFrame) -> bool:
        """Проверка объема"""
        try:
            current_volume = data["volume"].iloc[-1]
            current_volume = float(current_volume) if current_volume is not None and not pd.isna(current_volume) else 0.0 
            volume_ma = data["volume"].rolling(20).mean().iloc[-1]
            volume_ma = float(volume_ma) if volume_ma is not None and not pd.isna(volume_ma) else 1.0 
            return current_volume > volume_ma * self.obv_threshold
        except Exception as e:
            logger.error(f"Error checking volume: {str(e)}")
            return False

    def _calculate_stop_loss(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> float:
        """Расчет стоп-лосса"""
        try:
            if position_type == "long":
                # Для длинной позиции - стоп под нижней полосой Боллинджера
                bb_lower = data["bb_lower"].iloc[-1]
                bb_lower = float(bb_lower) if bb_lower is not None and not pd.isna(bb_lower) else entry_price * 0.95 
                return min(bb_lower, entry_price * 0.95)
            else:
                # Для короткой позиции - стоп над верхней полосой Боллинджера
                bb_upper = data["bb_upper"].iloc[-1]
                bb_upper = float(bb_upper) if bb_upper is not None and not pd.isna(bb_upper) else entry_price * 1.05 
                return max(bb_upper, entry_price * 1.05)
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            if position_type == "long":
                return entry_price * 0.95
            else:
                return entry_price * 1.05

    def _calculate_take_profit(
        self,
        data: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
        position_type: str,
    ) -> float:
        """Расчет тейк-профита"""
        try:
            if position_type == "long":
                # Для длинной позиции - тейк на уровне средней полосы Боллинджера
                bb_middle = data["bb_middle"].iloc[-1]
                bb_middle = float(bb_middle) if bb_middle is not None and not pd.isna(bb_middle) else entry_price * 1.05 
                # Минимальный риск-риворд 1:2
                min_take_profit = entry_price + 2 * abs(entry_price - stop_loss)
                return max(bb_middle, min_take_profit)
            else:
                # Для короткой позиции - тейк на уровне средней полосы Боллинджера
                bb_middle = data["bb_middle"].iloc[-1]
                bb_middle = float(bb_middle) if bb_middle is not None and not pd.isna(bb_middle) else entry_price * 0.95 
                # Минимальный риск-риворд 1:2
                min_take_profit = entry_price - 2 * abs(stop_loss - entry_price)
                return min(bb_middle, min_take_profit)
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            if position_type == "long":
                return entry_price * 1.05
            else:
                return entry_price * 0.95


def sideways_strategy_bb_rsi(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Стратегия на основе Bollinger Bands и RSI.
    Args:
        data: Рыночные данные
    Returns:
        Optional[Dict[str, Any]]: Сигнал на вход
    """
    try:
        strategy = SidewaysStrategy()
        df = strategy._calculate_indicators(data.copy())
        if df.shape[0] < 20:
            return None
        close = df["close"].iloc[-1]
        close = float(close) if close is not None and not pd.isna(close) else 0.0 
        bb_upper = df["bb_upper"].iloc[-1]
        bb_upper = float(bb_upper) if bb_upper is not None and not pd.isna(bb_upper) else 0.0 
        bb_lower = df["bb_lower"].iloc[-1]
        bb_lower = float(bb_lower) if bb_lower is not None and not pd.isna(bb_lower) else 0.0 
        rsi = df["rsi"].iloc[-1]
        rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else 50.0 
        # Сигнал на покупку
        if close <= bb_lower * 1.01 and rsi < 40:
            stop_loss = strategy._calculate_stop_loss(df, close, "long")
            take_profit = strategy._calculate_take_profit(df, close, stop_loss, "long")
            return {
                "side": "buy",
                "entry_price": close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "amount": 1.0,
            }
        # Сигнал на продажу
        elif close >= bb_upper * 0.99 and rsi > 60:
            stop_loss = strategy._calculate_stop_loss(df, close, "short")
            take_profit = strategy._calculate_take_profit(df, close, stop_loss, "short")
            return {
                "side": "sell",
                "entry_price": close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "amount": 1.0,
            }
        return None
    except Exception as e:
        logger.error(f"Error in BB-RSI strategy: {str(e)}")
        return None


def sideways_strategy_stoch_obv(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Стратегия на основе Stochastic и OBV.
    Args:
        data: Рыночные данные
    Returns:
        Optional[Dict[str, Any]]: Сигнал на вход
    """
    try:
        strategy = SidewaysStrategy()
        df = strategy._calculate_indicators(data.copy())
        if df.shape[0] < 20:
            return None
        close = df["close"].iloc[-1]
        close = float(close) if close is not None and not pd.isna(close) else 0.0 
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_k = float(stoch_k) if stoch_k is not None and not pd.isna(stoch_k) else 50.0 
        stoch_d = df["stoch_d"].iloc[-1]
        stoch_d = float(stoch_d) if stoch_d is not None and not pd.isna(stoch_d) else 50.0 
        # Сигнал на покупку
        if stoch_k < 20 and stoch_d < 20:
            stop_loss = strategy._calculate_stop_loss(df, close, "long")
            take_profit = strategy._calculate_take_profit(df, close, stop_loss, "long")
            return {
                "side": "buy",
                "entry_price": close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "amount": 1.0,
            }
        # Сигнал на продажу
        elif stoch_k > 80 and stoch_d > 80:
            stop_loss = strategy._calculate_stop_loss(df, close, "short")
            take_profit = strategy._calculate_take_profit(df, close, stop_loss, "short")
            return {
                "side": "sell",
                "entry_price": close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "amount": 1.0,
            }
        return None
    except Exception as e:
        logger.error(f"Error in Stochastic-OBV strategy: {str(e)}")
        return None
