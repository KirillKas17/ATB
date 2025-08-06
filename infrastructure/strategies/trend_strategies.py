from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from shared.decimal_utils import TradingDecimal, to_trading_decimal

from domain.services.technical_analysis import DefaultTechnicalAnalysisService
from domain.type_definitions.strategy_types import (
    MarketRegime,
    Signal,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.core.technical_analysis import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)

from .base_strategy import BaseStrategy


@dataclass
class TrendConfig:
    """Конфигурация трендовой стратегии"""

    # Параметры EMA
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    # Параметры индикаторов
    adx_threshold: float = 25.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Параметры управления рисками
    risk_reward: float = 2.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.2
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class TrendStrategy(BaseStrategy):
    """Базовый класс для трендовых стратегий"""

    def __init__(
        self, config: Optional[Union[Dict[str, Any], TrendConfig]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект TrendConfig
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, TrendConfig):
            config_dict = {
                "ema_fast": config.ema_fast,
                "ema_medium": config.ema_medium,
                "ema_slow": config.ema_slow,
                "adx_threshold": config.adx_threshold,
                "atr_period": config.atr_period,
                "macd_fast": config.macd_fast,
                "macd_slow": config.macd_slow,
                "macd_signal": config.macd_signal,
                "risk_reward": config.risk_reward,
                "risk_per_trade": config.risk_per_trade,
                "max_position_size": config.max_position_size,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию как словарь для базового класса
        if isinstance(config, TrendConfig):
            self.config = config_dict
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = {}
            
        # Сохраняем TrendConfig отдельно для доступа к типизированным атрибутам
        if isinstance(config, TrendConfig):
            self._trend_config = config
        elif isinstance(config, dict):
            self._trend_config = TrendConfig(**config)
        else:
            self._trend_config = TrendConfig()
            
        # Инициализация технического анализа
        self.technical_analysis = DefaultTechnicalAnalysisService()

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов"""
        # EMA
        data.loc[:, "ema_fast"] = calculate_ema(data["close"], self._trend_config.ema_fast)
        data.loc[:, "ema_medium"] = calculate_ema(data["close"], self._trend_config.ema_medium)
        data.loc[:, "ema_slow"] = calculate_ema(data["close"], self._trend_config.ema_slow)
        # MACD
        macd_result = calculate_macd(data["close"])
        if isinstance(macd_result, dict):
            macd_series = macd_result.get("macd", pd.Series())
            signal_series = macd_result.get("signal", pd.Series())
            histogram_series = macd_result.get("histogram", pd.Series())
            data.loc[:, "macd"] = macd_series
            data.loc[:, "macd_signal"] = signal_series
            data.loc[:, "macd_hist"] = histogram_series
        # ADX
        data.loc[:, "adx"] = self._calculate_adx(data)
        # ATR
        data.loc[:, "atr"] = calculate_atr(
            data["high"], data["low"], data["close"], self._trend_config.atr_period
        )
        return data

    def _calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Расчет ADX."""
        try:
            return calculate_adx(data["high"], data["low"], data["close"])
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD"""
        exp1 = prices.ewm(span=self._trend_config.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self._trend_config.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self._trend_config.macd_signal, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _check_trend_strength(self, data: pd.DataFrame) -> bool:
        """Правильная проверка силы тренда с множественными критериями"""
        try:
            if len(data) < 50:  # Недостаточно данных
                return False
                
            # 1. Проверяем ADX (сила тренда)
            adx_series = self._calculate_adx(data)
            if len(adx_series) == 0:
                return False
            adx_current = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
            
            # ADX > 25 указывает на сильный тренд
            if adx_current < self._trend_config.adx_threshold:
                return False
            
            # 2. Проверяем согласованность EMA
            ema_fast = calculate_ema(data["close"], self._trend_config.ema_fast)
            ema_medium = calculate_ema(data["close"], self._trend_config.ema_medium)
            ema_slow = calculate_ema(data["close"], self._trend_config.ema_slow)
            
            if len(ema_fast) == 0 or len(ema_medium) == 0 or len(ema_slow) == 0:
                return False
                
            ema_fast_val = float(ema_fast.iloc[-1])
            ema_medium_val = float(ema_medium.iloc[-1])
            ema_slow_val = float(ema_slow.iloc[-1])
            
            # Проверяем правильное расположение EMA для тренда
            uptrend = (ema_fast_val > ema_medium_val > ema_slow_val)
            downtrend = (ema_fast_val < ema_medium_val < ema_slow_val)
            
            if not (uptrend or downtrend):
                return False
            
            # 3. Проверяем наклон EMA (тренд должен быть устойчивым)
            if len(ema_medium) >= 5:
                ema_slope = (ema_medium.iloc[-1] - ema_medium.iloc[-5]) / 5
                min_slope = abs(ema_medium_val) * 0.001  # Минимальный наклон 0.1%
                if abs(ema_slope) < min_slope:
                    return False
            
            # 4. Проверяем последовательность движений цены
            closes = data["close"].tail(5)
            if len(closes) >= 5:
                if uptrend:
                    # Для восходящего тренда большинство движений должно быть вверх
                    up_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] > closes.iloc[i-1])
                    if up_moves < 3:  # Минимум 3 из 4 движений
                        return False
                elif downtrend:
                    # Для нисходящего тренда большинство движений должно быть вниз
                    down_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] < closes.iloc[i-1])
                    if down_moves < 3:  # Минимум 3 из 4 движений
                        return False
            
            # 5. Проверяем объем (должен подтверждать тренд)
            if "volume" in data.columns and len(data) >= 20:
                volume_ma = data["volume"].rolling(20).mean()
                current_volume = data["volume"].iloc[-1]
                avg_volume = volume_ma.iloc[-1]
                
                # Объем должен быть выше среднего для подтверждения тренда
                if current_volume < avg_volume * 0.8:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trend strength: {str(e)}")
            return False

    def _calculate_stop_loss(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> float:
        """Расширенный расчет уровня стоп-лосса.
        Args:
            data: DataFrame с рыночными данными
            entry_price: Цена входа
            position_type: Тип позиции ('long' или 'short')
        Returns:
            float: Уровень стоп-лосса
        """
        try:
            # Расчет ATR
            atr_series = calculate_atr(
                data["high"], data["low"], data["close"], self._trend_config.atr_period
            )
            atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            # Расчет волатильности
            volatility = float(data["close"].pct_change().rolling(20).std().iloc[-1])
            # Расчет структуры рынка
            market_structure = self.technical_analysis.calculate_market_structure(data)
            # Исправление: безопасное извлечение данных из market_structure
            support_resistance = []
            if hasattr(market_structure, '__iter__') and not isinstance(market_structure, (str, bytes)):
                support_resistance = [
                    float(level) if hasattr(level, '__float__') else 0.0
                    for level in market_structure
                ]
            liquidity_zones: List[float] = []
            # Расчет импульса
            rsi_series = calculate_rsi(data["close"], 14)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            macd_result = calculate_macd(data["close"])
            if isinstance(macd_result, dict):
                macd = macd_result.get("macd", pd.Series())
                signal = macd_result.get("signal", pd.Series())
                hist = macd_result.get("histogram", pd.Series())
            else:
                macd = pd.Series()
                signal = pd.Series()
                hist = pd.Series()
            # Расчет тренда
            ema_20 = calculate_ema(data["close"], 20)
            ema_50 = calculate_ema(data["close"], 50)
            trend_strength = float(
                abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            )
            # Расчет объема
            volume_ma = float(data["volume"].rolling(20).mean().iloc[-1])
            volume_ratio = float(data["volume"].iloc[-1] / volume_ma)
            # Базовый стоп на основе ATR с безопасными границами - используем Decimal для точности
            entry_decimal = to_trading_decimal(entry_price)
            atr_decimal = to_trading_decimal(atr)
            min_percent_stop = TradingDecimal.calculate_percentage(entry_decimal, to_trading_decimal(0.5))
            base_stop_decimal = max(atr_decimal * to_trading_decimal(1.5), min_percent_stop)
            base_stop = float(base_stop_decimal)  # Конвертируем обратно для совместимости
            
            # Безопасные корректировки с ограничениями
            volatility_multiplier = max(0.5, min(2.0, 1 + volatility * 5))  # 0.5x - 2.0x
            trend_multiplier = max(0.8, min(1.5, 1 + trend_strength * 2))    # 0.8x - 1.5x  
            volume_multiplier = max(0.8, min(1.3, 1 + (volume_ratio - 1) * 0.3))  # 0.8x - 1.3x
            momentum_multiplier = max(0.9, min(1.2, 1 + (abs(rsi - 50) / 50) * 0.2))  # 0.9x - 1.2x
            
            # Расчет финального множителя с ограничениями
            final_multiplier = float(
                volatility_multiplier
                * trend_multiplier  
                * volume_multiplier
                * momentum_multiplier
            )
            # Ограничиваем общий множитель
            final_multiplier = max(0.5, min(3.0, final_multiplier))  # 0.5x - 3.0x от базового стопа
            
            # Расчет стопа с абсолютными границами
            stop_distance = base_stop * final_multiplier
            
            # Абсолютные границы стоп-лосса - используем Decimal для точности
            min_stop_percent = to_trading_decimal(0.3)  # Минимум 0.3% от цены входа
            max_stop_percent = to_trading_decimal(5.0)  # Максимум 5% от цены входа
            
            min_stop_distance_decimal = TradingDecimal.calculate_percentage(entry_decimal, min_stop_percent)
            max_stop_distance_decimal = TradingDecimal.calculate_percentage(entry_decimal, max_stop_percent)
            min_stop_distance = float(min_stop_distance_decimal)
            max_stop_distance = float(max_stop_distance_decimal)
            
            stop_distance = max(min_stop_distance, min(stop_distance, max_stop_distance))
            # Поиск ближайшего уровня поддержки/сопротивления
            if position_type == "long":
                # Для длинной позиции ищем ближайший уровень поддержки
                support_levels = [
                    level for level in support_resistance if level < entry_price
                ]
                if support_levels:
                    nearest_support = max(support_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_support)
                # Проверка ликвидности
                liquidity_levels = [
                    level for level in liquidity_zones if level < entry_price
                ]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_liquidity)
                calculated_stop = float(entry_price - stop_distance)
                            # КРИТИЧЕСКАЯ ПРОВЕРКА: стоп-лосс для long должен быть меньше цены входа
            if calculated_stop >= entry_price:
                # Безопасный fallback с Decimal точностью
                calculated_stop = float(TradingDecimal.calculate_stop_loss(
                    entry_decimal, "long", to_trading_decimal(1.0)  # 1% стоп
                ))
                return calculated_stop
            else:
                # Для короткой позиции ищем ближайший уровень сопротивления
                resistance_levels = [
                    level for level in support_resistance if level > entry_price
                ]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels)
                    stop_distance = min(stop_distance, nearest_resistance - entry_price)
                # Проверка ликвидности
                liquidity_levels = [
                    level for level in liquidity_zones if level > entry_price
                ]
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels)
                    stop_distance = min(stop_distance, nearest_liquidity - entry_price)
                calculated_stop = float(entry_price + stop_distance)
                # КРИТИЧЕСКАЯ ПРОВЕРКА: стоп-лосс для short должен быть больше цены входа
                if calculated_stop <= entry_price:
                    # Безопасный fallback с Decimal точностью
                    calculated_stop = float(TradingDecimal.calculate_stop_loss(
                        entry_decimal, "short", to_trading_decimal(1.0)  # 1% стоп
                    ))
                return calculated_stop
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            # Безопасный fallback с Decimal точностью
            if position_type == "long":
                return float(TradingDecimal.calculate_stop_loss(
                    to_trading_decimal(entry_price), "long", to_trading_decimal(1.0)
                ))
            else:
                return float(TradingDecimal.calculate_stop_loss(
                    to_trading_decimal(entry_price), "short", to_trading_decimal(1.0)
                ))

    def _validate_signal(self, signal) -> bool:
        """Критическая валидация торгового сигнала"""
        try:
            # Проверка цены входа
            if not hasattr(signal, 'entry_price') or signal.entry_price <= 0:
                logger.warning("Invalid entry_price in signal")
                return False
                
            # Проверка стоп-лосса
            if not hasattr(signal, 'stop_loss') or signal.stop_loss <= 0:
                logger.warning("Invalid stop_loss in signal")
                return False
                
            # Проверка тейк-профита
            if not hasattr(signal, 'take_profit') or signal.take_profit <= 0:
                logger.warning("Invalid take_profit in signal")
                return False
                
            # Проверка направления сигнала
            if not hasattr(signal, 'direction') or signal.direction not in ["long", "short"]:
                logger.warning("Invalid direction in signal")
                return False
                
            # Критическая проверка логики стоп-лосса
            if signal.direction == "long":
                if signal.stop_loss >= signal.entry_price:
                    logger.warning(f"Long signal: stop_loss ({signal.stop_loss}) >= entry_price ({signal.entry_price})")
                    return False
                if signal.take_profit <= signal.entry_price:
                    logger.warning(f"Long signal: take_profit ({signal.take_profit}) <= entry_price ({signal.entry_price})")
                    return False
            elif signal.direction == "short":
                if signal.stop_loss <= signal.entry_price:
                    logger.warning(f"Short signal: stop_loss ({signal.stop_loss}) <= entry_price ({signal.entry_price})")
                    return False
                if signal.take_profit >= signal.entry_price:
                    logger.warning(f"Short signal: take_profit ({signal.take_profit}) >= entry_price ({signal.entry_price})")
                    return False
                    
            # Проверка разумности расстояний
            entry_price = float(signal.entry_price)
            stop_distance = abs(float(signal.stop_loss) - entry_price)
            profit_distance = abs(float(signal.take_profit) - entry_price)
            
            # Стоп-лосс не должен быть больше 10% от цены
            if stop_distance / entry_price > 0.1:
                logger.warning(f"Stop loss too wide: {stop_distance/entry_price:.2%}")
                return False
                
            # Тейк-профит не должен быть меньше стоп-лосса (плохое R/R)
            if profit_distance < stop_distance * 0.5:
                logger.warning(f"Poor risk/reward ratio: profit={profit_distance:.4f}, stop={stop_distance:.4f}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return False

    def _calculate_take_profit(
        self,
        data: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
        position_type: str,
    ) -> float:
        """Расчет тейк-профита"""
        try:
            # Расчет ATR
            atr_series = calculate_atr(
                data["high"], data["low"], data["close"], self._trend_config.atr_period
            )
            atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            # Расчет волатильности
            volatility = float(data["close"].pct_change().rolling(20).std().iloc[-1])
            # Расчет структуры рынка
            market_structure = self.technical_analysis.calculate_market_structure(data)
            # Исправление: безопасное извлечение данных из market_structure
            support_resistance = []
            if hasattr(market_structure, '__iter__') and not isinstance(market_structure, (str, bytes)):
                support_resistance = [
                    float(level) if hasattr(level, '__float__') else 0.0
                    for level in market_structure
                ]
            liquidity_zones: List[float] = []
            # Расчет импульса
            rsi_series = calculate_rsi(data["close"], 14)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            macd_result = calculate_macd(data["close"])
            if isinstance(macd_result, dict):
                macd = macd_result.get("macd", pd.Series())
                signal = macd_result.get("signal", pd.Series())
                hist = macd_result.get("histogram", pd.Series())
            else:
                macd = pd.Series()
                signal = pd.Series()
                hist = pd.Series()
            # Расчет диапазона
            high = float(data["high"].rolling(20).max().iloc[-1])
            low = float(data["low"].rolling(20).min().iloc[-1])
            range_size = float((high - low) / low)
            # Расчет объема
            volume_ma = float(data["volume"].rolling(20).mean().iloc[-1])
            volume_ratio = float(data["volume"].iloc[-1] / volume_ma)
            # Расчет риска
            risk = float(abs(entry_price - stop_loss))
            # Базовое соотношение риск/прибыль
            base_rr = 2.0  # Большее соотношение для тренда
            # Корректировка на волатильность
            volatility_multiplier = 1 + volatility * 5  # Больший множитель для тренда
            # Корректировка на диапазон
            range_multiplier = 1 + range_size * 2
            # Корректировка на объем
            volume_multiplier = 1 + (volume_ratio - 1) * 0.3
            # Корректировка на импульс
            momentum_multiplier = 1 + (abs(rsi - 50) / 50) * 0.3
            # Расчет финального множителя
            final_multiplier = float(
                volatility_multiplier
                * range_multiplier
                * volume_multiplier
                * momentum_multiplier
            )
            # Расчет тейка
            take_profit_distance = float(risk * base_rr * final_multiplier)
            # Поиск ближайшего уровня поддержки/сопротивления
            if position_type == "long":
                # Для длинной позиции ищем ближайший уровень сопротивления
                resistance_levels = [
                    float(level)
                    for level in support_resistance
                    if float(level) > float(entry_price)
                ]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(nearest_resistance - entry_price)
                    )
                # Проверка ликвидности
                liquidity_levels = [
                    float(level)
                    for level in liquidity_zones
                    if float(level) > float(entry_price)
                ]
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(nearest_liquidity - entry_price)
                    )
                return float(entry_price + take_profit_distance)
            else:
                # Для короткой позиции ищем ближайший уровень поддержки
                support_levels = [
                    float(level)
                    for level in support_resistance
                    if float(level) < float(entry_price)
                ]
                if support_levels:
                    nearest_support = max(support_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(entry_price - nearest_support)
                    )
                # Проверка ликвидности
                liquidity_levels = [
                    float(level)
                    for level in liquidity_zones
                    if float(level) < float(entry_price)
                ]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(entry_price - nearest_liquidity)
                    )
                return float(entry_price - take_profit_distance)
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return float(
                entry_price * 1.02 if position_type == "long" else entry_price * 0.98
            )

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных для определения тренда.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            dict[str, Any]: Результат анализа
        """
        from domain.type_definitions.strategy_types import (
            MarketRegime,
            StrategyAnalysis,
            StrategyMetrics,
        )

        try:
            # Валидация данных
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error_msg}")
            df = self._calculate_indicators(data.copy())
            # Используем только последние валидные данные для анализа
            # Находим первую строку без NaN в ключевых индикаторах
            key_columns = [
                "ema_fast",
                "ema_slow",
                "macd",
            ]  # Убираем ADX и ATR, так как они требуют больше данных
            valid_mask = df[key_columns].notna().all(axis=1)
            logger.info(
                f"Original data shape: {data.shape}, After indicators: {df.shape}, Valid points: {valid_mask.sum()}"
            )
            if valid_mask.sum() < 5:  # Уменьшаем минимальное количество точек
                raise ValueError(
                    f"Insufficient data for analysis: {valid_mask.sum()} points available"
                )
            # Используем только валидные данные
            df = df[valid_mask].copy()
            # Определение тренда
            trend_up = df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
            trend = "up" if trend_up else "down"
            # Проверка силы тренда
            trend_strength = self._check_trend_strength(df)
            # Определение рыночного режима
            market_regime = (
                MarketRegime.TRENDING_UP if trend_up else MarketRegime.TRENDING_DOWN
            )
            # Расчет метрик
            metrics = StrategyMetrics(
                total_signals=1,
                win_rate=0.5,
                expectancy=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=float(df["close"].pct_change().std()),
            )
            # Генерация сигналов
            signals = []
            if trend_strength:
                signal = self.generate_signal(data)
                if signal:
                    signals.append(signal)
            # Расчет индикаторов
            indicators = {
                "ema_fast": df["ema_fast"],
                "ema_slow": df["ema_slow"],
                "adx": df["adx"],
                "atr": df["atr"],
                "macd": df["macd"],
                "macd_signal": df["macd_signal"],
                "macd_hist": df["macd_hist"],
            }
            
            # Возвращаем словарь вместо StrategyAnalysis
            return {
                "strategy_id": f"trend_{id(self)}",
                "timestamp": pd.Timestamp.now(),
                "market_data": data,
                "indicators": indicators,
                "signals": signals,
                "metrics": metrics,
                "market_regime": market_regime,
                "confidence": min(
                    1.0, (df["adx"].iloc[-1] - self._trend_config.adx_threshold) / 20 + 0.7
                ),
                "risk_assessment": {
                    "volatility": float(df["atr"].iloc[-1]),
                    "trend_strength": float(df["adx"].iloc[-1]),
                    "support": float(df["low"].rolling(20).min().iloc[-1]),
                    "resistance": float(df["high"].rolling(20).max().iloc[-1]),
                },
                "recommendations": [
                    f"Trend direction: {trend}",
                    f"Trend strength: {'Strong' if trend_strength else 'Weak'}",
                    f"ADX: {df['adx'].iloc[-1]:.2f}",
                ],
                "metadata": {
                    "trend": trend,
                    "trend_strength": trend_strength,
                    "ema_fast": float(df["ema_fast"].iloc[-1]),
                    "ema_slow": float(df["ema_slow"].iloc[-1]),
                    "adx": float(df["adx"].iloc[-1]),
                    "atr": float(df["atr"].iloc[-1]),
                },
            }
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            raise

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала на основе трендовой логики и индикаторов.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[Signal]: Торговый сигнал или None
        """
        from domain.type_definitions.strategy_types import Signal as DomainSignal

        try:
            df = self._calculate_indicators(data.copy())
            if df.shape[0] < max(self._trend_config.ema_fast, self._trend_config.ema_slow):
                return None
            # Определение тренда
            trend_up = df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
            adx = df["adx"].iloc[-1]
            macd_hist = df["macd_hist"].iloc[-1]
            close = df["close"].iloc[-1]
            # Сигнал на покупку
            if trend_up and adx > self._trend_config.adx_threshold and macd_hist > 0:
                entry_price = close
                stop_loss = self._calculate_stop_loss(df, entry_price, "long")
                take_profit = self._calculate_take_profit(
                    df, entry_price, stop_loss, "long"
                )
                confidence = min(1.0, (adx - self._trend_config.adx_threshold) / 20 + 0.7)
                signal = DomainSignal(
                    direction="long",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                )
                # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ сигнала
                if not self._validate_signal(signal):
                    return None
                return signal
            # Сигнал на продажу
            elif not trend_up and adx > self._trend_config.adx_threshold and macd_hist < 0:
                entry_price = close
                stop_loss = self._calculate_stop_loss(df, entry_price, "short")
                take_profit = self._calculate_take_profit(
                    df, entry_price, stop_loss, "short"
                )
                confidence = min(1.0, (adx - self._trend_config.adx_threshold) / 20 + 0.7)
                signal = DomainSignal(
                    direction="short",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                )
                # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ сигнала
                if not self._validate_signal(signal):
                    return None
                return signal
            else:
                return None
        except Exception as e:
            logger.error(f"Error generating trend signal: {str(e)}")
            return None


def trend_strategy_ema_macd(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе EMA и MACD.
    Args:
        data: Рыночные данные
    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = TrendStrategy()
        data = strategy._calculate_indicators(data)
        # Проверка силы тренда
        if not strategy._check_trend_strength(data):
            return None
        # Определение тренда по EMA
        ema_trend = (
            data["ema_fast"].iloc[-1]
            > data["ema_medium"].iloc[-1]
            > data["ema_slow"].iloc[-1]
        )
        # Проверка кроссовера MACD
        macd_cross_up = (
            data["macd"].iloc[-2] < data["macd_signal"].iloc[-2]
            and data["macd"].iloc[-1] > data["macd_signal"].iloc[-1]
        )
        macd_cross_down = (
            data["macd"].iloc[-2] > data["macd_signal"].iloc[-2]
            and data["macd"].iloc[-1] < data["macd_signal"].iloc[-1]
        )
        # Генерация сигнала
        if ema_trend and macd_cross_up:
            side = "buy"
        elif not ema_trend and macd_cross_down:
            side = "sell"
        else:
            return None
        # Расчет стоп-лосса и тейк-профита
        stop_loss_func = strategy._calculate_stop_loss
        take_profit_func = strategy._calculate_take_profit
        
        # Безопасный вызов функций
        if callable(stop_loss_func):
            stop_loss = stop_loss_func(data, data["close"].iloc[-1], side)
        else:
            stop_loss = 0.0
            
        if callable(take_profit_func):
            take_profit = take_profit_func(
                data, data["close"].iloc[-1], stop_loss, side
            )
        else:
            take_profit = 0.0
        return {
            "side": side,
            "entry_price": data["close"].iloc[-1],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }
    except Exception as e:
        logger.error(f"Error in EMA-MACD strategy: {str(e)}")
        return None


def trend_strategy_price_action(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе Price Action.
    Args:
        data: Рыночные данные
    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = TrendStrategy()
        data = strategy._calculate_indicators(data)
        # Проверка силы тренда
        if not strategy._check_trend_strength(data):
            return None
        # Определение тренда
        trend = (
            "up" if data["ema_fast"].iloc[-1] > data["ema_slow"].iloc[-1] else "down"
        )
        # Проверка паттернов
        last_candle: pd.Series = data.iloc[-1] if hasattr(data, "iloc") and callable(data.iloc) else None
        prev_candle: pd.Series = data.iloc[-2] if hasattr(data, "iloc") and callable(data.iloc) else None
        
        if last_candle is None or prev_candle is None:
            return None
            
        # Простая проверка импульсной свечи (заменяем несуществующую функцию)
        is_impulse = (
            abs(last_candle["close"] - last_candle["open"]) > 
            abs(prev_candle["close"] - prev_candle["open"])
        )
        # Простая проверка внутреннего бара (заменяем несуществующую функцию)
        is_inner = (
            last_candle["high"] <= prev_candle["high"] and 
            last_candle["low"] >= prev_candle["low"]
        )
        # Проверка дивергенции MACD
        macd_divergence = (
            trend == "up"
            and data["close"].iloc[-1] > data["close"].iloc[-2]
            and data["macd"].iloc[-1] < data["macd"].iloc[-2]
        ) or (
            trend == "down"
            and data["close"].iloc[-1] < data["close"].iloc[-2]
            and data["macd"].iloc[-1] > data["macd"].iloc[-2]
        )
        # Генерация сигнала
        if trend == "up" and is_impulse and not is_inner and macd_divergence:
            side = "buy"
        elif trend == "down" and is_impulse and not is_inner and macd_divergence:
            side = "sell"
        else:
            return None
        # Расчет стоп-лосса и тейк-профита
        stop_loss_func = strategy._calculate_stop_loss
        take_profit_func = strategy._calculate_take_profit
        
        # Безопасный вызов функций
        if callable(stop_loss_func):
            stop_loss = stop_loss_func(data, data["close"].iloc[-1], side)
        else:
            stop_loss = 0.0
            
        if callable(take_profit_func):
            take_profit = take_profit_func(
                data, data["close"].iloc[-1], stop_loss, side
            )
        else:
            take_profit = 0.0
        return {
            "side": side,
            "entry_price": data["close"].iloc[-1],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }
    except Exception as e:
        logger.error(f"Error in Price Action strategy: {str(e)}")
        return None
