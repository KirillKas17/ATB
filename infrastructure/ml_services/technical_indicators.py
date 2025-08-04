"""
Технические индикаторы для анализа рынка.
"""

import pandas as pd
from shared.numpy_utils import np
from typing import Dict, List, Optional, Any, Union, cast
from pandas import Series, DataFrame

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""

    @staticmethod
    def calculate_ema(data: Series, period: int) -> Series:
        """Расчет экспоненциальной скользящей средней"""
        return data.ewm(span=period).mean()

    @staticmethod
    def calculate_sma(data: Series, period: int) -> Series:
        """Расчет простой скользящей средней"""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(data: Series, period: int = 14) -> Series:
        """Расчет RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = delta.where(delta < 0, 0).abs().rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        data: Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, Series]:
        """Расчет MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_bollinger_bands(
        data: Series, period: int = 20, std_dev: float = 2
    ) -> Dict[str, Series]:
        """Расчет полос Боллинджера"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {"upper": upper_band, "middle": sma, "lower": lower_band}

    @staticmethod
    def calculate_atr(
        high: Series, low: Series, close: Series, period: int = 14
    ) -> Series:
        """Расчет Average True Range"""
        high_low = high - low
        if hasattr(high, '__sub__') and hasattr(close, 'shift'):
            high_close: pd.Series = (high - close.shift()).abs()
            low_close: pd.Series = (low - close.shift()).abs()
        else:
            high_close = pd.Series()
            low_close = pd.Series()
        true_range = pd.DataFrame({"hl": high_low, "hc": high_close, "lc": low_close}).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_stochastic(
        high: Series,
        low: Series,
        close: Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, Series]:
        """Расчет стохастического осциллятора"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return {"k": k_percent, "d": d_percent}

    @staticmethod
    def calculate_williams_r(
        high: Series, low: Series, close: Series, period: int = 14
    ) -> Series:
        """Расчет Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def calculate_cci(
        high: Series, low: Series, close: Series, period: int = 20
    ) -> Series:
        """Расчет Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean((x - x.mean()).abs())
        )
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def calculate_adx(
        high: Series, low: Series, close: Series, period: int = 14
    ) -> Dict[str, Series]:
        """Расчет Average Directional Index"""
        # True Range
        tr1 = high - low
        if hasattr(high, '__sub__') and hasattr(close, 'shift'):
            tr2: pd.Series = (high - close.shift()).abs()
            tr3: pd.Series = (low - close.shift()).abs()
        else:
            tr2 = pd.Series()
            tr3 = pd.Series()
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        # Создаем Series с нулями для сравнения
        zero_series = pd.Series([0.0] * len(up_move), index=up_move.index)
        # Проверяем, что up_move и down_move поддерживают операции сравнения
        if hasattr(up_move, '__gt__') and hasattr(down_move, '__gt__'):
            plus_dm = up_move.where((up_move.gt(down_move)) & (up_move.gt(0.0)), 0.0)
            minus_dm = down_move.where((down_move.gt(up_move)) & (down_move.gt(0.0)), 0.0)
        else:
            # Альтернативный способ для не-pandas объектов
            if hasattr(up_move, '__iter__') and hasattr(down_move, '__iter__'):
                plus_dm = pd.Series([max(0.0, float(u)) if float(u) > float(d) and float(u) > 0.0 else 0.0 for u, d in zip(up_move, down_move)], index=up_move.index)
                minus_dm = pd.Series([max(0.0, float(d)) if float(d) > float(u) and float(d) > 0.0 else 0.0 for u, d in zip(up_move, down_move)], index=down_move.index)
            else:
                plus_dm = pd.Series(0.0, index=up_move.index)
                minus_dm = pd.Series(0.0, index=down_move.index)
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        if isinstance(dx, pd.Series):
            adx = dx.rolling(window=period).mean()
        else:
            # Если dx это numpy array, конвертируем в Series
            dx_series = pd.Series(dx)
            adx = dx_series.rolling(window=period).mean()
        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

    @staticmethod
    def get_all_indicators(data: DataFrame) -> Dict[str, float]:
        """Получение всех технических индикаторов для последней свечи"""
        if len(data) < 50:
            return {}
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        indicators: Dict[str, float] = {}
        
        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close)
        if isinstance(rsi, Series) and not rsi.empty:
            indicators["rsi"] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        else:
            indicators["rsi"] = 50.0
            
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(close)
        if isinstance(macd_data["macd"], Series) and not macd_data["macd"].empty:
            indicators["macd"] = (
                float(macd_data["macd"].iloc[-1])
                if not np.isnan(macd_data["macd"].iloc[-1])
                else 0.0
            )
        else:
            indicators["macd"] = 0.0
        if isinstance(macd_data["signal"], Series) and not macd_data["signal"].empty:
            indicators["macd_signal"] = (
                float(macd_data["signal"].iloc[-1])
                if not np.isnan(macd_data["signal"].iloc[-1])
                else 0.0
            )
        else:
            indicators["macd_signal"] = 0.0
        if isinstance(macd_data["histogram"], Series) and not macd_data["histogram"].empty:
            indicators["macd_histogram"] = (
                float(macd_data["histogram"].iloc[-1])
                if not np.isnan(macd_data["histogram"].iloc[-1])
                else 0.0
            )
        else:
            indicators["macd_histogram"] = 0.0
            
        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(close)
        if isinstance(bb_data["upper"], Series) and not bb_data["upper"].empty:
            indicators["bb_upper"] = (
                float(bb_data["upper"].iloc[-1])
                if not np.isnan(bb_data["upper"].iloc[-1])
                else 0.0
            )
        else:
            indicators["bb_upper"] = 0.0
        if isinstance(bb_data["middle"], Series) and not bb_data["middle"].empty:
            indicators["bb_middle"] = (
                float(bb_data["middle"].iloc[-1])
                if not np.isnan(bb_data["middle"].iloc[-1])
                else 0.0
            )
        else:
            indicators["bb_middle"] = 0.0
        if isinstance(bb_data["lower"], Series) and not bb_data["lower"].empty:
            indicators["bb_lower"] = (
                float(bb_data["lower"].iloc[-1])
                if not np.isnan(bb_data["lower"].iloc[-1])
                else 0.0
            )
        else:
            indicators["bb_lower"] = 0.0
            
        # ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close)
        indicators["atr"] = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(high, low, close)
        indicators["stoch_k"] = (
            float(stoch_data["k"].iloc[-1])
            if not np.isnan(stoch_data["k"].iloc[-1])
            else 50.0
        )
        indicators["stoch_d"] = (
            float(stoch_data["d"].iloc[-1])
            if not np.isnan(stoch_data["d"].iloc[-1])
            else 50.0
        )
        
        # Williams %R
        williams_r = TechnicalIndicators.calculate_williams_r(high, low, close)
        indicators["williams_r"] = (
            float(williams_r.iloc[-1])
            if not np.isnan(williams_r.iloc[-1])
            else -50.0
        )
        
        # CCI
        cci_result = TechnicalIndicators.calculate_cci(high, low, close)
        if callable(cci_result):
            cci = cci_result()
        else:
            cci = cci_result
        if isinstance(cci, Series) and len(cci) > 0:
            indicators["cci"] = float(cci.iloc[-1]) if not np.isnan(cci.iloc[-1]) else 0.0
        else:
            indicators["cci"] = 0.0
        
        # ADX
        adx_result = TechnicalIndicators.calculate_adx(high, low, close)
        if isinstance(adx_result, dict):
            adx_data = adx_result
        else:
            adx_data = {}  # type: ignore[unreachable]
        if isinstance(adx_data, dict) and "adx" in adx_data and isinstance(adx_data["adx"], Series) and len(adx_data["adx"]) > 0:
            adx_series: pd.Series = adx_data["adx"]
            indicators["adx"] = (
                float(adx_series.iloc[-1])
                if not np.isnan(adx_series.iloc[-1])
                else 25.0
            )
        else:
            indicators["adx"] = 25.0
        if isinstance(adx_data, dict) and "plus_di" in adx_data and isinstance(adx_data["plus_di"], Series) and len(adx_data["plus_di"]) > 0:
            plus_di_series: pd.Series = adx_data["plus_di"]
            indicators["plus_di"] = (
                float(plus_di_series.iloc[-1])
                if not np.isnan(plus_di_series.iloc[-1])
                else 25.0
            )
        else:
            indicators["plus_di"] = 25.0
        if isinstance(adx_data, dict) and "minus_di" in adx_data and isinstance(adx_data["minus_di"], Series) and len(adx_data["minus_di"]) > 0:
            minus_di_series: pd.Series = adx_data["minus_di"]
            indicators["minus_di"] = (
                float(minus_di_series.iloc[-1])
                if not np.isnan(minus_di_series.iloc[-1])
                else 25.0
            )
        else:
            indicators["minus_di"] = 25.0
        return indicators

    @staticmethod
    def calculate_vwap(data: DataFrame) -> Series:
        """Расчет VWAP (Volume Weighted Average Price)"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap

    @staticmethod
    def calculate_pivot_points(data: DataFrame) -> Dict[str, float]:
        """Расчет точек разворота"""
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        close = data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return {
            'pivot': float(pivot),
            'r1': float(r1),
            'r2': float(r2),
            's1': float(s1),
            's2': float(s2)
        }

    @staticmethod
    def calculate_fibonacci_retracements(data: DataFrame) -> Dict[str, float]:
        """Расчет уровней Фибоначчи"""
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        
        return {
            '0.0': float(low),
            '0.236': float(low + 0.236 * diff),
            '0.382': float(low + 0.382 * diff),
            '0.5': float(low + 0.5 * diff),
            '0.618': float(low + 0.618 * diff),
            '0.786': float(low + 0.786 * diff),
            '1.0': float(high)
        }

    @staticmethod
    def detect_support_resistance(data: DataFrame) -> Dict[str, List[float]]:
        """Обнаружение уровней поддержки и сопротивления"""
        # Простая реализация - поиск локальных минимумов и максимумов
        highs = data['high'].rolling(window=5, center=True).max()
        lows = data['low'].rolling(window=5, center=True).min()
        
        resistance_levels = highs.dropna().unique().tolist()
        support_levels = lows.dropna().unique().tolist()
        
        return {
            'support': sorted(support_levels)[:5],  # Топ-5 уровней поддержки
            'resistance': sorted(resistance_levels, reverse=True)[:5]  # Топ-5 уровней сопротивления
        }

    @staticmethod
    def calculate_volume_profile(data: DataFrame) -> Dict[str, Any]:
        """Расчет профиля объема"""
        # Простая реализация
        price_bins = pd.cut(data['close'], bins=10)
        volume_profile = data.groupby(price_bins)['volume'].sum()
        
        return {
            'price_levels': [float(interval.mid) for interval in volume_profile.index],
            'volumes': volume_profile.values.tolist(),
            'poc': float(data['close'].iloc[-1])  # Point of Control
        }

    @staticmethod
    def calculate_market_structure(data: DataFrame) -> Dict[str, Any]:
        """Расчет структуры рынка"""
        # Простая реализация
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        
        return {
            'trend': 'uptrend' if data['close'].iloc[-1] > data['close'].iloc[-20] else 'downtrend',
            'volatility': float(data['close'].pct_change().std()),
            'momentum': float(data['close'].pct_change().mean()),
            'highs': highs.dropna().tolist(),
            'lows': lows.dropna().tolist()
        }

    @staticmethod
    def calculate_volatility_indicators(data: DataFrame) -> Dict[str, float]:
        """Расчет индикаторов волатильности"""
        returns = data['close'].pct_change()
        
        # Получаем ATR
        atr_result = TechnicalIndicators.calculate_atr(data['high'], data['low'], data['close'])
        if callable(atr_result):
            atr = atr_result()
        else:
            atr = atr_result
            
        # Получаем Bollinger Bands
        bb_result = TechnicalIndicators.calculate_bollinger_bands(data['close'])
        if callable(bb_result):
            bb_data = bb_result()
        else:
            bb_data = bb_result
        
        atr_value = 0.0
        if hasattr(atr, 'iloc') and len(atr) > 0:
            atr_value = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
            
        bb_width = 0.0
        if hasattr(bb_data, '__getitem__') and "upper" in bb_data and "lower" in bb_data:
            if hasattr(bb_data['upper'], 'iloc') and hasattr(bb_data['lower'], 'iloc'):
                bb_width = float(bb_data['upper'].iloc[-1] - bb_data['lower'].iloc[-1]) if not np.isnan(bb_data['upper'].iloc[-1]) and not np.isnan(bb_data['lower'].iloc[-1]) else 0.0
        
        return {
            'volatility': float(returns.std()),
            'atr': atr_value,
            'bollinger_width': bb_width
        }

    @staticmethod
    def calculate_momentum_indicators(data: DataFrame) -> Dict[str, float]:
        """Расчет индикаторов моментума"""
        # Получаем RSI
        rsi_result = TechnicalIndicators.calculate_rsi(data['close'])
        if callable(rsi_result):
            rsi = rsi_result()
        else:
            rsi = rsi_result
            
        # Получаем Stochastic
        stoch_result = TechnicalIndicators.calculate_stochastic(data['high'], data['low'], data['close'])
        if callable(stoch_result):
            stoch_data = stoch_result()
        else:
            stoch_data = stoch_result
            
        # Получаем Williams %R
        williams_result = TechnicalIndicators.calculate_williams_r(data['high'], data['low'], data['close'])
        if callable(williams_result):
            williams_r = williams_result()
        else:
            williams_r = williams_result
        
        rsi_value = 50.0
        if hasattr(rsi, 'iloc') and len(rsi) > 0:
            rsi_value = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            
        stoch_k_value = 50.0
        stoch_d_value = 50.0
        if hasattr(stoch_data, '__getitem__') and "k" in stoch_data and "d" in stoch_data:
            if hasattr(stoch_data['k'], 'iloc') and len(stoch_data['k']) > 0:
                stoch_k_value = float(stoch_data['k'].iloc[-1]) if not np.isnan(stoch_data['k'].iloc[-1]) else 50.0
            if hasattr(stoch_data['d'], 'iloc') and len(stoch_data['d']) > 0:
                stoch_d_value = float(stoch_data['d'].iloc[-1]) if not np.isnan(stoch_data['d'].iloc[-1]) else 50.0
                
        williams_value = -50.0
        if hasattr(williams_r, 'iloc') and len(williams_r) > 0:
            williams_value = float(williams_r.iloc[-1]) if not np.isnan(williams_r.iloc[-1]) else -50.0
        
        return {
            'rsi': rsi_value,
            'stochastic_k': stoch_k_value,
            'stochastic_d': stoch_d_value,
            'williams_r': williams_value
        }

    @staticmethod
    def calculate_trend_indicators(data: DataFrame) -> Dict[str, float]:
        """Расчет трендовых индикаторов"""
        # Получаем SMA
        sma_20_result = TechnicalIndicators.calculate_sma(data['close'], 20)
        if callable(sma_20_result):
            sma_20 = sma_20_result()
        else:
            sma_20 = sma_20_result
            
        sma_50_result = TechnicalIndicators.calculate_sma(data['close'], 50)
        if callable(sma_50_result):
            sma_50 = sma_50_result()
        else:
            sma_50 = sma_50_result
            
        # Получаем EMA
        ema_20_result = TechnicalIndicators.calculate_ema(data['close'], 20)
        if callable(ema_20_result):
            ema_20 = ema_20_result()
        else:
            ema_20 = ema_20_result
            
        # Получаем MACD
        macd_result = TechnicalIndicators.calculate_macd(data['close'])
        if callable(macd_result):
            macd_data = macd_result()
        else:
            macd_data = macd_result
        
        sma_20_value = 0.0
        if hasattr(sma_20, 'iloc') and len(sma_20) > 0:
            sma_20_value = float(sma_20.iloc[-1]) if not np.isnan(sma_20.iloc[-1]) else 0.0
            
        sma_50_value = 0.0
        if hasattr(sma_50, 'iloc') and len(sma_50) > 0:
            sma_50_value = float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else 0.0
            
        ema_20_value = 0.0
        if hasattr(ema_20, 'iloc') and len(ema_20) > 0:
            ema_20_value = float(ema_20.iloc[-1]) if not np.isnan(ema_20.iloc[-1]) else 0.0
            
        macd_value = 0.0
        macd_signal_value = 0.0
        if hasattr(macd_data, '__getitem__') and "macd" in macd_data and "signal" in macd_data:
            if hasattr(macd_data['macd'], 'iloc') and len(macd_data['macd']) > 0:
                macd_value = float(macd_data['macd'].iloc[-1]) if not np.isnan(macd_data['macd'].iloc[-1]) else 0.0
            if hasattr(macd_data['signal'], 'iloc') and len(macd_data['signal']) > 0:
                macd_signal_value = float(macd_data['signal'].iloc[-1]) if not np.isnan(macd_data['signal'].iloc[-1]) else 0.0
        
        return {
            'sma_20': sma_20_value,
            'sma_50': sma_50_value,
            'ema_20': ema_20_value,
            'macd': macd_value,
            'macd_signal': macd_signal_value
        }

    @staticmethod
    def calculate_volume_indicators(data: DataFrame) -> Dict[str, float]:
        """Расчет объемных индикаторов"""
        return {
            'volume_sma': float(data['volume'].rolling(window=20).mean().iloc[-1]),
            'volume_ratio': float(data['volume'].iloc[-1] / data['volume'].rolling(window=20).mean().iloc[-1]),
            'obv': float(data['volume'].cumsum().iloc[-1])  # Простая реализация OBV
        }

    @staticmethod
    def generate_trading_signals(data: DataFrame) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов"""
        signals = []
        
        # RSI сигналы
        rsi_result = TechnicalIndicators.calculate_rsi(data['close'])
        if callable(rsi_result):
            rsi = rsi_result()
        else:
            rsi = rsi_result
            
        if hasattr(rsi, 'iloc') and len(rsi) > 0:
            rsi_value = float(rsi.iloc[-1])
            if rsi_value < 30:
                signals.append({
                    'type': 'buy',
                    'indicator': 'rsi',
                    'strength': 0.8,
                    'value': rsi_value
                })
            elif rsi_value > 70:
                signals.append({
                    'type': 'sell',
                    'indicator': 'rsi',
                    'strength': 0.7,
                    'value': rsi_value
                })
        
        # MACD сигналы
        macd_result = TechnicalIndicators.calculate_macd(data['close'])
        if callable(macd_result):
            macd_data = macd_result()
        else:
            macd_data = macd_result
            
        if hasattr(macd_data, '__getitem__') and "macd" in macd_data and "signal" in macd_data:
            if hasattr(macd_data['macd'], 'iloc') and hasattr(macd_data['signal'], 'iloc'):
                if len(macd_data['macd']) > 1 and len(macd_data['signal']) > 1:
                    macd_current = float(macd_data['macd'].iloc[-1])
                    signal_current = float(macd_data['signal'].iloc[-1])
                    macd_prev = float(macd_data['macd'].iloc[-2])
                    signal_prev = float(macd_data['signal'].iloc[-2])
                    
                    if macd_current > signal_current and macd_prev <= signal_prev:
                        signals.append({
                            'type': 'buy',
                            'indicator': 'macd',
                            'strength': 0.6,
                            'value': macd_current
                        })
        
        return signals

    @staticmethod
    def calculate_risk_metrics(data: DataFrame) -> Dict[str, float]:
        """Расчет метрик риска"""
        returns = data['close'].pct_change().dropna()
        
        return {
            'var_95': float(returns.quantile(0.05)),
            'max_drawdown': float((data['close'] / data['close'].cummax() - 1).min()),
            'sharpe_ratio': float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
            'volatility': float(returns.std())
        }

    @staticmethod
    def detect_patterns(data: DataFrame) -> List[str]:
        """Обнаружение паттернов"""
        patterns = []
        
        # Простая реализация - проверка трендов
        if len(data) >= 20:
            sma_result = TechnicalIndicators.calculate_sma(data['close'], 20)
            if callable(sma_result):
                sma_20 = sma_result()
            else:
                sma_20 = sma_result
                
            if hasattr(data['close'], 'iloc') and hasattr(sma_20, 'iloc'):
                if len(data['close']) > 0 and len(sma_20) > 0:
                    if data['close'].iloc[-1] > sma_20.iloc[-1]:
                        patterns.append('uptrend')
                    else:
                        patterns.append('downtrend')
        
        return patterns

    @staticmethod
    def calculate_correlation_matrix(data: DataFrame) -> Dict[str, Dict[str, float]]:
        """Расчет корреляционной матрицы"""
        # Простая реализация для одного инструмента
        returns = data['close'].pct_change().dropna()
        
        return {
            'close': {
                'close': 1.0,
                'volume': float(returns.corr(data['volume'].pct_change().dropna())) if len(data) > 1 else 0.0
            }
        }

    @staticmethod
    def calculate_market_regime(data: DataFrame) -> str:
        """Определение режима рынка"""
        volatility = data['close'].pct_change().std()
        trend = data['close'].iloc[-1] - data['close'].iloc[-20] if len(data) >= 20 else 0
        
        if volatility > 0.05:  # Высокая волатильность
            return 'volatile'
        elif trend > 0:
            return 'bull'
        elif trend < 0:
            return 'bear'
        else:
            return 'sideways'

    @staticmethod
    def calculate_liquidity_metrics(data: DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
        """Расчет продвинутых метрик ликвидности"""
        
        try:
            if 'volume' not in data.columns or len(data) == 0:
                return {
                    'avg_volume': 0.0,
                    'volume_volatility': 0.0,
                    'spread_estimate': 0.001,
                    'liquidity_score': 0.0,
                    'market_impact': 0.0,
                    'bid_ask_spread': 0.001,
                    'depth_of_market': 0.0,
                    'turnover_ratio': 0.0
                }
            
            volumes = data['volume'].values
            prices = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
            
            # Базовые метрики
            avg_volume = float(np.mean(volumes))
            volume_volatility = float(np.std(volumes))
            
            # Продвинутые метрики ликвидности
            
            # 1. Коэффициент вариации объема
            cv_volume = volume_volatility / avg_volume if avg_volume > 0 else 0
            
            # 2. Амихуд Illiquidity Ratio
            returns = np.diff(prices) / prices[:-1]
            daily_returns = np.abs(returns[1:])  # Абсолютные доходности
            dollar_volumes = volumes[2:] * prices[2:]  # Долларовый объем
            
            amihud_ratios = []
            for i in range(len(daily_returns)):
                if dollar_volumes[i] > 0:
                    amihud_ratios.append(daily_returns[i] / dollar_volumes[i] * 1e6)
            
            amihud_illiquidity = float(np.mean(amihud_ratios)) if amihud_ratios else 0.0
            
            # 3. Spread estimate на основе Roll model
            if len(returns) > 1:
                # Коvariance между соседними доходностями
                cov_returns = np.cov(returns[:-1], returns[1:])[0, 1]
                spread_estimate = 2 * np.sqrt(-cov_returns) if cov_returns < 0 else 0.001
            else:
                spread_estimate = 0.001
                
            # 4. Pastor-Stambaugh Liquidity Measure
            ps_liquidity = 0.0
            if len(returns) > 20:
                # Регрессия доходности на объем с лагом
                volume_impact = []
                for i in range(1, min(len(returns), len(volumes)-1)):
                    if volumes[i] > 0:
                        volume_impact.append(abs(returns[i]) / volumes[i])
                ps_liquidity = float(np.mean(volume_impact)) if volume_impact else 0.0
            
            # 5. Market Depth Proxy (через волатильность цен к объему)
            price_volatility = np.std(prices)
            depth_proxy = avg_volume / price_volatility if price_volatility > 0 else 0.0
            
            # 6. Turnover Ratio
            if len(prices) > 1:
                market_cap_proxy = np.mean(prices) * avg_volume  # Упрощенная оценка
                turnover_ratio = avg_volume / market_cap_proxy if market_cap_proxy > 0 else 0.0
            else:
                turnover_ratio = 0.0
            
            # 7. Composite Liquidity Score (нормализованный индекс от 0 до 1)
            # Чем выше объем и меньше спред/волатильность, тем выше ликвидность
            liquidity_components = {
                'volume_score': min(avg_volume / 1000000, 1.0),  # Нормализация по миллиону
                'spread_score': max(0, 1 - spread_estimate * 1000),  # Инвертированный спред
                'stability_score': max(0, 1 - cv_volume),  # Стабильность объема
                'depth_score': min(depth_proxy / 10000, 1.0)  # Глубина рынка
            }
            
            liquidity_score = np.mean(list(liquidity_components.values()))
            
            return {
                'avg_volume': float(avg_volume),
                'volume_volatility': float(volume_volatility),
                'spread_estimate': float(spread_estimate),
                'liquidity_score': float(liquidity_score),
                'market_impact': float(amihud_illiquidity),
                'bid_ask_spread': float(spread_estimate),
                'depth_of_market': float(depth_proxy),
                'turnover_ratio': float(turnover_ratio),
                'cv_volume': float(cv_volume),
                'ps_liquidity': float(ps_liquidity),
                'liquidity_components': {k: float(v) for k, v in liquidity_components.items()}
            }
            
        except Exception as e:
            # В случае ошибки возвращаем базовые значения
            return {
                'avg_volume': float(data['volume'].mean()) if 'volume' in data.columns else 0.0,
                'volume_volatility': float(data['volume'].std()) if 'volume' in data.columns else 0.0,
                'spread_estimate': 0.001,
                'liquidity_score': 0.5,
                'market_impact': 0.0001,
                'bid_ask_spread': 0.001,
                'depth_of_market': 1000.0,
                'turnover_ratio': 0.1
            }

    @staticmethod
    def calculate_market_impact(data: DataFrame) -> float:
        """Расчет продвинутого влияния на рынок (Market Impact)"""
        
        try:
            if len(data) < 10 or 'volume' not in data.columns:
                return 0.0001
            
            prices = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
            volumes = data['volume'].values
            
            # Метод 1: Kyle's Lambda (линейная модель влияния)
            # Изменение цены = λ * объем торгов
            returns = np.diff(prices) / prices[:-1]
            volume_changes = np.diff(volumes)
            
            if len(returns) > 1 and len(volume_changes) > 1:
                # Корреляция между изменениями объема и доходностью
                correlation = np.corrcoef(abs(returns[:-1]), abs(volume_changes[1:]))[0, 1]
                if not np.isnan(correlation):
                    kyle_lambda = correlation * np.std(returns) / np.mean(volumes)
                else:
                    kyle_lambda = 0.0001
            else:
                kyle_lambda = 0.0001
            
            # Метод 2: Almgren-Chriss модель
            # Влияние зависит от волатильности и ликвидности
            price_volatility = np.std(returns) if len(returns) > 0 else 0.01
            avg_volume = np.mean(volumes)
            
            # Постоянная составляющая влияния
            permanent_impact = price_volatility / np.sqrt(avg_volume) if avg_volume > 0 else 0.001
            
            # Временная составляющая влияния
            temporary_impact = price_volatility * 0.5 / avg_volume if avg_volume > 0 else 0.0005
            
            # Общее влияние на рынок
            total_impact = permanent_impact + temporary_impact
            
            # Метод 3: Влияние на основе глубины рынка
            # Предполагаем, что глубина обратно пропорциональна волатильности
            market_depth = avg_volume / price_volatility if price_volatility > 0 else avg_volume
            depth_impact = 1.0 / market_depth if market_depth > 0 else 0.001
            
            # Комбинированная оценка влияния на рынок
            combined_impact = np.mean([kyle_lambda, total_impact, depth_impact])
            
            # Нормализация и ограничение разумными пределами
            market_impact = max(0.00001, min(combined_impact, 0.01))  # От 0.001% до 1%
            
            return float(market_impact)
            
        except Exception as e:
            # В случае ошибки возвращаем консервативную оценку
            return 0.0001
