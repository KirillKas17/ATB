"""
Продвинутый движок прогнозирования с современными методами анализа
Включает анализ FVG (Fair Value Gaps), SNR (Signal-to-Noise Ratio), OrderFlow и другие методы
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) - разрыв справедливой стоимости"""
    start_time: datetime
    end_time: datetime
    high: float
    low: float
    direction: str  # bullish/bearish
    strength: float  # 0-1
    volume_confirmation: bool
    retest_count: int = 0
    filled_percentage: float = 0.0
    
    @property
    def size(self) -> float:
        """Размер FVG в пунктах"""
        return abs(self.high - self.low)
    
    @property
    def mid_point(self) -> float:
        """Средняя точка FVG"""
        return (self.high + self.low) / 2


@dataclass
class OrderFlowImbalance:
    """Дисбаланс ордерфлоу"""
    timestamp: datetime
    price_level: float
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float  # (buy - sell) / (buy + sell)
    significance: float  # 0-1
    
    @property
    def is_bullish(self) -> bool:
        return self.imbalance_ratio > 0.3
    
    @property
    def is_bearish(self) -> bool:
        return self.imbalance_ratio < -0.3


@dataclass
class LiquidityLevel:
    """Уровень ликвидности"""
    price: float
    volume: float
    strength: float
    level_type: str  # support/resistance/poc
    touch_count: int
    last_touch: datetime
    significance: float
    
    
@dataclass
class SignalNoiseMetrics:
    """Метрики соотношения сигнал/шум"""
    snr_ratio: float
    signal_strength: float
    noise_level: float
    clarity_score: float  # 0-1
    confidence: float
    
    @property
    def is_high_quality(self) -> bool:
        return self.snr_ratio > 2.0 and self.clarity_score > 0.7


@dataclass
class AdvancedPrediction:
    """Продвинутый прогноз"""
    timestamp: datetime
    symbol: str
    direction: str  # buy/sell/neutral
    confidence: float
    target_price: float
    stop_loss: float
    timeframe: str
    
    # Компоненты анализа
    fvg_signals: List[FairValueGap]
    orderflow_signals: List[OrderFlowImbalance]
    liquidity_levels: List[LiquidityLevel]
    snr_metrics: SignalNoiseMetrics
    
    # Дополнительная информация
    risk_reward_ratio: float
    expected_duration: timedelta
    market_structure: str  # trending/ranging/transition
    volatility_regime: str  # low/normal/high
    
    
class AdvancedPredictionEngine:
    """Продвинутый движок прогнозирования"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Конфигурация анализа
        """
        self.config = config or {}
        self.fvg_history: List[FairValueGap] = []
        self.orderflow_history: List[OrderFlowImbalance] = []
        self.liquidity_levels: List[LiquidityLevel] = []
        
        # Параметры анализа
        self.min_fvg_size = self.config.get("min_fvg_size", 0.001)  # 0.1%
        self.fvg_lookback = self.config.get("fvg_lookback", 100)
        self.orderflow_window = self.config.get("orderflow_window", 20)
        self.liquidity_threshold = self.config.get("liquidity_threshold", 0.7)
        self.snr_window = self.config.get("snr_window", 50)
        
        logger.info(f"AdvancedPredictionEngine initialized with config: {self.config}")
    
    def analyze_market(
        self, 
        symbol: str,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        volume_profile: Optional[Dict[str, Any]] = None
    ) -> AdvancedPrediction:
        """
        Комплексный анализ рынка с продвинутыми методами
        
        Args:
            symbol: Торговая пара
            ohlcv_data: OHLCV данные
            orderbook_data: Данные ордербука
            volume_profile: Профиль объема
            
        Returns:
            AdvancedPrediction: Продвинутый прогноз
        """
        logger.info(f"Starting advanced analysis for {symbol}")
        
        try:
            # 1. Анализ Fair Value Gaps
            fvg_signals = self._analyze_fair_value_gaps(ohlcv_data)
            
            # 2. Анализ OrderFlow
            orderflow_signals = self._analyze_orderflow(ohlcv_data, orderbook_data)
            
            # 3. Анализ уровней ликвидности
            liquidity_levels = self._analyze_liquidity_levels(ohlcv_data, volume_profile)
            
            # 4. Расчет Signal-to-Noise Ratio
            snr_metrics = self._calculate_snr_metrics(ohlcv_data)
            
            # 5. Определение структуры рынка
            market_structure = self._determine_market_structure(ohlcv_data)
            
            # 6. Анализ режима волатильности
            volatility_regime = self._analyze_volatility_regime(ohlcv_data)
            
            # 7. Генерация прогноза
            prediction = self._generate_prediction(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                fvg_signals=fvg_signals,
                orderflow_signals=orderflow_signals,
                liquidity_levels=liquidity_levels,
                snr_metrics=snr_metrics,
                market_structure=market_structure,
                volatility_regime=volatility_regime
            )
            
            logger.info(f"Generated prediction for {symbol}: {prediction.direction} "
                       f"(confidence: {prediction.confidence:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in advanced analysis for {symbol}: {e}")
            raise
    
    def _analyze_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """Анализ Fair Value Gaps (FVG)"""
        fvgs = []
        
        for i in range(2, len(data)):
            # Проверяем bullish FVG
            if (data.iloc[i-2]['high'] < data.iloc[i]['low'] and 
                data.iloc[i-1]['high'] < data.iloc[i]['low']):
                
                gap_size = data.iloc[i]['low'] - data.iloc[i-2]['high']
                if gap_size > self.min_fvg_size * data.iloc[i]['close']:
                    
                    # Проверяем подтверждение объемом
                    volume_confirmation = (data.iloc[i]['volume'] > 
                                         data.iloc[i-5:i]['volume'].mean() * 1.5)
                    
                    fvg = FairValueGap(
                        start_time=data.index[i-2],
                        end_time=data.index[i],
                        high=data.iloc[i]['low'],
                        low=data.iloc[i-2]['high'],
                        direction='bullish',
                        strength=min(1.0, gap_size / (data.iloc[i]['close'] * 0.005)),
                        volume_confirmation=volume_confirmation
                    )
                    fvgs.append(fvg)
            
            # Проверяем bearish FVG
            elif (data.iloc[i-2]['low'] > data.iloc[i]['high'] and 
                  data.iloc[i-1]['low'] > data.iloc[i]['high']):
                
                gap_size = data.iloc[i-2]['low'] - data.iloc[i]['high']
                if gap_size > self.min_fvg_size * data.iloc[i]['close']:
                    
                    volume_confirmation = (data.iloc[i]['volume'] > 
                                         data.iloc[i-5:i]['volume'].mean() * 1.5)
                    
                    fvg = FairValueGap(
                        start_time=data.index[i-2],
                        end_time=data.index[i],
                        high=data.iloc[i-2]['low'],
                        low=data.iloc[i]['high'],
                        direction='bearish',
                        strength=min(1.0, gap_size / (data.iloc[i]['close'] * 0.005)),
                        volume_confirmation=volume_confirmation
                    )
                    fvgs.append(fvg)
        
        # Обновляем историю и проверяем ретесты
        self._update_fvg_retests(fvgs, data)
        
        return fvgs[-5:]  # Возвращаем последние 5 FVG
    
    def _analyze_orderflow(
        self, 
        ohlcv_data: pd.DataFrame, 
        orderbook_data: Optional[pd.DataFrame]
    ) -> List[OrderFlowImbalance]:
        """Анализ дисбалансов ордерфлоу"""
        orderflow_signals = []
        
        if orderbook_data is None:
            # Эмулируем orderflow анализ на основе OHLCV
            orderflow_signals = self._estimate_orderflow_from_ohlcv(ohlcv_data)
        else:
            # Полный анализ orderflow с данными ордербука
            orderflow_signals = self._analyze_real_orderflow(orderbook_data)
        
        return orderflow_signals
    
    def _estimate_orderflow_from_ohlcv(self, data: pd.DataFrame) -> List[OrderFlowImbalance]:
        """Оценка orderflow на основе OHLCV данных"""
        signals = []
        
        for i in range(self.orderflow_window, len(data)):
            window_data = data.iloc[i-self.orderflow_window:i]
            
            # Эвристический анализ buying/selling pressure
            close_position = ((data.iloc[i]['close'] - data.iloc[i]['low']) / 
                            (data.iloc[i]['high'] - data.iloc[i]['low']))
            
            # Relative volume
            rel_volume = (data.iloc[i]['volume'] / 
                         window_data['volume'].mean())
            
            # Оценка buy/sell volumes
            if close_position > 0.7 and rel_volume > 1.2:
                buy_volume = data.iloc[i]['volume'] * close_position
                sell_volume = data.iloc[i]['volume'] * (1 - close_position)
            elif close_position < 0.3 and rel_volume > 1.2:
                sell_volume = data.iloc[i]['volume'] * (1 - close_position)
                buy_volume = data.iloc[i]['volume'] * close_position
            else:
                buy_volume = data.iloc[i]['volume'] * 0.5
                sell_volume = data.iloc[i]['volume'] * 0.5
            
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                imbalance_ratio = (buy_volume - sell_volume) / total_volume
                
                # Significance based on volume and price movement
                price_change = abs(data.iloc[i]['close'] - data.iloc[i]['open']) / data.iloc[i]['open']
                significance = min(1.0, (abs(imbalance_ratio) * rel_volume * price_change) / 0.1)
                
                if significance > 0.3:  # Минимальная значимость
                    signal = OrderFlowImbalance(
                        timestamp=data.index[i],
                        price_level=data.iloc[i]['close'],
                        buy_volume=buy_volume,
                        sell_volume=sell_volume,
                        imbalance_ratio=imbalance_ratio,
                        significance=significance
                    )
                    signals.append(signal)
        
        return signals[-10:]  # Последние 10 сигналов
    
    def _analyze_liquidity_levels(
        self, 
        data: pd.DataFrame,
        volume_profile: Optional[Dict[str, Any]]
    ) -> List[LiquidityLevel]:
        """Анализ уровней ликвидности"""
        levels = []
        
        # 1. Support/Resistance levels
        levels.extend(self._find_support_resistance_levels(data))
        
        # 2. Volume-based levels
        if volume_profile:
            levels.extend(self._find_volume_levels(data, volume_profile))
        
        # 3. POC (Point of Control)
        poc_levels = self._find_poc_levels(data)
        levels.extend(poc_levels)
        
        # Сортируем по значимости
        levels.sort(key=lambda x: x.significance, reverse=True)
        
        return levels[:15]  # Топ 15 уровней
    
    def _find_support_resistance_levels(self, data: pd.DataFrame) -> List[LiquidityLevel]:
        """Поиск уровней поддержки/сопротивления"""
        levels = []
        window = 20
        
        # Находим локальные максимумы и минимумы
        for i in range(window, len(data) - window):
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            
            # Проверка на локальный максимум (сопротивление)
            if (current_high == data.iloc[i-window:i+window]['high'].max()):
                touch_count = self._count_touches(data, current_high, 'resistance')
                if touch_count >= 2:
                    significance = min(1.0, touch_count / 5.0)
                    levels.append(LiquidityLevel(
                        price=current_high,
                        volume=data.iloc[i]['volume'],
                        strength=significance,
                        level_type='resistance',
                        touch_count=touch_count,
                        last_touch=data.index[i],
                        significance=significance
                    ))
            
            # Проверка на локальный минимум (поддержка)
            if (current_low == data.iloc[i-window:i+window]['low'].min()):
                touch_count = self._count_touches(data, current_low, 'support')
                if touch_count >= 2:
                    significance = min(1.0, touch_count / 5.0)
                    levels.append(LiquidityLevel(
                        price=current_low,
                        volume=data.iloc[i]['volume'],
                        strength=significance,
                        level_type='support',
                        touch_count=touch_count,
                        last_touch=data.index[i],
                        significance=significance
                    ))
        
        return levels
    
    def _calculate_snr_metrics(self, data: pd.DataFrame) -> SignalNoiseMetrics:
        """Расчет метрик Signal-to-Noise Ratio"""
        
        # Вычисляем signal (тренд) с помощью EMA
        signal = data['close'].ewm(span=self.snr_window).mean()
        
        # Noise - отклонение от сигнала
        noise = data['close'] - signal
        noise_power = (noise ** 2).mean()
        
        # Signal power
        signal_power = (signal.diff() ** 2).mean()
        
        # SNR ratio
        if noise_power > 0:
            snr_ratio = signal_power / noise_power
        else:
            snr_ratio = float('inf')
        
        # Signal strength (нормализованная)
        signal_strength = min(1.0, signal_power / (data['close'].std() ** 2))
        
        # Noise level
        noise_level = min(1.0, noise_power / (data['close'].std() ** 2))
        
        # Clarity score (чистота сигнала)
        clarity_score = 1.0 / (1.0 + np.exp(-snr_ratio + 1))
        
        # Overall confidence
        confidence = (signal_strength * clarity_score * min(1.0, snr_ratio / 5.0)) ** 0.5
        
        return SignalNoiseMetrics(
            snr_ratio=float(snr_ratio),
            signal_strength=float(signal_strength),
            noise_level=float(noise_level),
            clarity_score=float(clarity_score),
            confidence=float(confidence)
        )
    
    def _determine_market_structure(self, data: pd.DataFrame) -> str:
        """Определение структуры рынка"""
        
        # ADX для определения силы тренда
        adx = self._calculate_adx(data)
        
        # Volatility index
        volatility = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        recent_volatility = volatility.iloc[-10:].mean()
        
        # Trend strength analysis
        ema_short = data['close'].ewm(span=10).mean()
        ema_long = data['close'].ewm(span=30).mean()
        trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
        
        if adx > 25 and trend_strength > 0.02:
            return "trending"
        elif adx < 15 and recent_volatility < 0.02:
            return "ranging"
        else:
            return "transition"
    
    def _analyze_volatility_regime(self, data: pd.DataFrame) -> str:
        """Анализ режима волатильности"""
        
        # 20-период rolling volatility
        volatility = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        current_vol = volatility.iloc[-1]
        
        # Historical percentiles
        vol_percentile = (volatility <= current_vol).mean()
        
        if vol_percentile < 0.25:
            return "low"
        elif vol_percentile > 0.75:
            return "high"
        else:
            return "normal"
    
    def _generate_prediction(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        fvg_signals: List[FairValueGap],
        orderflow_signals: List[OrderFlowImbalance],
        liquidity_levels: List[LiquidityLevel],
        snr_metrics: SignalNoiseMetrics,
        market_structure: str,
        volatility_regime: str
    ) -> AdvancedPrediction:
        """Генерация итогового прогноза"""
        
        current_price = float(ohlcv_data['close'].iloc[-1])
        
        # Scoring system
        bullish_score = 0.0
        bearish_score = 0.0
        
        # 1. FVG Analysis
        for fvg in fvg_signals[-3:]:  # Последние 3 FVG
            if fvg.direction == 'bullish' and current_price > fvg.low:
                bullish_score += fvg.strength * 0.3
            elif fvg.direction == 'bearish' and current_price < fvg.high:
                bearish_score += fvg.strength * 0.3
        
        # 2. OrderFlow Analysis
        recent_orderflow = [of for of in orderflow_signals if of.significance > 0.5][-5:]
        for of in recent_orderflow:
            if of.is_bullish:
                bullish_score += of.significance * 0.25
            elif of.is_bearish:
                bearish_score += of.significance * 0.25
        
        # 3. Liquidity Levels
        nearby_levels = [lvl for lvl in liquidity_levels 
                        if abs(lvl.price - current_price) / current_price < 0.02]
        for level in nearby_levels[:3]:
            if level.level_type == 'support' and current_price > level.price:
                bullish_score += level.significance * 0.2
            elif level.level_type == 'resistance' and current_price < level.price:
                bearish_score += level.significance * 0.2
        
        # 4. SNR Quality Boost
        if snr_metrics.is_high_quality:
            quality_multiplier = 1.5
        else:
            quality_multiplier = 0.8
        
        bullish_score *= quality_multiplier
        bearish_score *= quality_multiplier
        
        # 5. Market Structure Adjustment
        if market_structure == "trending":
            # Boost the dominant direction
            if bullish_score > bearish_score:
                bullish_score *= 1.3
            else:
                bearish_score *= 1.3
        elif market_structure == "ranging":
            # Reduce confidence in strong directional moves
            bullish_score *= 0.8
            bearish_score *= 0.8
        
        # Final decision
        total_score = bullish_score + bearish_score
        if total_score == 0:
            direction = "neutral"
            confidence = 0.0
        elif bullish_score > bearish_score:
            direction = "buy" 
            confidence = min(0.95, bullish_score / max(1.0, total_score))
        else:
            direction = "sell"
            confidence = min(0.95, bearish_score / max(1.0, total_score))
        
        # Apply SNR confidence multiplier
        confidence *= snr_metrics.confidence
        
        # Calculate targets and stops
        atr = self._calculate_atr(ohlcv_data)
        
        if direction == "buy":
            target_price = current_price + (atr * 2.0)
            stop_loss = current_price - (atr * 1.0)
        elif direction == "sell":
            target_price = current_price - (atr * 2.0)
            stop_loss = current_price + (atr * 1.0)
        else:
            target_price = current_price
            stop_loss = current_price
        
        risk_reward_ratio = abs(target_price - current_price) / abs(stop_loss - current_price)
        
        # Expected duration based on market structure
        if market_structure == "trending":
            expected_duration = timedelta(hours=8)
        elif market_structure == "ranging":
            expected_duration = timedelta(hours=2)
        else:
            expected_duration = timedelta(hours=4)
        
        return AdvancedPrediction(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            confidence=float(confidence),
            target_price=float(target_price),
            stop_loss=float(stop_loss),
            timeframe="4H",
            fvg_signals=fvg_signals,
            orderflow_signals=orderflow_signals,
            liquidity_levels=liquidity_levels,
            snr_metrics=snr_metrics,
            risk_reward_ratio=float(risk_reward_ratio),
            expected_duration=expected_duration,
            market_structure=market_structure,
            volatility_regime=volatility_regime
        )
    
    def _update_fvg_retests(self, fvgs: List[FairValueGap], data: pd.DataFrame) -> None:
        """Обновление информации о ретестах FVG"""
        current_price = data['close'].iloc[-1]
        
        for fvg in fvgs:
            # Проверяем ретест FVG
            if fvg.low <= current_price <= fvg.high:
                fvg.retest_count += 1
                
            # Рассчитываем процент заполнения
            if fvg.direction == 'bullish':
                if current_price < fvg.mid_point:
                    fvg.filled_percentage = ((fvg.high - current_price) / fvg.size) * 100
            else:  # bearish
                if current_price > fvg.mid_point:
                    fvg.filled_percentage = ((current_price - fvg.low) / fvg.size) * 100
    
    def _count_touches(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Подсчет касаний уровня"""
        tolerance = level * 0.002  # 0.2% tolerance
        touches = 0
        
        if level_type == 'resistance':
            touches = int(((data['high'] >= level - tolerance) & 
                          (data['high'] <= level + tolerance)).sum())
        else:  # support
            touches = int(((data['low'] >= level - tolerance) & 
                          (data['low'] <= level + tolerance)).sum())
        
        return touches
    
    def _find_volume_levels(self, data: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[LiquidityLevel]:
        """Поиск уровней на основе объема"""
        levels = []
        
        if 'histogram' in volume_profile and 'bins' in volume_profile:
            volumes = volume_profile['histogram']
            bins = volume_profile['bins']
            
            # Находим POC (Point of Control) - уровень с максимальным объемом
            max_volume_idx = np.argmax(volumes)
            poc_price = (bins[max_volume_idx] + bins[max_volume_idx + 1]) / 2
            
            levels.append(LiquidityLevel(
                price=poc_price,
                volume=volumes[max_volume_idx],
                strength=1.0,
                level_type='poc',
                touch_count=5,  # POC обычно имеет много касаний
                last_touch=data.index[-1],
                significance=1.0
            ))
            
            # Добавляем другие значимые объемные уровни
            volume_threshold = np.percentile(volumes, 80)
            for i, volume in enumerate(volumes):
                if volume >= volume_threshold and i != max_volume_idx:
                    price = (bins[i] + bins[i + 1]) / 2
                    significance = volume / max(volumes)
                    
                    levels.append(LiquidityLevel(
                        price=price,
                        volume=volume,
                        strength=significance,
                        level_type='volume_node',
                        touch_count=3,
                        last_touch=data.index[-1],
                        significance=significance
                    ))
        
        return levels
    
    def _find_poc_levels(self, data: pd.DataFrame) -> List[LiquidityLevel]:
        """Поиск Points of Control"""
        levels = []
        
        # Простой POC на основе OHLCV
        # Создаем ценовые уровни и суммируем объемы
        price_levels = {}
        for _, row in data.iterrows():
            # Распределяем объем по OHLC
            for price in [row['open'], row['high'], row['low'], row['close']]:
                price_level = round(price, 4)  # Округляем до 4 знаков
                if price_level not in price_levels:
                    price_levels[price_level] = 0
                price_levels[price_level] += row['volume'] / 4
        
        # Находим уровни с максимальным объемом
        sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        
        for i, (price, volume) in enumerate(sorted_levels[:5]):  # Топ 5 уровней
            significance = volume / sorted_levels[0][1]  # Относительно максимального
            
            if significance > 0.6:  # Только значимые уровни
                levels.append(LiquidityLevel(
                    price=price,
                    volume=volume,
                    strength=significance,
                    level_type='poc',
                    touch_count=int(significance * 10),
                    last_touch=data.index[-1],
                    significance=significance
                ))
        
        return levels
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Расчет Average Directional Index"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = high.diff()
            dm_minus = low.diff() * -1
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Smoothed averages
            atr = tr.ewm(span=period).mean()
            di_plus = (dm_plus.ewm(span=period).mean() / atr) * 100
            di_minus = (dm_minus.ewm(span=period).mean() / atr) * 100
            
            # ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = dx.ewm(span=period).mean()
            
            return float(adx.iloc[-1])
            
        except Exception:
            return 20.0  # Default neutral value
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Расчет Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            atr_value = atr.iloc[-1]
            return float(atr_value) if not pd.isna(atr_value) else float(data['close'].iloc[-1] * 0.02)
            
        except Exception:
            return float(data['close'].iloc[-1] * 0.02)  # 2% fallback
    
    def _analyze_real_orderflow(self, orderbook_data: pd.DataFrame) -> List[OrderFlowImbalance]:
        """Анализ реального orderflow с данными ордербука"""
        signals = []
        
        # Группируем по ценовым уровням и времени
        for timestamp in orderbook_data['timestamp'].unique()[-self.orderflow_window:]:
            timestamp_data = orderbook_data[orderbook_data['timestamp'] == timestamp]
            
            # Анализируем bid/ask дисбаланс
            bid_volume = timestamp_data[timestamp_data['side'] == 'bid']['volume'].sum()
            ask_volume = timestamp_data[timestamp_data['side'] == 'ask']['volume'].sum()
            
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance_ratio = (bid_volume - ask_volume) / total_volume
                
                # Определяем значимость
                volume_threshold = timestamp_data['volume'].quantile(0.7)
                significance = min(1.0, total_volume / volume_threshold) * abs(imbalance_ratio)
                
                if significance > 0.4:
                    mid_price = timestamp_data['price'].mean()
                    
                    signal = OrderFlowImbalance(
                        timestamp=pd.to_datetime(timestamp),
                        price_level=mid_price,
                        buy_volume=bid_volume,
                        sell_volume=ask_volume,
                        imbalance_ratio=imbalance_ratio,
                        significance=significance
                    )
                    signals.append(signal)
        
        return signals