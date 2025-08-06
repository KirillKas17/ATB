"""
Класс для управления адаптивными порогами торговых стратегий.
Адаптирует пороги на основе характеристик актива, волатильности и рыночных условий.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from loguru import logger


class AdaptiveThresholds:
    """Управление адаптивными порогами для торговых стратегий"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация адаптивных порогов.
        
        Args:
            config: Конфигурация с базовыми параметрами
        """
        self.config = config or {}
        
        # Базовые пороги для нормализации
        self.base_price_level = self.config.get('base_price_level', 100.0)
        
        # Базовые пороги RSI
        self.rsi_oversold_base = self.config.get('rsi_oversold_base', 30)
        self.rsi_overbought_base = self.config.get('rsi_overbought_base', 70)
        
        # Базовые пороги волатильности
        self.volatility_low_threshold = self.config.get('volatility_low_threshold', 0.01)
        self.volatility_high_threshold = self.config.get('volatility_high_threshold', 0.03)
        
        # Базовые пороги тренда
        self.trend_strength_threshold = self.config.get('trend_strength_threshold', 0.02)
        
        # Пороги для объема
        self.volume_normal_threshold = self.config.get('volume_normal_threshold', 1.0)
        self.volume_high_threshold = self.config.get('volume_high_threshold', 1.5)
        
    def get_price_level_factor(self, data: pd.DataFrame) -> float:
        """
        Получение коэффициента нормализации на основе уровня цен актива.
        
        Args:
            data: DataFrame с ценовыми данными
            
        Returns:
            float: Коэффициент нормализации (обычно от 0.1 до 10.0)
        """
        try:
            avg_price = float(data["close"].mean())
            price_level_factor = max(0.1, min(10.0, avg_price / self.base_price_level))
            return float(price_level_factor)
        except Exception as e:
            logger.error(f"Error calculating price level factor: {str(e)}")
            return 1.0
    
    def get_volatility_regime(self, data: pd.DataFrame, window: int = 20) -> str:
        """
        Определение режима волатильности.
        
        Args:
            data: DataFrame с ценовыми данными
            window: Окно для расчета волатильности
            
        Returns:
            str: 'low', 'normal', 'high'
        """
        try:
            if len(data) < window:
                return 'normal'
                
            volatility = data["close"].pct_change().rolling(window).std().iloc[-1]
            volatility = float(volatility) if not pd.isna(volatility) else 0.02
            
            price_level_factor = self.get_price_level_factor(data)
            low_threshold = self.volatility_low_threshold * price_level_factor
            high_threshold = self.volatility_high_threshold * price_level_factor
            
            if volatility < low_threshold:
                return 'low'
            elif volatility > high_threshold:
                return 'high'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error determining volatility regime: {str(e)}")
            return 'normal'
    
    def get_adaptive_rsi_thresholds(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Получение адаптивных порогов RSI на основе рыночных условий.
        
        Args:
            data: DataFrame с рыночными данными
            
        Returns:
            Tuple[float, float]: (oversold_threshold, overbought_threshold)
        """
        try:
            volatility_regime = self.get_volatility_regime(data)
            
            if volatility_regime == 'high':
                # В высоковолатильных условиях делаем пороги более строгими
                oversold = self.rsi_oversold_base - 5   # 25 вместо 30
                overbought = self.rsi_overbought_base + 5  # 75 вместо 70
            elif volatility_regime == 'low':
                # В низковолатильных условиях делаем пороги менее строгими
                oversold = self.rsi_oversold_base + 10  # 40 вместо 30
                overbought = self.rsi_overbought_base - 10  # 60 вместо 70
            else:
                # Нормальные условия
                oversold = self.rsi_oversold_base
                overbought = self.rsi_overbought_base
                
            return oversold, overbought
            
        except Exception as e:
            logger.error(f"Error calculating adaptive RSI thresholds: {str(e)}")
            return self.rsi_oversold_base, self.rsi_overbought_base
    
    def get_adaptive_trend_threshold(self, data: pd.DataFrame) -> float:
        """
        Получение адаптивного порога для определения силы тренда.
        
        Args:
            data: DataFrame с рыночными данными
            
        Returns:
            float: Пороговое значение для силы тренда
        """
        try:
            price_level_factor = self.get_price_level_factor(data)
            volatility_regime = self.get_volatility_regime(data)
            
            base_threshold = self.trend_strength_threshold * price_level_factor
            
            if volatility_regime == 'high':
                # В высоковолатильных условиях требуем более сильный тренд
                return float(base_threshold * 1.5)
            elif volatility_regime == 'low':
                # В низковолатильных условиях достаточно слабого тренда
                return float(base_threshold * 0.7)
            else:
                return float(base_threshold)
                
        except Exception as e:
            logger.error(f"Error calculating adaptive trend threshold: {str(e)}")
            return float(self.trend_strength_threshold)
    
    def get_adaptive_volume_thresholds(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Получение адаптивных порогов для анализа объема.
        
        Args:
            data: DataFrame с данными об объеме
            
        Returns:
            Tuple[float, float]: (normal_threshold, high_threshold)
        """
        try:
            if len(data) < 20 or "volume" not in data.columns:
                return self.volume_normal_threshold, self.volume_high_threshold
                
            # Анализируем историческое распределение объемов
            volume_std = data["volume"].rolling(50).std().iloc[-1]
            volume_mean = data["volume"].rolling(50).mean().iloc[-1]
            
            if volume_mean > 0:
                cv = volume_std / volume_mean  # Коэффициент вариации
                
                if cv > 1.0:  # Высокая изменчивость объемов
                    normal_threshold = self.volume_normal_threshold * 0.8
                    high_threshold = self.volume_high_threshold * 0.8
                elif cv < 0.3:  # Низкая изменчивость объемов
                    normal_threshold = self.volume_normal_threshold * 1.2
                    high_threshold = self.volume_high_threshold * 1.2
                else:
                    normal_threshold = self.volume_normal_threshold
                    high_threshold = self.volume_high_threshold
                    
                return normal_threshold, high_threshold
            else:
                return self.volume_normal_threshold, self.volume_high_threshold
                
        except Exception as e:
            logger.error(f"Error calculating adaptive volume thresholds: {str(e)}")
            return self.volume_normal_threshold, self.volume_high_threshold
    
    def get_adaptive_bollinger_tolerance(self, data: pd.DataFrame) -> float:
        """
        Получение адаптивной толерантности для касания Bollinger Bands.
        
        Args:
            data: DataFrame с ценовыми данными
            
        Returns:
            float: Толерантность в процентах (например, 0.005 = 0.5%)
        """
        try:
            volatility_regime = self.get_volatility_regime(data)
            price_level_factor = self.get_price_level_factor(data)
            
            base_tolerance = 0.005 * price_level_factor
            
            if volatility_regime == 'high':
                # В высоковолатильных условиях увеличиваем толерантность
                return base_tolerance * 2.0
            elif volatility_regime == 'low':
                # В низковолатильных условиях уменьшаем толерантность
                return base_tolerance * 0.5
            else:
                return base_tolerance
                
        except Exception as e:
            logger.error(f"Error calculating adaptive Bollinger tolerance: {str(e)}")
            return 0.005
    
    def get_market_state_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Получение сводки о текущем состоянии рынка и адаптивных порогах.
        
        Args:
            data: DataFrame с рыночными данными
            
        Returns:
            Dict: Сводка состояния рынка
        """
        try:
            price_level_factor = self.get_price_level_factor(data)
            volatility_regime = self.get_volatility_regime(data)
            rsi_oversold, rsi_overbought = self.get_adaptive_rsi_thresholds(data)
            trend_threshold = self.get_adaptive_trend_threshold(data)
            volume_normal, volume_high = self.get_adaptive_volume_thresholds(data)
            bb_tolerance = self.get_adaptive_bollinger_tolerance(data)
            
            return {
                'price_level_factor': price_level_factor,
                'volatility_regime': volatility_regime,
                'rsi_thresholds': {
                    'oversold': rsi_oversold,
                    'overbought': rsi_overbought
                },
                'trend_threshold': trend_threshold,
                'volume_thresholds': {
                    'normal': volume_normal,
                    'high': volume_high
                },
                'bollinger_tolerance': bb_tolerance
            }
            
        except Exception as e:
            logger.error(f"Error generating market state summary: {str(e)}")
            return {}