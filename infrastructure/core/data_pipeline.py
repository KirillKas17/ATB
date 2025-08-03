"""
Data Pipeline для обработки рыночных данных в ATB Trading System.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal

# Безопасный импорт pandas
try:
    import pandas as pd
    from pandas import DataFrame, Series
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Создаем заглушку DataFrame
    class DataFrameFallback:
        def __init__(self, data=None, **kwargs):
            self.data = data if data is not None else {}
            self._shape = (0, 0)
        
        @property
        def shape(self):
            return self._shape
        
        def dropna(self):
            return self
        
        def set_index(self, keys):
            return self
        
        def __getitem__(self, key):
            return self.data.get(key, [])
        
        def __setitem__(self, key, value):
            self.data[key] = value
    
    DataFrame = DataFrameFallback
    Series = list
    pd = type('PandasFallback', (), {'DataFrame': DataFrame})()

logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline для обработки рыночных данных."""
    
    def __init__(self):
        self.trade_history: Optional[Union[DataFrame, Dict]] = None
        self.processed_data: Dict[str, Any] = {}
    
    def clean_data(self, data: Union[DataFrame, Dict]) -> Union[DataFrame, Dict]:
        """Очистка данных от аномалий и пропущенных значений."""
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                # Удаление NaN значений
                cleaned_data = data.dropna()
                logger.info(f"Data cleaned: {data.shape[0]} -> {cleaned_data.shape[0]} rows")
                return cleaned_data
            else:
                # Fallback для случая без pandas
                if isinstance(data, dict):
                    cleaned_data = {}
                    for key, value in data.items():
                        if value is not None and value != "":
                            cleaned_data[key] = value
                    return cleaned_data
                return data
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data
    
    def validate_data(self, data: Union[DataFrame, Dict]) -> bool:
        """Валидация данных."""
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                return not data.empty and data.shape[0] > 0
            elif isinstance(data, dict):
                return len(data) > 0
            return False
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    def transform_data(self, data: Union[DataFrame, Dict]) -> Union[DataFrame, Dict]:
        """Трансформация данных."""
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                # Сортировка по времени если есть колонка timestamp
                if 'timestamp' in data.columns:
                    transformed_data = data.set_index('timestamp').sort_index()
                else:
                    transformed_data = data
                
                logger.info(f"Data transformed: {data.shape[0]} rows processed")
                return transformed_data
            else:
                # Fallback обработка
                if isinstance(data, dict) and 'timestamp' in data:
                    # Простая сортировка по timestamp
                    return dict(sorted(data.items(), key=lambda x: x[0]))
                return data
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return data
    
    def aggregate_data(self, data: Union[DataFrame, Dict], timeframe: str = "1h") -> Union[DataFrame, Dict]:
        """Агрегация данных по временным интервалам."""
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                # Агрегация OHLCV данных
                if 'price' in data.columns:
                    aggregated = data.resample(timeframe).agg({
                        'price': ['first', 'max', 'min', 'last'],
                        'volume': 'sum' if 'volume' in data.columns else 'count'
                    })
                    logger.info(f"Data aggregated to {timeframe} timeframe")
                    return aggregated
                return data
            else:
                # Fallback агрегация
                logger.warning("Aggregation not available without pandas")
                return data
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return data
    
    def calculate_indicators(self, data: Union[DataFrame, Dict]) -> Dict[str, Any]:
        """Расчет технических индикаторов."""
        indicators = {}
        
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame) and 'price' in data.columns:
                prices = data['price'].astype(float)
                
                # Простые скользящие средние
                indicators['sma_20'] = prices.rolling(20).mean().iloc[-1] if len(prices) >= 20 else None
                indicators['sma_50'] = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else None
                
                # Волатильность
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    indicators['volatility'] = returns.std() if len(returns) > 0 else 0.0
                
            elif isinstance(data, dict) and 'price' in data:
                # Fallback расчеты
                prices = [float(p) for p in data['price'] if p is not None]
                if prices:
                    indicators['current_price'] = prices[-1]
                    indicators['min_price'] = min(prices)
                    indicators['max_price'] = max(prices)
                    indicators['avg_price'] = sum(prices) / len(prices)
                    
                    if len(prices) > 1:
                        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                        variance = sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)
                        indicators['volatility'] = variance ** 0.5
            
            logger.info(f"Calculated {len(indicators)} indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def process_trade_data(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Обработка торговых данных."""
        if not trades:
            return {}
        
        try:
            # Определение типа trade_history
            if PANDAS_AVAILABLE:
                self.trade_history = pd.DataFrame(trades)
            else:
                self.trade_history = {'trades': trades}
            
            # Расчет базовых метрик
            total_volume = sum(float(trade.get('volume', 0)) for trade in trades)
            avg_price = sum(float(trade.get('price', 0)) for trade in trades) / len(trades)
            
            # Извлечение данных для анализа
            if isinstance(self.trade_history, dict):
                trade_data = self.trade_history
            else:
                trade_data = {'trades': trades}
            
            processed_data = {
                'total_trades': len(trades),
                'total_volume': total_volume,
                'average_price': avg_price,
                'trade_history': trade_data
            }
            
            self.processed_data.update(processed_data)
            logger.info(f"Processed {len(trades)} trades")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            return {}
    
    def get_processed_data(self) -> Dict[str, Any]:
        """Получение обработанных данных."""
        return self.processed_data
    
    def clear_cache(self) -> None:
        """Очистка кэша обработанных данных."""
        self.processed_data.clear()
        self.trade_history = None
        logger.info("Data pipeline cache cleared")
