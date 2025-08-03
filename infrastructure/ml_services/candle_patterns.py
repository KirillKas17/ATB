"""
Анализ свечных паттернов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pandas import Series, DataFrame

class CandlePatternAnalyzer:
    """Анализатор свечных паттернов."""

    def __init__(self) -> None:
        self.patterns = {
            "doji": self._detect_doji,
            "hammer": self._detect_hammer,
            "shooting_star": self._detect_shooting_star,
            "engulfing": self._detect_engulfing,
            "morning_star": self._detect_morning_star,
            "evening_star": self._detect_evening_star,
            "three_white_soldiers": self._detect_three_white_soldiers,
            "three_black_crows": self._detect_three_black_crows,
        }

    def detect_patterns(self, data: DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Обнаружение всех паттернов в данных."""
        if len(data) < 3:
            return {}
        
        results: Dict[str, List[Dict[str, Any]]] = {}
        for pattern_name, pattern_func in self.patterns.items():
            pattern_results = pattern_func(data)
            if pattern_results:
                results[pattern_name] = pattern_results
        
        return results

    def _detect_doji(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Doji."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return []
        
        doji_patterns = []
        
        # Проверяем последние 20 свечей
        if len(data) >= 20:
            recent_data = data.iloc[-20:]
        else:
            recent_data = data
        
        for i in range(len(recent_data)):
            if not isinstance(recent_data, pd.DataFrame) or len(recent_data) == 0:
                continue
            row = recent_data.iloc[i]
            body_size = abs(float(row['close']) - float(row['open']))
            total_range = float(row['high']) - float(row['low'])
            
            # Doji: тело свечи составляет менее 10% от общего диапазона
            if total_range > 0 and body_size / total_range < 0.1:
                doji_patterns.append({
                    'index': i,
                    'timestamp': row.name if hasattr(row, 'name') else None,
                    'confidence': 0.8,
                    'type': 'neutral'
                })
        
        return doji_patterns

    def _detect_hammer(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Hammer."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return []
        
        hammer_patterns = []
        
        # Проверяем последние 20 свечей
        if len(data) >= 20:
            recent_data = data.iloc[-20:]
        else:
            recent_data = data
        
        for i in range(len(recent_data)):
            if not isinstance(recent_data, pd.DataFrame) or len(recent_data) == 0:
                continue
            row = recent_data.iloc[i]
            body_size = abs(float(row['close']) - float(row['open']))
            lower_shadow = min(float(row['open']), float(row['close'])) - float(row['low'])
            upper_shadow = float(row['high']) - max(float(row['open']), float(row['close']))
            
            # Hammer: длинная нижняя тень, короткая верхняя тень
            if (lower_shadow > 2 * body_size and 
                upper_shadow < body_size * 0.5):
                hammer_patterns.append({
                    'index': i,
                    'timestamp': row.name if hasattr(row, 'name') else None,
                    'confidence': 0.7,
                    'type': 'bullish'
                })
        
        return hammer_patterns

    def _detect_shooting_star(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Shooting Star."""
        if len(data) < 1:
            return []
        
        shooting_star_patterns = []
        
        # Получаем последние 20 свечей
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(len(recent_data)):
            if len(recent_data) == 0:
                continue
            row = recent_data.iloc[i]
            body_size = abs(float(row['close']) - float(row['open']))
            lower_shadow = min(float(row['open']), float(row['close'])) - float(row['low'])
            upper_shadow = float(row['high']) - max(float(row['open']), float(row['close']))
            
            # Shooting Star: длинная верхняя тень, короткая нижняя тень
            if (upper_shadow > 2 * body_size and 
                lower_shadow < body_size * 0.5):
                shooting_star_patterns.append({
                    'index': i,
                    'timestamp': row.name if hasattr(row, 'name') else None,
                    'confidence': 0.7,
                    'type': 'bearish'
                })
        
        return shooting_star_patterns

    def _detect_engulfing(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Engulfing."""
        if len(data) < 2:
            return []
        
        engulfing_patterns = []
        
        # Получаем последние 20 свечей
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(1, len(recent_data)):
            if len(recent_data) > 1:
                prev_row = recent_data.iloc[i-1]
                curr_row = recent_data.iloc[i]
            else:
                continue
            
            prev_body_size = abs(float(prev_row['close']) - float(prev_row['open']))
            curr_body_size = abs(float(curr_row['close']) - float(curr_row['open']))
            
            # Bullish Engulfing
            if (float(prev_row['close']) < float(prev_row['open']) and  # Предыдущая свеча красная
                float(curr_row['close']) > float(curr_row['open']) and  # Текущая свеча зеленая
                float(curr_row['open']) < float(prev_row['close']) and  # Текущая свеча открывается ниже закрытия предыдущей
                float(curr_row['close']) > float(prev_row['open']) and  # Текущая свеча закрывается выше открытия предыдущей
                curr_body_size > prev_body_size):  # Текущая свеча больше предыдущей
                
                engulfing_patterns.append({
                    'index': i,
                    'timestamp': curr_row.name if hasattr(curr_row, 'name') else None,
                    'confidence': 0.8,
                    'type': 'bullish'
                })
            
            # Bearish Engulfing
            elif (float(prev_row['close']) > float(prev_row['open']) and  # Предыдущая свеча зеленая
                  float(curr_row['close']) < float(curr_row['open']) and  # Текущая свеча красная
                  float(curr_row['open']) > float(prev_row['close']) and  # Текущая свеча открывается выше закрытия предыдущей
                  float(curr_row['close']) < float(prev_row['open']) and  # Текущая свеча закрывается ниже открытия предыдущей
                  curr_body_size > prev_body_size):  # Текущая свеча больше предыдущей
                
                engulfing_patterns.append({
                    'index': i,
                    'timestamp': curr_row.name if hasattr(curr_row, 'name') else None,
                    'confidence': 0.8,
                    'type': 'bearish'
                })
        
        return engulfing_patterns

    def _detect_morning_star(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Morning Star."""
        if len(data) < 3:
            return []
        
        morning_star_patterns = []
        
        # Получаем последние 20 свечей
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(2, len(recent_data)):
            if len(recent_data) > 2:
                first_row = recent_data.iloc[i-2]
                second_row = recent_data.iloc[i-1]
                third_row = recent_data.iloc[i]
            else:
                continue
            
            # Morning Star: большая красная свеча, маленькая свеча, большая зеленая свеча
            if (float(first_row['close']) < float(first_row['open']) and  # Первая свеча красная
                abs(float(second_row['close']) - float(second_row['open'])) < abs(float(first_row['close']) - float(first_row['open'])) * 0.3 and  # Вторая свеча маленькая
                float(third_row['close']) > float(third_row['open']) and  # Третья свеча зеленая
                float(third_row['close']) > (float(first_row['open']) + float(first_row['close'])) / 2):  # Третья свеча закрывается выше середины первой
                
                morning_star_patterns.append({
                    'index': i,
                    'timestamp': third_row.name if hasattr(third_row, 'name') else None,
                    'confidence': 0.9,
                    'type': 'bullish'
                })
        
        return morning_star_patterns

    def _detect_evening_star(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Evening Star."""
        if len(data) < 3:
            return []
        
        evening_star_patterns = []
        
        # Получаем последние 20 свечи
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(2, len(recent_data)):
            if len(recent_data) > 2:
                first_row = recent_data.iloc[i-2]
                second_row = recent_data.iloc[i-1]
                third_row = recent_data.iloc[i]
            else:
                continue
            
            # Evening Star: большая зеленая свеча, маленькая свеча, большая красная свеча
            if (float(first_row['close']) > float(first_row['open']) and  # Первая свеча зеленая
                abs(float(second_row['close']) - float(second_row['open'])) < abs(float(first_row['close']) - float(first_row['open'])) * 0.3 and  # Вторая свеча маленькая
                float(third_row['close']) < float(third_row['open']) and  # Третья свеча красная
                float(third_row['close']) < (float(first_row['open']) + float(first_row['close'])) / 2):  # Третья свеча закрывается ниже середины первой
                
                evening_star_patterns.append({
                    'index': i,
                    'timestamp': third_row.name if hasattr(third_row, 'name') else None,
                    'confidence': 0.9,
                    'type': 'bearish'
                })
        
        return evening_star_patterns

    def _detect_three_white_soldiers(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Three White Soldiers."""
        if len(data) < 3:
            return []
        
        three_white_soldiers_patterns = []
        
        # Получаем последние 20 свечей
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(2, len(recent_data)):
            if len(recent_data) > 2:
                first_row = recent_data.iloc[i-2]
                second_row = recent_data.iloc[i-1]
                third_row = recent_data.iloc[i]
            else:
                continue
            
            # Three White Soldiers: три последовательные зеленые свечи
            if (float(first_row['close']) > float(first_row['open']) and
                float(second_row['close']) > float(second_row['open']) and
                float(third_row['close']) > float(third_row['open']) and
                float(second_row['open']) > float(first_row['open']) and
                float(third_row['open']) > float(second_row['open'])):
                
                three_white_soldiers_patterns.append({
                    'index': i,
                    'timestamp': third_row.name if hasattr(third_row, 'name') else None,
                    'confidence': 0.8,
                    'type': 'bullish'
                })
        
        return three_white_soldiers_patterns

    def _detect_three_black_crows(self, data: DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттерна Three Black Crows."""
        if len(data) < 3:
            return []
        
        three_black_crows_patterns = []
        
        # Получаем последние 20 свечей
        recent_data = data.iloc[-20:] if len(data) >= 20 else data
        
        for i in range(2, len(recent_data)):
            if len(recent_data) > 2:
                first_row = recent_data.iloc[i-2]
                second_row = recent_data.iloc[i-1]
                third_row = recent_data.iloc[i]
            else:
                continue
            
            # Three Black Crows: три последовательные красные свечи
            if (float(first_row['close']) < float(first_row['open']) and
                float(second_row['close']) < float(second_row['open']) and
                float(third_row['close']) < float(third_row['open']) and
                float(second_row['open']) < float(first_row['open']) and
                float(third_row['open']) < float(second_row['open'])):
                
                three_black_crows_patterns.append({
                    'index': i,
                    'timestamp': third_row.name if hasattr(third_row, 'name') else None,
                    'confidence': 0.8,
                    'type': 'bearish'
                })
        
        return three_black_crows_patterns
