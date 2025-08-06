"""
Универсальный валидатор торговых сигналов.
Обеспечивает безопасность всех торговых операций.
"""

from typing import Any, Optional, Union
import pandas as pd
from loguru import logger


class UniversalSignalValidator:
    """Универсальный валидатор торговых сигналов для предотвращения потерь"""
    
    def __init__(self) -> None:
        # Максимальные допустимые риски
        self.max_stop_loss_percent = 0.10  # Максимум 10% стоп-лосс
        self.min_risk_reward_ratio = 0.5   # Минимальное соотношение прибыль/риск
        self.max_take_profit_percent = 0.50  # Максимум 50% тейк-профит
        
    def validate_signal(self, signal: Any) -> bool:
        """
        Критическая валидация торгового сигнала.
        
        Args:
            signal: Торговый сигнал любого типа
            
        Returns:
            bool: True если сигнал безопасен для торговли
        """
        try:
            # 1. Проверка базовых атрибутов
            if not self._validate_basic_attributes(signal):
                return False
                
            # 2. Проверка цены входа
            if not self._validate_entry_price(signal):
                return False
                
            # 3. Проверка стоп-лосса
            if not self._validate_stop_loss(signal):
                return False
                
            # 4. Проверка тейк-профита
            if not self._validate_take_profit(signal):
                return False
                
            # 5. Проверка направления и логики
            if not self._validate_direction_logic(signal):
                return False
                
            # 6. Проверка соотношения риск/прибыль
            if not self._validate_risk_reward_ratio(signal):
                return False
                
            # 7. Проверка размера позиции
            if not self._validate_position_size(signal):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Critical error in signal validation: {str(e)}")
            return False  # В случае ошибки отклоняем сигнал
            
    def _validate_basic_attributes(self, signal: Any) -> bool:
        """Проверка базовых атрибутов сигнала"""
        try:
            # Проверка наличия направления
            if not hasattr(signal, 'direction') or not signal.direction:
                logger.warning("Signal missing direction")
                return False
                
            # Проверка допустимых направлений
            if signal.direction not in ["long", "short", "buy", "sell", "close"]:
                logger.warning(f"Invalid signal direction: {signal.direction}")
                return False
                
            return True
        except Exception:
            return False
            
    def _validate_entry_price(self, signal: Any) -> bool:
        """Проверка цены входа"""
        try:
            if not hasattr(signal, 'entry_price'):
                logger.warning("Signal missing entry_price")
                return False
                
            entry_price = float(signal.entry_price)
            
            if entry_price <= 0:
                logger.warning(f"Invalid entry_price: {entry_price}")
                return False
                
            # Проверка разумности цены (не может быть слишком большой или маленькой)
            if entry_price > 1_000_000 or entry_price < 0.0001:
                logger.warning(f"Unrealistic entry_price: {entry_price}")
                return False
                
            return True
        except (ValueError, TypeError):
            logger.warning("Invalid entry_price format")
            return False
            
    def _validate_stop_loss(self, signal: Any) -> bool:
        """Проверка стоп-лосса"""
        try:
            if not hasattr(signal, 'stop_loss') or signal.stop_loss is None:
                logger.warning("Signal missing stop_loss")
                return False
                
            stop_loss = float(signal.stop_loss)
            entry_price = float(signal.entry_price)
            
            if stop_loss <= 0:
                logger.warning(f"Invalid stop_loss: {stop_loss}")
                return False
                
            # Проверка логики стоп-лосса относительно направления
            if signal.direction in ["long", "buy"]:
                if stop_loss >= entry_price:
                    logger.warning(f"Long signal: stop_loss ({stop_loss}) >= entry_price ({entry_price})")
                    return False
            elif signal.direction in ["short", "sell"]:
                if stop_loss <= entry_price:
                    logger.warning(f"Short signal: stop_loss ({stop_loss}) <= entry_price ({entry_price})")
                    return False
                    
            # Проверка максимального размера стоп-лосса
            stop_distance = abs(stop_loss - entry_price)
            stop_percent = stop_distance / entry_price
            
            if stop_percent > self.max_stop_loss_percent:
                logger.warning(f"Stop loss too wide: {stop_percent:.2%} > {self.max_stop_loss_percent:.2%}")
                return False
                
            return True
        except (ValueError, TypeError):
            logger.warning("Invalid stop_loss format")
            return False
            
    def _validate_take_profit(self, signal: Any) -> bool:
        """Проверка тейк-профита"""
        try:
            if not hasattr(signal, 'take_profit') or signal.take_profit is None:
                logger.warning("Signal missing take_profit")
                return False
                
            take_profit = float(signal.take_profit)
            entry_price = float(signal.entry_price)
            
            if take_profit <= 0:
                logger.warning(f"Invalid take_profit: {take_profit}")
                return False
                
            # Проверка логики тейк-профита относительно направления
            if signal.direction in ["long", "buy"]:
                if take_profit <= entry_price:
                    logger.warning(f"Long signal: take_profit ({take_profit}) <= entry_price ({entry_price})")
                    return False
            elif signal.direction in ["short", "sell"]:
                if take_profit >= entry_price:
                    logger.warning(f"Short signal: take_profit ({take_profit}) >= entry_price ({entry_price})")
                    return False
                    
            # Проверка максимального размера тейк-профита
            profit_distance = abs(take_profit - entry_price)
            profit_percent = profit_distance / entry_price
            
            if profit_percent > self.max_take_profit_percent:
                logger.warning(f"Take profit too wide: {profit_percent:.2%} > {self.max_take_profit_percent:.2%}")
                return False
                
            return True
        except (ValueError, TypeError):
            logger.warning("Invalid take_profit format")
            return False
            
    def _validate_direction_logic(self, signal: Any) -> bool:
        """Проверка общей логики направления"""
        try:
            # Для сигналов закрытия позиции менее строгие требования
            if signal.direction == "close":
                return True
                
            # Остальные проверки уже выполнены в _validate_stop_loss и _validate_take_profit
            return True
        except Exception:
            return False
            
    def _validate_risk_reward_ratio(self, signal: Any) -> bool:
        """Проверка соотношения риск/прибыль"""
        try:
            if signal.direction == "close":
                return True  # Для закрытия позиции не проверяем R/R
                
            entry_price = float(signal.entry_price)
            stop_loss = float(signal.stop_loss)
            take_profit = float(signal.take_profit)
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk <= 0:
                logger.warning("Zero or negative risk")
                return False
                
            risk_reward_ratio = reward / risk
            
            if risk_reward_ratio < self.min_risk_reward_ratio:
                logger.warning(f"Poor risk/reward ratio: {risk_reward_ratio:.2f} < {self.min_risk_reward_ratio}")
                return False
                
            return True
        except (ValueError, TypeError, ZeroDivisionError):
            logger.warning("Error calculating risk/reward ratio")
            return False
            
    def _validate_position_size(self, signal: Any) -> bool:
        """Проверка размера позиции"""
        try:
            if hasattr(signal, 'volume') and signal.volume is not None:
                volume = float(signal.volume)
                if volume <= 0:
                    logger.warning(f"Invalid volume: {volume}")
                    return False
                    
                # Проверка разумности размера позиции
                if volume > 1_000_000:  # Максимум 1M единиц
                    logger.warning(f"Position size too large: {volume}")
                    return False
                    
            return True
        except (ValueError, TypeError):
            logger.warning("Invalid volume format")
            return False
            
    def get_safe_fallback_signal(self, entry_price: float, direction: str) -> dict:
        """
        Создание безопасного fallback сигнала с консервативными параметрами.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки
            
        Returns:
            dict: Безопасный сигнал
        """
        try:
            if direction in ["long", "buy"]:
                stop_loss = entry_price * 0.98  # 2% стоп-лосс
                take_profit = entry_price * 1.04  # 4% тейк-профит (R/R = 2:1)
            elif direction in ["short", "sell"]:
                stop_loss = entry_price * 1.02  # 2% стоп-лосс
                take_profit = entry_price * 0.96  # 4% тейк-профит (R/R = 2:1)
            else:
                raise ValueError(f"Invalid direction: {direction}")
                
            return {
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 0.5,  # Низкая уверенность для fallback
                'volume': 1.0
            }
        except Exception as e:
            logger.error(f"Error creating fallback signal: {str(e)}")
            return None


# Глобальный экземпляр валидатора
signal_validator = UniversalSignalValidator()


def validate_trading_signal(signal: Any) -> bool:
    """
    Удобная функция для валидации торговых сигналов.
    
    Args:
        signal: Торговый сигнал
        
    Returns:
        bool: True если сигнал безопасен
    """
    return signal_validator.validate_signal(signal)


def get_safe_price(price_series: pd.Series, index: int, series_name: str = "price") -> float:
    """
    Универсальная функция для безопасного получения цены.
    
    Args:
        price_series: Серия с ценами
        index: Индекс цены
        series_name: Название серии для логирования
        
    Returns:
        float: Безопасная цена
        
    Raises:
        ValueError: Если невозможно получить корректную цену
    """
    try:
        # Проверка основной цены
        if 0 <= index < len(price_series):
            price = price_series.iloc[index]
            if price is not None and not pd.isna(price) and price > 0:
                return float(price)
        
        # Поиск ближайшей корректной цены (в пределах 10 элементов)
        for offset in range(1, min(11, len(price_series))):
            # Проверяем назад
            if index - offset >= 0:
                price = price_series.iloc[index - offset]
                if price is not None and not pd.isna(price) and price > 0:
                    logger.warning(f"Using {series_name} from offset -{offset}: {price}")
                    return float(price)
            
            # Проверяем вперед
            if index + offset < len(price_series):
                price = price_series.iloc[index + offset]
                if price is not None and not pd.isna(price) and price > 0:
                    logger.warning(f"Using {series_name} from offset +{offset}: {price}")
                    return float(price)
        
        # Используем среднюю цену как последний resort
        valid_prices = price_series.dropna()
        valid_prices = valid_prices[valid_prices > 0]
        
        if len(valid_prices) > 0:
            avg_price = valid_prices.mean()
            logger.warning(f"Using average {series_name}: {avg_price}")
            return float(avg_price)
        
        # Если ничего не помогло - ошибка
        raise ValueError(f"Cannot find valid {series_name} at index {index}")
        
    except Exception as e:
        logger.error(f"Critical error getting {series_name} at index {index}: {str(e)}")
        raise ValueError(f"Invalid {series_name} data at index {index}")