# -*- coding: utf-8 -*-
"""Валидаторы для модуля symbols."""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd
from loguru import logger

from domain.types import (
    DataInsufficientError,
    MarketDataError,
    MarketDataFrame,
    OrderBookData,
    OrderBookError,
    PatternMemoryData,
    SessionData,
    ValidationError,
)


@runtime_checkable
class MarketDataValidator(Protocol):
    """Протокол для валидатора рыночных данных."""

    def validate_ohlcv_data(self, data: MarketDataFrame) -> bool:
        """Валидация OHLCV данных."""
        ...

    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """Валидация стакана заявок."""
        ...

    def validate_pattern_memory(self, pattern_memory: Optional[PatternMemoryData]) -> bool:
        """Валидация памяти паттернов."""
        ...


class SymbolDataValidator:
    """Валидатор данных для торговых символов."""

    def __init__(self) -> None:
        """Инициализация валидатора."""
        self.logger = logger.bind(name=self.__class__.__name__)

    def validate_ohlcv_data(self, data: MarketDataFrame) -> bool:
        """Валидация OHLCV данных."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValidationError("Market data must be a pandas DataFrame")
            # Проверка обязательных колонок
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            # Проверка на пустые данные
            if len(data) == 0:
                raise DataInsufficientError("Market data is empty")
            # Проверка на отрицательные значения
            for col in ["open", "high", "low", "close", "volume"]:
                if (data[col] < 0).any():
                    raise ValidationError(f"Negative values found in {col} column")
            # Проверка логики OHLC
            invalid_high = (data["high"] < data[["open", "close"]].max(axis=1)).any()
            invalid_low = (data["low"] > data[["open", "close"]].min(axis=1)).any()
            if invalid_high:
                raise ValidationError("High price cannot be lower than open or close")
            if invalid_low:
                raise ValidationError("Low price cannot be higher than open or close")
            # Проверка на NaN значения
            if data[required_columns].isnull().any().any():
                raise ValidationError("NaN values found in OHLCV data")
            # Проверка минимального количества данных
            if len(data) < 20:
                raise DataInsufficientError(
                    "Insufficient data points (minimum 20 required)"
                )
            return True
        except Exception as e:
            self.logger.error(f"Error validating OHLCV data: {e}")
            raise

    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """Валидация стакана заявок."""
        try:
            if not isinstance(order_book, dict):
                raise ValidationError("Order book must be a dictionary")
            # Проверка обязательных ключей
            required_keys = ["bids", "asks"]
            missing_keys = [key for key in required_keys if key not in order_book]
            if missing_keys:
                raise ValidationError(f"Missing required keys: {missing_keys}")
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            # Проверка типов данных
            if not isinstance(bids, list) or not isinstance(asks, list):
                raise ValidationError("Bids and asks must be lists")
            # Проверка структуры данных
            for bid in bids:
                if not isinstance(bid, (list, tuple)) or len(bid) < 2:
                    raise ValidationError(
                        "Each bid must be a list/tuple with at least 2 elements [price, volume]"
                    )
                if not isinstance(bid[0], (int, float)) or not isinstance(
                    bid[1], (int, float)
                ):
                    raise ValidationError("Bid price and volume must be numbers")
                if bid[0] <= 0 or bid[1] <= 0:
                    raise ValidationError("Bid price and volume must be positive")
            for ask in asks:
                if not isinstance(ask, (list, tuple)) or len(ask) < 2:
                    raise ValidationError(
                        "Each ask must be a list/tuple with at least 2 elements [price, volume]"
                    )
                if not isinstance(ask[0], (int, float)) or not isinstance(
                    ask[1], (int, float)
                ):
                    raise ValidationError("Ask price and volume must be numbers")
                if ask[0] <= 0 or ask[1] <= 0:
                    raise ValidationError("Ask price and volume must be positive")
            # Проверка логики стакана
            if bids and asks:
                best_bid = max(bid[0] for bid in bids)
                best_ask = min(ask[0] for ask in asks)
                if best_bid >= best_ask:
                    raise ValidationError(
                        "Best bid cannot be higher than or equal to best ask"
                    )
            # Проверка сортировки
            if len(bids) > 1:
                bid_prices = [bid[0] for bid in bids]
                if bid_prices != sorted(bid_prices, reverse=True):
                    raise ValidationError("Bids must be sorted in descending order")
            if len(asks) > 1:
                ask_prices = [ask[0] for ask in asks]
                if ask_prices != sorted(ask_prices):
                    raise ValidationError("Asks must be sorted in ascending order")
            return True
        except Exception as e:
            self.logger.error(f"Error validating order book: {e}")
            raise

    def validate_pattern_memory(self, pattern_memory: Optional[PatternMemoryData]) -> bool:
        """Валидация данных паттернов."""
        try:
            if pattern_memory is None:
                return True  # Паттерны могут отсутствовать
            if not isinstance(pattern_memory, dict):
                raise ValidationError("Pattern memory must be a dictionary")
            # Проверка структуры данных для каждого символа
            for symbol, patterns in pattern_memory.items():
                if not isinstance(symbol, str):
                    raise ValidationError("Symbol keys must be strings")
                if not isinstance(patterns, dict):
                    raise ValidationError(
                        f"Pattern data for {symbol} must be a dictionary"
                    )
                # Проверка обязательных полей
                required_fields = ["confidence", "historical_match", "complexity"]
                for field in required_fields:
                    if field in patterns:
                        value = patterns[field]
                        if not isinstance(value, (int, float)):
                            raise ValidationError(f"Field {field} must be a number")
                        if not 0.0 <= value <= 1.0:
                            raise ValidationError(
                                f"Field {field} must be between 0.0 and 1.0"
                            )
            return True
        except Exception as e:
            self.logger.error(f"Error validating pattern memory: {e}")
            raise

    def validate_session_data(self, session_data: Optional[SessionData]) -> bool:
        """Валидация данных сессии."""
        try:
            if session_data is None:
                return True  # Данные сессии могут отсутствовать
            if not isinstance(session_data, dict):
                raise ValidationError("Session data must be a dictionary")
            # Проверка обязательных полей
            required_fields = ["alignment", "activity", "volatility"]
            for field in required_fields:
                if field in session_data:
                    value = session_data[field]
                    if not isinstance(value, (int, float)):
                        raise ValidationError(f"Field {field} must be a number")
                    if not 0.0 <= value <= 1.0:
                        raise ValidationError(
                            f"Field {field} must be between 0.0 and 1.0"
                        )
            return True
        except Exception as e:
            self.logger.error(f"Error validating session data: {e}")
            raise

    def validate_symbol(self, symbol: str) -> bool:
        """Валидация торгового символа."""
        try:
            if not isinstance(symbol, str):
                raise ValidationError("Symbol must be a string")
            if not symbol.strip():
                raise ValidationError("Symbol cannot be empty")
            # Проверка формата символа (базовая проверка)
            if len(symbol) < 3 or len(symbol) > 20:
                raise ValidationError(
                    "Symbol length must be between 3 and 20 characters"
                )
            # Проверка на допустимые символы
            allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
            if not all(c in allowed_chars for c in symbol.upper()):
                raise ValidationError("Symbol contains invalid characters")
            return True
        except Exception as e:
            self.logger.error(f"Error validating symbol: {e}")
            raise

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Валидация конфигурации."""
        try:
            if not isinstance(config, dict):
                raise ValidationError("Configuration must be a dictionary")
            # Проверка обязательных параметров
            required_params = ["min_volume_threshold", "max_spread_threshold"]
            for param in required_params:
                if param in config:
                    value = config[param]
                    if not isinstance(value, (int, float)):
                        raise ValidationError(f"Parameter {param} must be a number")
                    if value <= 0:
                        raise ValidationError(f"Parameter {param} must be positive")
            # Проверка весов (если есть)
            weight_params = [
                "alpha1_liquidity_score",
                "alpha2_volume_stability",
                "alpha3_structural_predictability",
                "alpha4_orderbook_symmetry",
                "alpha5_session_alignment",
                "alpha6_historical_pattern_match",
            ]
            weights = [config.get(param, 0.0) for param in weight_params]
            total_weight = sum(weights)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValidationError(f"Weights must sum to 1.0, got {total_weight}")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            raise

    def validate_all(
        self,
        symbol: str,
        market_data: MarketDataFrame,
        order_book: OrderBookData,
        pattern_memory: Optional[PatternMemoryData] = None,
        session_data: Optional[SessionData] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Валидация всех входных данных."""
        try:
            self.validate_symbol(symbol)
            self.validate_ohlcv_data(market_data)
            self.validate_order_book(order_book)
            self.validate_pattern_memory(pattern_memory)
            self.validate_session_data(session_data)
            if config:
                self.validate_configuration(config)
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise


class SymbolValidator:
    """Публичный валидатор для экспорта через __init__.py."""

    def __init__(self) -> None:
        self._validator = SymbolDataValidator()

    def validate_ohlcv_data(self, data: MarketDataFrame) -> bool:
        """Валидация OHLCV данных."""
        return self._validator.validate_ohlcv_data(data)

    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """Валидация стакана заявок."""
        return self._validator.validate_order_book(order_book)

    def validate_pattern_memory(self, pattern_memory: Optional[PatternMemoryData]) -> bool:
        """Валидация памяти паттернов."""
        return self._validator.validate_pattern_memory(pattern_memory)

    def validate_session_data(self, session_data: Optional[SessionData]) -> bool:
        """Валидация данных сессии."""
        return self._validator.validate_session_data(session_data)

    def validate_symbol(self, symbol: str) -> bool:
        """Валидация символа."""
        return self._validator.validate_symbol(symbol)

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Валидация конфигурации."""
        return self._validator.validate_configuration(config)

    def validate_all(
        self,
        symbol: str,
        market_data: MarketDataFrame,
        order_book: OrderBookData,
        pattern_memory: Optional[PatternMemoryData] = None,
        session_data: Optional[SessionData] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Валидация всех данных."""
        return self._validator.validate_all(
            symbol, market_data, order_book, pattern_memory, session_data, config
        )
