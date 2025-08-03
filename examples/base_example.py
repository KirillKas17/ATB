"""
Базовый класс для всех примеров ATB системы.

Обеспечивает единообразную архитектуру, типизацию и функциональность
для всех примеров использования системы.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from shared.models.example_types import (
    ExampleConfig,
    ExampleResult,
    ExampleMode,
    MarketDataProvider,
    StrategyExecutor,
    RiskManager
)
from domain.types import Symbol
from domain.value_objects.timestamp import Timestamp

__module__ = "examples.base_example"

@runtime_checkable
class DataProvider(Protocol):
    """Протокол для провайдера данных."""
    
    async def get_market_data(
        self, 
        symbol: Symbol, 
        start_time: pd.Timestamp, 
        end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Получить рыночные данные."""
        ...
    
    async def get_order_book(self, symbol: Symbol) -> Dict[str, Any]:
        """Получить ордербук."""
        ...
    
    async def get_recent_trades(self, symbol: Symbol, limit: int = 100) -> List[Dict[str, Any]]:
        """Получить последние сделки."""
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Протокол для сбора метрик."""
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Записать метрику."""
        ...
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получить сводку метрик."""
        ...
    
    def reset_metrics(self) -> None:
        """Сбросить метрики."""
        ...


@dataclass
class ExampleContext:
    """Контекст выполнения примера."""
    example_id: UUID = field(default_factory=uuid4)
    start_time: Optional[Timestamp] = None
    end_time: Optional[Timestamp] = None
    symbols: List[Symbol] = field(default_factory=list)
    config: Optional[ExampleConfig] = None
    data_provider: Optional[DataProvider] = None
    metrics_collector: Optional[MetricsCollector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExample(ABC):
    """
    Базовый класс для всех примеров ATB системы.
    
    Обеспечивает:
    - Единообразную архитектуру
    - Строгую типизацию
    - Сбор метрик и логирование
    - Обработку ошибок
    - Конфигурацию и валидацию
    """
    
    def __init__(self, config: Optional[ExampleConfig] = None):
        self.config = config or ExampleConfig()
        self.context = ExampleContext(config=self.config)
        self._setup_logging()
        self._validate_config()
    
    @abstractmethod
    async def setup(self) -> None:
        """Настройка примера перед выполнением."""
        pass
    
    @abstractmethod
    async def run(self) -> ExampleResult:
        """Основная логика выполнения примера."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Очистка ресурсов после выполнения."""
        pass
    
    @abstractmethod
    def validate_prerequisites(self) -> bool:
        """Проверка необходимых условий для выполнения."""
        pass
    
    def _setup_logging(self) -> None:
        """Настройка логирования для примера."""
        if self.config.enable_logging:
            logger.add(
                f"logs/examples/{self.__class__.__name__}.log",
                rotation="1 day",
                retention="7 days",
                level="INFO"
            )
    
    def _validate_config(self) -> None:
        """Валидация конфигурации."""
        if not self.config.symbols and self.config.mode != ExampleMode.DEMO:
            raise ValueError("Symbols must be specified for non-demo modes")
        
        if self.config.risk_level <= 0 or self.config.risk_level > 1:
            raise ValueError("Risk level must be between 0 and 1")
        
        if self.config.max_positions <= 0:
            raise ValueError("Max positions must be positive")
    
    async def execute_with_metrics(self) -> ExampleResult:
        """Выполнение примера с метриками и обработкой ошибок."""
        start_time = time.time()
        self.context.start_time = Timestamp(datetime.now())  # [1] правильное создание Timestamp
        
        try:
            # Проверка предварительных условий
            if not self.validate_prerequisites():
                return ExampleResult(
                    success=False,
                    duration_seconds=0.0,
                    trades_executed=0,
                    total_pnl=Decimal("0"),
                    max_drawdown=Decimal("0"),
                    sharpe_ratio=0.0,
                    error_message="Prerequisites validation failed"
                )
            
            # Настройка
            await self.setup()
            
            # Выполнение
            result = await self.run()
            
            # Очистка
            await self.cleanup()
            
            # Расчет длительности
            duration = time.time() - start_time
            self.context.end_time = Timestamp(datetime.now())  # [1] правильное создание Timestamp
            
            # Создаем новый результат с обновленными данными
            updated_result = ExampleResult(
                success=result.success,
                duration_seconds=duration,
                trades_executed=result.trades_executed,
                total_pnl=result.total_pnl,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
                error_message=result.error_message,
                metadata={
                    **result.metadata,
                    "example_id": str(self.context.example_id),
                    "start_time": str(self.context.start_time) if self.context.start_time else None,
                    "end_time": str(self.context.end_time) if self.context.end_time else None,
                    "symbols": self.context.symbols
                }
            )
            
            return updated_result
            
        except Exception as e:
            logger.error(f"Error in example execution: {e}")
            duration = time.time() - start_time
            
            return ExampleResult(
                success=False,
                duration_seconds=duration,
                trades_executed=0,
                total_pnl=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0,
                error_message=str(e),
                metadata={
                    "example_id": str(self.context.example_id),
                    "error_type": type(e).__name__
                }
            )
    
    def calculate_performance_metrics(
        self, 
        trades: List[Dict[str, Any]], 
        initial_balance: Decimal = Decimal("10000")
    ) -> Dict[str, Any]:
        """Расчет метрик производительности."""
        if not trades:
            return {
                "total_pnl": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            }
        
        # Расчет P&L
        total_pnl = sum(Decimal(str(trade.get("pnl", 0))) for trade in trades)
        
        # Расчет максимальной просадки
        balance = initial_balance
        max_balance = balance
        max_drawdown = Decimal("0")
        
        for trade in trades:
            pnl = Decimal(str(trade.get("pnl", 0)))
            balance += pnl
            max_balance = max(max_balance, balance)
            drawdown = (max_balance - balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        # Расчет Sharpe Ratio
        returns = [float(trade.get("pnl", 0)) for trade in trades]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Расчет win rate
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        win_rate = winning_trades / len(trades) if trades else 0.0
        
        return {
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "total_trades": len(trades)
        }
    
    def generate_test_data(
        self, 
        symbol: Symbol, 
        periods: int = 1000,
        start_price: float = 100.0
    ) -> pd.DataFrame:
        """Генерация тестовых данных."""
        np.random.seed(42)
        
        # Генерация цен с случайным блужданием
        returns = np.random.normal(0, 0.02, periods)
        prices = [start_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Создание DataFrame
        # Исправлено: используем правильные pandas функции
        timestamps = pd.date_range(
            start=datetime.now() - pd.Timedelta(days=periods),  # type: ignore
            periods=periods,
            freq='1min'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.randint(100, 10000, periods)
        })
        
        df['symbol'] = symbol
        # Исправлено: используем правильный метод set_index
        return df.set_index('timestamp')
    
    def log_performance_summary(self, result: ExampleResult) -> None:
        """Логирование сводки производительности."""
        logger.info(f"Example execution completed:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Duration: {result.duration_seconds:.2f}s")
        logger.info(f"  Trades executed: {result.trades_executed}")
        logger.info(f"  Total P&L: {result.total_pnl}")
        logger.info(f"  Max drawdown: {result.max_drawdown}")
        logger.info(f"  Sharpe ratio: {result.sharpe_ratio:.4f}")
        
        if result.error_message:
            logger.error(f"  Error: {result.error_message}")


class ExampleFactory:
    """Фабрика для создания примеров."""
    
    _examples: Dict[str, type[BaseExample]] = {}
    
    @classmethod
    def register(cls, name: str, example_class: type[BaseExample]) -> None:
        """Регистрация примера."""
        cls._examples[name] = example_class
    
    @classmethod
    def create(cls, name: str, config: Optional[ExampleConfig] = None) -> BaseExample:
        """Создание примера по имени."""
        if name not in cls._examples:
            raise ValueError(f"Unknown example: {name}")
        return cls._examples[name](config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Список доступных примеров."""
        return list(cls._examples.keys())


def register_example(name: str):
    """Декоратор для регистрации примера."""
    def decorator(cls: type[BaseExample]) -> type[BaseExample]:
        ExampleFactory.register(name, cls)
        return cls
    return decorator 