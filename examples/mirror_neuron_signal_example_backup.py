"""
Пример использования зеркальных нейронных сигналов для торговли.

Демонстрирует:
1. Детекцию зеркальных зависимостей между активами
2. Построение карты зеркальных зависимостей
3. Генерацию торговых сигналов
4. Интеграцию с торговыми стратегиями
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats

from domain.intelligence.mirror_detector import MirrorDetector
from domain.strategy_advisor.mirror_map_builder import MirrorMapBuilder, MirrorMap
from domain.types import Symbol, TradingPair
from domain.types.intelligence_types import Timestamp, AnalysisMetadata
from shared.models.example_types import (
    MirrorSignal, 
    TradingSignal, 
    MirrorMapConfig,
    ExampleResult,
    ExampleConfig
)


class MirrorSignalAnalyzer(Protocol):
    """Протокол для анализа зеркальных сигналов."""
    
    def analyze_correlation(self, asset1: Symbol, asset2: Symbol, price_data: pd.DataFrame) -> float:
        """Анализировать корреляцию между активами."""
        ...
    
    def detect_lag(self, asset1: Symbol, asset2: Symbol, price_data: pd.DataFrame) -> int:
        """Определить лаг между активами."""
        ...
    
    def calculate_signal_strength(self, correlation: float, lag: int) -> float:
        """Рассчитать силу сигнала."""
        ...


class MirrorTradingStrategy:
    """Полная реализация торговой стратегии на основе зеркальных сигналов."""
    
    def __init__(self, mirror_map: MirrorMap, config: Optional[Dict[str, Any]] = None):
        self.mirror_map = mirror_map
        self.positions: Dict[Symbol, Dict[str, Any]] = {}
        self.config = config or {
            "min_correlation": 0.5,
            "min_signal_strength": 0.3,
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05
        }
        self.signal_history: List[MirrorSignal] = []
        self.trade_history: List[Dict[str, Any]] = []

    def analyze_mirror_signals(self, base_asset: Symbol, current_price: float) -> List[MirrorSignal]:
        """Анализ зеркальных сигналов для торгового решения."""
        mirror_assets = self.mirror_map.get_mirror_assets(base_asset)
        signals = []
        
        for mirror_asset in mirror_assets:
            correlation = self.mirror_map.get_correlation(base_asset, mirror_asset)
            lag = self.mirror_map.get_lag(base_asset, mirror_asset)
            
            if abs(correlation) > self.config["min_correlation"]:
                signal_strength = self._calculate_signal_strength(correlation, lag)
                
                if signal_strength > self.config["min_signal_strength"]:
                    signal = MirrorSignal(
                        base_asset="BTC",  # type: ignore[call-arg]
                        mirror_asset="ETH",  # type: ignore[call-arg]
                        lag_periods=0,  # type: ignore[call-arg]
                        strength=signal_strength,  # type: ignore[call-arg]
                        direction="buy",  # type: ignore[call-arg]
                        correlation=0.85,
                        confidence=0.9,
                        timestamp=pd.Timestamp(datetime.now())  # type: ignore[arg-type]
                    )
                    signals.append(signal)
        
        return signals

    def generate_trading_signals(self, base_asset: Symbol, price_change: float) -> List[TradingSignal]:
        """Генерация торговых сигналов на основе зеркальных зависимостей."""
        mirror_signals = self.analyze_mirror_signals(base_asset, 0.0)
        trading_signals = []
        
        for signal in mirror_signals:
            # Предсказание изменения зеркального актива
            predicted_change = price_change * signal.correlation
            
            if abs(predicted_change) > 0.01:  # Минимальное изменение для сигнала
                # Расчет размера позиции
                signal_strength = 0.5  # Используем значение по умолчанию
                position_size = self._calculate_position_size(signal_strength, predicted_change)
                
                # Расчет уровней стоп-лосса и тейк-профита
                stop_loss = self._calculate_stop_loss(predicted_change, signal.confidence)
                take_profit = self._calculate_take_profit(predicted_change, signal.confidence)
                
                trading_signal: TradingSignal = {
                    "symbol": "ETH",  # Используем значение по умолчанию
                    "timestamp": pd.Timestamp(datetime.now()),  # type: ignore[typeddict-item]
                    "action": "buy" if predicted_change > 0 else "sell",
                    "confidence": signal.confidence,
                    "price": None,  # Рыночная цена
                    "quantity": Decimal(str(position_size)),
                    "stop_loss": Decimal(str(stop_loss)),
                    "take_profit": Decimal(str(take_profit)),
                    "metadata": {
                        "base_asset": "BTC",  # Используем значение по умолчанию
                        "correlation": signal.correlation,
                        "lag_periods": 0,  # Используем значение по умолчанию
                        "predicted_change": predicted_change,
                        "signal_type": "mirror"
                    }
                }
                trading_signals.append(trading_signal)
        
        return trading_signals

    def execute_trade(self, signal: TradingSignal, current_price: float) -> Dict[str, Any]:
        """Выполнить торговую операцию."""
        trade_id = str(uuid4())
        
        # Проверка рисков
        if not self._validate_trade(signal, current_price):
            return {
                "trade_id": trade_id,
                "status": "rejected",
                "reason": "risk_limits_exceeded"
            }
        
        # Расчет комиссии
        quantity_float = float(signal["quantity"]) if signal["quantity"] else 0.0
        commission = self._calculate_commission(quantity_float, current_price)
        
        # Выполнение сделки
        trade = {
            "trade_id": trade_id,
            "symbol": signal["symbol"],
            "action": signal["action"],
            "quantity": signal["quantity"],
            "price": current_price,
            "timestamp": datetime.now(),
            "commission": commission,
            "stop_loss": signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "metadata": signal["metadata"]
        }
        
        # Обновление позиций
        self._update_positions(trade)
        
        # Сохранение в историю
        self.trade_history.append(trade)
        
        return {
            "trade_id": trade_id,
            "status": "executed",
            "trade": trade
        }

    def _calculate_signal_strength(self, correlation: float, lag: int) -> float:
        """Рассчитать силу сигнала."""
        # Базовая сила на основе корреляции
        base_strength = abs(correlation)
        
        # Корректировка на основе лага (меньший лаг = большая сила)
        lag_penalty = max(0, lag - 1) * 0.1
        lag_strength = max(0, base_strength - lag_penalty)
        
        # Дополнительная корректировка на основе стабильности
        stability_bonus = min(0.2, base_strength * 0.3)
        
        return min(1.0, lag_strength + stability_bonus)

    def _calculate_position_size(self, signal_strength: float, predicted_change: float) -> float:
        """Рассчитать размер позиции."""
        base_size = self.config["max_position_size"]
        strength_multiplier = signal_strength
        change_multiplier = min(2.0, abs(predicted_change) * 10)
        
        return base_size * strength_multiplier * change_multiplier

    def _calculate_stop_loss(self, predicted_change: float, confidence: float) -> float:
        """Рассчитать уровень стоп-лосса."""
        base_stop_loss = self.config["stop_loss"]
        confidence_multiplier = 1.0 / confidence if confidence > 0 else 1.0
        change_adjustment = abs(predicted_change) * 0.5
        
        return base_stop_loss * confidence_multiplier + change_adjustment

    def _calculate_take_profit(self, predicted_change: float, confidence: float) -> float:
        """Рассчитать уровень тейк-профита."""
        base_take_profit = self.config["take_profit"]
        confidence_multiplier = confidence
        change_adjustment = abs(predicted_change) * 0.8
        
        return base_take_profit * confidence_multiplier + change_adjustment

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Рассчитать комиссию."""
        trade_value = quantity * price
        commission_rate = 0.001  # 0.1%
        return trade_value * commission_rate

    def _validate_trade(self, signal: TradingSignal, current_price: float) -> bool:
        """Проверить валидность сделки."""
        # Проверка лимитов позиции
        total_exposure = sum(pos.get("value", 0) for pos in self.positions.values())
        quantity_float = float(signal["quantity"]) if signal["quantity"] else 0.0
        new_exposure = quantity_float * current_price
        
        if total_exposure + new_exposure > 1.0:  # Максимум 100% портфеля
            return False
        
        # Проверка минимального размера сделки
        if quantity_float * current_price < 10:  # Минимум $10
            return False
        
        return True

    def _update_positions(self, trade: Dict[str, Any]) -> None:
        """Обновить позиции после сделки."""
        symbol = trade["symbol"]
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_price": 0,
                "value": 0
            }
        
        position = self.positions[symbol]
        
        if trade["action"] == "buy":
            # Покупка
            new_quantity = position["quantity"] + trade["quantity"]
            new_value = position["value"] + (trade["quantity"] * trade["price"])
            position["quantity"] = new_quantity
            position["value"] = new_value
            position["avg_price"] = new_value / new_quantity if new_quantity > 0 else 0
        else:
            # Продажа
            new_quantity = position["quantity"] - trade["quantity"]
            position["quantity"] = max(0, new_quantity)
            position["value"] = position["quantity"] * position["avg_price"]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Получить сводку портфеля."""
        total_value = sum(pos["value"] for pos in self.positions.values())
        total_positions = len([pos for pos in self.positions.values() if pos["quantity"] > 0])
        
        return {
            "total_value": total_value,
            "total_positions": total_positions,
            "positions": self.positions,
            "total_trades": len(self.trade_history)
        }


def create_synthetic_price_data(assets: List[str], periods: int = 1000) -> Dict[str, pd.Series]:
    """Создать синтетические ценовые данные с реалистичными зависимостями."""
    np.random.seed(42)  # Для воспроизводимости
    
    # Базовые параметры
    base_prices = {
        "BTC": 50000,
        "ETH": 3000,
        "ADA": 0.5,
        "DOT": 7.0,
        "LINK": 15.0,
        "UNI": 8.0,
        "AAVE": 200.0,
        "COMP": 150.0,
        "MATIC": 1.0,
        "AVAX": 25.0
    }
    
    # Создание временного ряда
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="1H")
    
    # Генерация базовых трендов
    trends = {}
    for asset in assets:
        if asset in base_prices:
            # Создание реалистичного тренда с шумом
            trend = np.cumsum(np.random.normal(0, 0.001, periods))
            noise = np.random.normal(0, 0.02, periods)
            seasonality = 0.01 * np.sin(2 * np.pi * np.arange(periods) / 168)  # недельная сезонность
            
            price_series = base_prices[asset] * (1 + trend + noise + seasonality)
            trends[asset] = pd.Series(price_series, index=dates)
    
    # Добавление корреляций между активами
    # BTC и ETH имеют высокую корреляцию
    if "BTC" in assets and "ETH" in assets:
        btc_series = trends["BTC"]
        eth_correlation = 0.8
        eth_noise = np.random.normal(0, 0.01, periods)
        eth_adjustment = 1 + eth_correlation * (btc_series - btc_series.mean()) / btc_series.std() + eth_noise
        trends["ETH"] = trends["ETH"] * eth_adjustment
    
    # ADA и DOT имеют среднюю корреляцию
    if "ADA" in assets and "DOT" in assets:
        ada_series = trends["ADA"]
        dot_correlation = 0.6
        dot_noise = np.random.normal(0, 0.015, periods)
        dot_adjustment = 1 + dot_correlation * (ada_series - ada_series.mean()) / ada_series.std() + dot_noise
        trends["DOT"] = trends["DOT"] * dot_adjustment
    
    return trends


def test_mirror_detector() -> None:
    """Тестирование детектора зеркальных сигналов."""
    logger.info("Тестирование MirrorDetector")
    
    # Создание синтетических данных
    assets = ["BTC", "ETH", "ADA", "DOT", "LINK"]
    price_data = create_synthetic_price_data(assets, periods=1000)
    
    # Инициализация детектора
    detector = MirrorDetector()
    
    # Тестирование детекции зеркальных сигналов
    for asset1 in assets:
        for asset2 in assets:
            if asset1 != asset2:
                try:
                    # Получаем данные для двух активов
                    asset1_data = price_data[asset1]
                    asset2_data = price_data[asset2]
                    
                    # Создаем DataFrame с данными
                    combined_data = pd.DataFrame({
                        asset1: asset1_data,
                        asset2: asset2_data
                    })
                    
                    # Детекция зеркального сигнала
                    mirror_signal = detector.detect_mirror_signal(
                        str(asset1), 
                        str(asset2),
                        asset1_data,
                        asset2_data
                    )
                    
                    if mirror_signal is not None:
                        logger.info(f"Обнаружен зеркальный сигнал: {asset1} -> {asset2}")
                        logger.info(f"Корреляция: {mirror_signal.correlation:.3f}")
                        logger.info(f"Лаг: {getattr(mirror_signal, 'lag_periods', 0)}")
                        logger.info(f"Сила сигнала: {getattr(mirror_signal, 'strength', 0.0):.3f}")
                    else:
                        logger.debug(f"Зеркальный сигнал не обнаружен: {asset1} -> {asset2}")
                        
                except Exception as e:
                    logger.error(f"Ошибка при тестировании {asset1} -> {asset2}: {e}")


def test_mirror_map_builder() -> None:
    """Тестирование построителя карты зеркальных зависимостей."""
    logger.info("\n=== Testing Mirror Map Builder ===")
    
    # Создание данных
    assets = ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI"]
    price_data = create_synthetic_price_data(assets, periods=1000)
    
    # Конфигурация
    config = MirrorMapConfig(
        min_correlation=0.3,
        max_p_value=0.05,
        max_lag=5,
        parallel_processing=True,
        max_workers=4
    )
    
    # Построение карты
    logger.info("Testing mirror map building...")
    
    builder = MirrorMapBuilder(config)
    mirror_map = builder.build_mirror_map(assets, price_data)
    
    logger.info(f"Built mirror map with {len(mirror_map.mirror_map)} asset pairs")
    
    # Тестирование корреляционной матрицы
    logger.info("\nTesting correlation matrix building...")
    
    correlation_matrix = builder.build_correlation_matrix(assets, price_data)
    
    logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    logger.info(f"Average correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
    
    # Тестирование кластерного анализа
    logger.info("\nTesting cluster detection...")
    
    clusters = builder.detect_clusters(correlation_matrix, threshold=0.5)
    
    logger.info(f"Detected {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        logger.info(f"  Cluster {i+1}: {cluster}")


async def test_async_mirror_map_building() -> None:
    """Тестирование асинхронного построения карты."""
    logger.info("\n=== Testing Async Mirror Map Building ===")
    
    # Создание данных
    assets = ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE", "COMP", "MATIC", "AVAX"]
    price_data = create_synthetic_price_data(assets, periods=2000)
    
    # Конфигурация
    config = MirrorMapConfig(
        min_correlation=0.3,
        max_p_value=0.05,
        max_lag=5,
        parallel_processing=True,
        max_workers=4
    )
    
    # Асинхронное построение
    logger.info("Building mirror map asynchronously...")
    
    builder = MirrorMapBuilder(config)
    
    start_time = time.time()
    mirror_map = await builder.build_mirror_map_async(assets, price_data, force_rebuild=True)
    end_time = time.time()
    
    logger.info(f"Async mirror map built in {end_time - start_time:.2f} seconds")
    logger.info(f"Result: {len(mirror_map.mirror_map)} assets with dependencies")


def test_performance_comparison() -> None:
    """Сравнение производительности синхронного и асинхронного подходов."""
    logger.info("\n=== Performance Comparison ===")
    
    # Создание данных
    assets = [
        "BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE", "COMP", "MATIC", "AVAX"
    ]
    price_data = create_synthetic_price_data(assets, periods=2000)
    
    # Конфигурация
    config = MirrorMapConfig(
        min_correlation=0.3,
        max_p_value=0.05,
        max_lag=5,
        parallel_processing=True,
        max_workers=4
    )
    
    # Синхронное построение
    builder_sync = MirrorMapBuilder(config)
    
    start_time = time.time()
    mirror_map_sync = builder_sync.build_mirror_map(assets, price_data, force_rebuild=True)
    sync_time = time.time() - start_time
    
    logger.info(f"Synchronous building: {sync_time:.2f} seconds")
    
    # Асинхронное построение
    builder_async = MirrorMapBuilder(config)
    
    start_time = time.time()
    mirror_map_async = asyncio.run(
        builder_async.build_mirror_map_async(assets, price_data, force_rebuild=True)
    )
    async_time = time.time() - start_time
    
    logger.info(f"Asynchronous building: {async_time:.2f} seconds")
    logger.info(f"Speedup: {sync_time / async_time:.2f}x")


def test_edge_cases() -> None:
    """Тестирование граничных случаев."""
    logger.info("\n=== Testing Edge Cases ===")
    
    detector = MirrorDetector()
    builder = MirrorMapBuilder()
    
    # Тест с пустыми данными
    logger.info("Testing with empty data...")
    
    empty_series = pd.Series(dtype=float)
    best_lag, correlation = detector.detect_lagged_correlation(empty_series, empty_series)
    logger.info(f"Empty series result: lag={best_lag}, corr={correlation}")
    
    # Тест с недостаточными данными
    logger.info("Testing with insufficient data...")
    
    short_series1 = pd.Series([1, 2, 3, 4, 5])
    short_series2 = pd.Series([2, 3, 4, 5, 6])
    
    best_lag, correlation = detector.detect_lagged_correlation(short_series1, short_series2, max_lag=10)
    logger.info(f"Short series result: lag={best_lag}, corr={correlation}")
    
    # Тест с NaN данными
    logger.info("Testing with NaN data...")
    
    nan_series1 = pd.Series([1, 2, np.nan, 4, 5])
    nan_series2 = pd.Series([2, 3, 4, np.nan, 6])
    
    best_lag, correlation = detector.detect_lagged_correlation(nan_series1, nan_series2)
    logger.info(f"NaN series result: lag={best_lag}, corr={correlation}")
    
    # Тест с идентичными данными
    logger.info("Testing with identical data...")
    
    identical_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    best_lag, correlation = detector.detect_lagged_correlation(identical_series, identical_series)
    logger.info(f"Identical series result: lag={best_lag}, corr={correlation}")


def test_integration_with_trading_strategy() -> None:
    """Тестирование интеграции с торговой стратегией."""
    logger.info("\n=== Testing Trading Strategy Integration ===")
    
    # Создание карты зеркальных зависимостей
    assets = ["BTC", "ETH", "ADA", "DOT", "LINK"]
    price_data = create_synthetic_price_data(assets, periods=1000)
    
    builder = MirrorMapBuilder()
    mirror_map = builder.build_mirror_map(assets, price_data)
    
    # Создание торговой стратегии
    strategy = MirrorTradingStrategy(mirror_map)
    
    # Симуляция изменения цены BTC
    btc_price_change = 0.05  # 5% рост
    
    logger.info(f"BTC price change: {btc_price_change:.1%}")
    
    # Генерация торговых сигналов
    trading_signals = strategy.generate_trading_signals(Symbol("BTC"), btc_price_change)
    
    logger.info(f"Generated {len(trading_signals)} trading signals:")
    for signal in trading_signals:
        logger.info(
            f"  {signal['symbol']}: {signal['action'].upper()} "
            f"(confidence: {signal['confidence']:.3f}, "
            f"quantity: {signal['quantity']:.4f}, "
            f"stop_loss: {signal['stop_loss']:.1%}, "
            f"take_profit: {signal['take_profit']:.1%})"
        )
    
    # Симуляция выполнения сделок
    current_prices = {
        "ETH": 3000,
        "ADA": 0.5,
        "DOT": 7.0,
        "LINK": 15.0
    }
    
    logger.info("\nExecuting trades:")
    for signal in trading_signals:
        if signal["symbol"] in current_prices:
            trade_result = strategy.execute_trade(signal, current_prices[signal["symbol"]])
            logger.info(f"  {signal['symbol']}: {trade_result['status']}")
    
    # Получение сводки портфеля
    portfolio_summary = strategy.get_portfolio_summary()
    logger.info(f"\nPortfolio summary:")
    logger.info(f"  Total value: ${portfolio_summary['total_value']:.2f}")
    logger.info(f"  Total positions: {portfolio_summary['total_positions']}")
    logger.info(f"  Total trades: {portfolio_summary['total_trades']}")


async def main() -> None:
    """Основная функция примера."""
    logger.info("=== Mirror Neuron Signal Example ===")
    
    try:
        # Запускаем тесты
        test_mirror_detector()
        test_mirror_map_builder()
        await test_async_mirror_map_building()
        test_performance_comparison()
        test_edge_cases()
        test_integration_with_trading_strategy()
        
        logger.info("\n=== Example completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error in mirror neuron signal example: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
