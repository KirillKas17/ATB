# -*- coding: utf-8 -*-
"""
Пример использования системы гравитации ликвидности.

Этот пример демонстрирует:
1. Создание и настройку модели гравитации ликвидности
2. Анализ ордербука и вычисление гравитации
3. Оценку рисков и корректировку агрессивности агента
4. Интеграцию с торговыми решениями
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from application.risk.liquidity_gravity_monitor import (
    LiquidityGravityMonitor,
    LiquidityGravityResult,
    RiskAssessmentResult
)
from domain.market.liquidity_gravity import (
    LiquidityGravityConfig,
    LiquidityGravityModel,
    OrderBookSnapshot
)
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from shared.models.example_types import (
    LiquidityGravitySignal,
    ExampleResult,
    ExampleConfig,
    ExampleMode
)
from examples.base_example import BaseExample, register_example


class OrderBookGenerator(Protocol):
    """Протокол для генерации ордербуков."""
    
    def generate_normal_order_book(self, symbol: str, spread_percentage: float) -> OrderBookSnapshot:
        """Генерировать нормальный ордербук."""
        ...
    
    def generate_high_gravity_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Генерировать ордербук с высокой гравитацией."""
        ...
    
    def generate_stress_test_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Генерировать ордербук для стресс-тестирования."""
        ...


class LiquidityAnalyzer:
    """Полная реализация анализатора ликвидности."""
    
    def __init__(self, config: Optional[LiquidityGravityConfig] = None):
        self.config = config or LiquidityGravityConfig(
            gravitational_constant=1e-6,
            min_volume_threshold=0.001,
            max_price_distance=0.1,
            volume_weight=1.0,
            price_weight=1.0
        )
        self.model = LiquidityGravityModel(self.config)
        self.analysis_history: List[Dict[str, Any]] = []
        
    def analyze_order_book(self, order_book: OrderBookSnapshot) -> LiquidityGravitySignal:
        """Полный анализ ордербука."""
        # Вычисление гравитации
        gravity = self.model.compute_liquidity_gravity(order_book)
        
        # Полный анализ
        analysis_result = self.model.analyze_liquidity_gravity(order_book)
        
        # Определение уровня риска
        risk_level = self._determine_risk_level(gravity, analysis_result)
        
        # Расчет влияния на спред и объем
        spread_impact = self._calculate_spread_impact(order_book)
        volume_impact = self._calculate_volume_impact(order_book)
        
        # Генерация рекомендации
        recommendation = self._generate_recommendation(risk_level, gravity, spread_impact)
        
        # Создание сигнала
        signal = LiquidityGravitySignal(
            symbol=order_book.symbol,
            timestamp=pd.Timestamp(datetime.now()),  # Исправление 94: используем pandas.Timestamp
            gravity_score=gravity,
            risk_level=risk_level,
            spread_impact=spread_impact,
            volume_impact=volume_impact,
            recommendation=recommendation
        )
        
        # Сохранение в историю
        self.analysis_history.append({
            "timestamp": datetime.now(),
            "symbol": order_book.symbol,
            "gravity": gravity,
            "risk_level": risk_level,
            "analysis_result": analysis_result
        })
        
        return signal
    
    def _determine_risk_level(self, gravity: float, analysis_result: Any) -> str:
        """Определить уровень риска."""
        if gravity < 0.1:
            return "low"
        elif gravity < 0.5:
            return "medium"
        elif gravity < 1.0:
            return "high"
        else:
            return "extreme"
    
    def _calculate_spread_impact(self, order_book: OrderBookSnapshot) -> float:
        """Рассчитать влияние на спред."""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        best_bid = order_book.bids[0][0]
        best_ask = order_book.asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        return (spread / mid_price) * 100  # В процентах
    
    def _calculate_volume_impact(self, order_book: OrderBookSnapshot) -> float:
        """Рассчитать влияние на объем."""
        total_bid_volume = sum(volume for _, volume in order_book.bids)
        total_ask_volume = sum(volume for _, volume in order_book.asks)
        total_volume = total_bid_volume + total_ask_volume
        
        if total_volume == 0:
            return 0.0
        
        # Нормализованный объем
        return total_volume / 1000  # Нормализация к 1000 единицам
    
    def _generate_recommendation(self, risk_level: str, gravity: float, spread_impact: float) -> str:
        """Генерировать торговую рекомендацию."""
        if risk_level == "low":
            return "normal_trading"
        elif risk_level == "medium":
            return "reduce_position_size"
        elif risk_level == "high":
            return "increase_spread"
        else:  # extreme
            return "stop_trading"
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Получить сводку анализов."""
        if not self.analysis_history:
            return {}
        
        gravity_values = [entry["gravity"] for entry in self.analysis_history]
        risk_levels = [entry["risk_level"] for entry in self.analysis_history]
        
        return {
            "total_analyses": len(self.analysis_history),
            "average_gravity": np.mean(gravity_values),
            "max_gravity": np.max(gravity_values),
            "min_gravity": np.min(gravity_values),
            "risk_distribution": {
                "low": risk_levels.count("low"),
                "medium": risk_levels.count("medium"),
                "high": risk_levels.count("high"),
                "extreme": risk_levels.count("extreme")
            }
        }


class OrderBookSimulator:
    """Симулятор ордербуков для тестирования."""
    
    def __init__(self):
        self.base_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "ADA/USDT": 0.5,
            "DOT/USDT": 7.0,
            "LINK/USDT": 15.0
        }
    
    def generate_normal_order_book(self, symbol: str, spread_percentage: float = 0.1) -> OrderBookSnapshot:
        """Создание нормального ордербука."""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Создаем биды (покупатели)
        bids = []
        for i in range(10):
            price = base_price * (1 - (i + 1) * 0.001)
            volume = np.random.uniform(0.1, 2.0)
            bids.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        # Создаем аски (продавцы)
        asks = []
        for i in range(10):
            price = base_price * (1 + (i + 1) * 0.001)
            volume = np.random.uniform(0.1, 2.0)
            asks.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        return OrderBookSnapshot(
            exchange="example",
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=Timestamp(datetime.now())
        )
    
    def generate_high_gravity_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Создание ордербука с высокой гравитацией ликвидности."""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Концентрированные ордера около цены
        bids = []
        for i in range(5):
            price = base_price * (1 - (i + 1) * 0.0001)  # Очень близко к цене
            volume = np.random.uniform(5.0, 20.0)  # Большие объемы
            bids.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        asks = []
        for i in range(5):
            price = base_price * (1 + (i + 1) * 0.0001)  # Очень близко к цене
            volume = np.random.uniform(5.0, 20.0)  # Большие объемы
            asks.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        return OrderBookSnapshot(
            exchange="example",
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=Timestamp(datetime.now())
        )
    
    def generate_stress_test_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Создание ордербука для стресс-тестирования."""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Неравномерное распределение ликвидности
        bids = []
        for i in range(10):
            if i < 3:  # Первые 3 уровня - большие объемы
                price = base_price * (1 - (i + 1) * 0.001)
                volume = np.random.uniform(10.0, 50.0)
            else:  # Остальные - маленькие объемы
                price = base_price * (1 - (i + 1) * 0.002)
                volume = np.random.uniform(0.01, 0.1)
            bids.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        asks = []
        for i in range(10):
            if i < 3:  # Первые 3 уровня - большие объемы
                price = base_price * (1 + (i + 1) * 0.001)
                volume = np.random.uniform(10.0, 50.0)
            else:  # Остальные - маленькие объемы
                price = base_price * (1 + (i + 1) * 0.002)
                volume = np.random.uniform(0.01, 0.1)
            asks.append((Price(Decimal(str(price))), Volume(Decimal(str(volume)))))
        
        return OrderBookSnapshot(
            exchange="example",
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=Timestamp(datetime.now())
        )


@register_example("liquidity_gravity")
class LiquidityGravityExample(BaseExample):
    """Пример использования системы гравитации ликвидности."""
    
    def __init__(self, config: Optional[ExampleConfig] = None):
        super().__init__(config)
        self.analyzer = LiquidityAnalyzer()
        self.simulator = OrderBookSimulator()
        self.monitor = LiquidityGravityMonitor()
        self.analysis_results: List[Dict[str, Any]] = []
        
    async def setup(self) -> None:
        """Инициализация примера."""
        logger.info("Setting up Liquidity Gravity example...")
        
        # Добавляем символы для мониторинга
        symbols = self.config.symbols if self.config.symbols else ["BTC/USDT"]
        for symbol in symbols:
            self.monitor.add_symbol(symbol)
        
        logger.info("Liquidity Gravity example setup completed")
        
    async def run(self) -> ExampleResult:
        """Выполнение примера."""
        logger.info("Running Liquidity Gravity example...")
        start_time = time.time()
        
        try:
            # Симуляция различных сценариев
            scenarios = self._simulate_trading_decisions()
            
            # Анализ результатов
            total_analyses = len(scenarios)
            high_risk_count = sum(1 for s in scenarios if s.get("risk_level") == "high")
            extreme_risk_count = sum(1 for s in scenarios if s.get("risk_level") == "extreme")
            
            # Расчет метрик
            avg_gravity = Decimal(str(np.mean([s.get("gravity_score", 0) for s in scenarios]) or 0))
            avg_spread_impact = Decimal(str(np.mean([s.get("spread_impact", 0) for s in scenarios]) or 0))
            
            duration = time.time() - start_time
            
            return ExampleResult(
                success=True,
                duration_seconds=duration,
                trades_executed=total_analyses,
                total_pnl=Decimal("0"),  # Симуляция без реальной торговли
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0,
                metadata={
                    "total_analyses": total_analyses,
                    "high_risk_detections": high_risk_count,
                    "extreme_risk_detections": extreme_risk_count,
                    "average_gravity": avg_gravity,
                    "average_spread_impact": avg_spread_impact,
                    "risk_distribution": {
                        "low": sum(1 for s in scenarios if s.get("risk_level") == "low"),
                        "medium": sum(1 for s in scenarios if s.get("risk_level") == "medium"),
                        "high": high_risk_count,
                        "extreme": extreme_risk_count
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Liquidity Gravity example: {e}")
            duration = time.time() - start_time
            
            return ExampleResult(
                success=False,
                duration_seconds=duration,
                trades_executed=0,
                total_pnl=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0,
                error_message=str(e)
            )
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        self.monitor.stop_monitoring()
        logger.info("Liquidity Gravity example cleanup completed")
    
    def validate_prerequisites(self) -> bool:
        """Проверка необходимых условий."""
        return self.config is not None and len(self.config.symbols) > 0
    
    def _simulate_trading_decisions(self) -> List[Dict[str, Any]]:
        """Симуляция торговых решений на основе гравитации ликвидности."""
        scenarios = []
        symbols = self.config.symbols if self.config.symbols else ["BTC/USDT"]
        
        for symbol in symbols:
            # Нормальный ордербук
            normal_ob = self.simulator.generate_normal_order_book(symbol)
            normal_signal = self.analyzer.analyze_order_book(normal_ob)
            scenarios.append({
                "scenario": "normal",
                "symbol": symbol,
                "gravity_score": normal_signal.gravity_score,
                "risk_level": normal_signal.risk_level,
                "spread_impact": normal_signal.spread_impact,
                "recommendation": normal_signal.recommendation
            })
            
            # Высокая гравитация
            high_gravity_ob = self.simulator.generate_high_gravity_order_book(symbol)
            high_gravity_signal = self.analyzer.analyze_order_book(high_gravity_ob)
            scenarios.append({
                "scenario": "high_gravity",
                "symbol": symbol,
                "gravity_score": high_gravity_signal.gravity_score,
                "risk_level": high_gravity_signal.risk_level,
                "spread_impact": high_gravity_signal.spread_impact,
                "recommendation": high_gravity_signal.recommendation
            })
            
            # Стресс-тест
            stress_ob = self.simulator.generate_stress_test_order_book(symbol)
            stress_signal = self.analyzer.analyze_order_book(stress_ob)
            scenarios.append({
                "scenario": "stress_test",
                "symbol": symbol,
                "gravity_score": stress_signal.gravity_score,
                "risk_level": stress_signal.risk_level,
                "spread_impact": stress_signal.spread_impact,
                "recommendation": stress_signal.recommendation
            })
        
        return scenarios


def test_liquidity_gravity_model() -> None:
    """Тестирование модели гравитации ликвидности."""
    logger.info("Testing Liquidity Gravity Model...")
    
    # Создание модели
    config = LiquidityGravityConfig(
        gravitational_constant=1e-6,
        min_volume_threshold=0.001,
        max_price_distance=0.1
    )
    model = LiquidityGravityModel(config)
    
    # Создание тестового ордербука
    simulator = OrderBookSimulator()
    order_book = simulator.generate_normal_order_book("BTC/USDT")
    
    # Анализ
    result = model.analyze_liquidity_gravity(order_book)
    
    logger.info(f"Gravity analysis result: {result}")
    logger.info("Liquidity Gravity Model test completed")


def test_risk_assessor() -> None:
    """Тестирование оценщика рисков."""
    logger.info("Testing Risk Assessor...")
    
    # Создание анализатора
    analyzer = LiquidityAnalyzer()
    
    # Создание тестового ордербука
    simulator = OrderBookSimulator()
    order_book = simulator.generate_high_gravity_order_book("BTC/USDT")
    
    # Анализ
    signal = analyzer.analyze_order_book(order_book)
    
    logger.info(f"Risk assessment: {signal.risk_level}")
    logger.info(f"Gravity score: {signal.gravity_score}")
    logger.info(f"Recommendation: {signal.recommendation}")
    logger.info("Risk Assessor test completed")


def test_gravity_filter() -> None:
    """Тестирование фильтра гравитации."""
    logger.info("Testing Gravity Filter...")
    
    # Создание монитора
    monitor = LiquidityGravityMonitor()
    
    # Добавление символа
    monitor.add_symbol("BTC/USDT")
    
    # Получение статистики
    stats = monitor.get_monitoring_statistics()
    
    logger.info(f"Monitoring statistics: {stats}")
    logger.info("Gravity Filter test completed")


def test_market_maker_integration() -> None:
    """Тестирование интеграции с маркет-мейкером."""
    logger.info("Testing Market Maker Integration...")
    
    # Создание анализатора
    analyzer = LiquidityAnalyzer()
    
    # Симуляция различных рыночных условий
    simulator = OrderBookSimulator()
    
    scenarios = [
        ("normal", simulator.generate_normal_order_book("BTC/USDT")),
        ("high_gravity", simulator.generate_high_gravity_order_book("BTC/USDT")),
        ("stress", simulator.generate_stress_test_order_book("BTC/USDT"))
    ]
    
    for scenario_name, order_book in scenarios:
        signal = analyzer.analyze_order_book(order_book)
        logger.info(f"{scenario_name}: {signal.recommendation}")
    
    logger.info("Market Maker Integration test completed")


def test_performance_analysis() -> None:
    """Тестирование анализа производительности."""
    logger.info("Testing Performance Analysis...")
    
    # Создание анализатора
    analyzer = LiquidityAnalyzer()
    
    # Множественные анализы
    simulator = OrderBookSimulator()
    for i in range(10):
        order_book = simulator.generate_normal_order_book("BTC/USDT")
        signal = analyzer.analyze_order_book(order_book)
    
    # Получение сводки
    summary = analyzer.get_analysis_summary()
    
    logger.info(f"Performance summary: {summary}")
    logger.info("Performance Analysis test completed")


def test_edge_cases() -> None:
    """Тестирование граничных случаев."""
    logger.info("Testing Edge Cases...")
    
    # Пустой ордербук
    empty_order_book = OrderBookSnapshot(
        symbol="BTC/USDT",
        bids=[],
        asks=[],
        timestamp=Timestamp(datetime.now())
    )
    
    analyzer = LiquidityAnalyzer()
    signal = analyzer.analyze_order_book(empty_order_book)
    
    logger.info(f"Empty order book analysis: {signal.risk_level}")
    logger.info("Edge Cases test completed")


async def main() -> None:
    """Основная функция для запуска примера."""
    config = ExampleConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        mode=ExampleMode.DEMO,
        risk_level=Decimal("0.5"),
        max_positions=5,
        enable_logging=True
    )
    
    example = LiquidityGravityExample(config)
    result = await example.execute_with_metrics()
    
    example.log_performance_summary(result)
    
    # Дополнительные тесты
    test_liquidity_gravity_model()
    test_risk_assessor()
    test_gravity_filter()
    test_market_maker_integration()
    test_performance_analysis()
    test_edge_cases()


if __name__ == "__main__":
    asyncio.run(main())
