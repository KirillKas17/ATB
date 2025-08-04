"""
Пример использования эволюционного подхода для генерации и отбора торговых стратегий.

Демонстрирует:
- Интеграцию с новым модулем стратегий (domain.entities.strategy_new)
- Генерацию, мутацию и отбор стратегий
- Оценку производительности стратегий
- Логирование результатов
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.entities.strategy import Strategy, StrategyParameters, StrategyPerformance
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from shared.models.example_types import ExampleConfig, ExampleResult, ExampleMode
from examples.base_example import BaseExample, register_example


@register_example("strategy_evolution")
class StrategyEvolutionExample(BaseExample):
    """
    Пример эволюции торговых стратегий.
    """
    def __init__(self, config: Optional[ExampleConfig] = None):
        super().__init__(config)
        self.population: List[Strategy] = []
        self.performance_history: List[StrategyPerformance] = []
        self.market_data: Optional[pd.DataFrame] = None

    async def setup(self) -> None:
        logger.info("Setting up strategy evolution example...")
        self.market_data = self.generate_market_data()
        self.population = self.generate_initial_population(size=10)
        logger.info("Setup completed.")

    async def run(self) -> ExampleResult:
        logger.info("Running strategy evolution...")
        start_time = time.time()
        generations = 5
        best_strategy: Optional[Strategy] = None
        best_performance: Optional[StrategyPerformance] = None

        for gen in range(generations):
            logger.info(f"Generation {gen+1}/{generations}")
            performances = []
            for strategy in self.population:
                perf = self.evaluate_strategy(strategy)
                performances.append(perf)
                self.performance_history.append(perf)
            # Отбор лучших стратегий
            performances.sort(key=lambda p: p.total_pnl.value, reverse=True)
            best_ids = [p.strategy_id for p in performances[:5]]
            self.population = [s for s in self.population if s.id in best_ids]
            # Мутация и кроссовер
            mutated = self.mutate_population(self.population, size=5)
            self.population.extend(mutated)
            # Сохраняем лучшую стратегию
            if not best_performance or performances[0].total_pnl.value > best_performance.total_pnl.value:
                best_performance = performances[0]
                best_strategy = next((s for s in self.population if s.id == performances[0].strategy_id), None)
        duration = time.time() - start_time
        logger.info(f"Best strategy total_pnl: {best_performance.total_pnl.value if best_performance else 'N/A'}")
        return ExampleResult(
            success=True,
            duration_seconds=duration,
            trades_executed=best_performance.total_trades if best_performance else 0,
            total_pnl=best_performance.total_pnl.value if best_performance else Decimal("0"),
            max_drawdown=best_performance.max_drawdown.value if best_performance else Decimal("0"),
            sharpe_ratio=float(best_performance.sharpe_ratio) if best_performance else 0.0,
            metadata={
                "best_strategy": best_strategy.to_dict() if best_strategy else {},
                "total_pnl": float(best_performance.total_pnl.value) if best_performance else 0.0
            }
        )

    async def cleanup(self) -> None:
        logger.info("Cleanup after strategy evolution example.")

    def validate_prerequisites(self) -> bool:
        return self.config is not None and self.config.risk_level > 0

    def generate_market_data(self) -> pd.DataFrame:
        np.random.seed(42)
        periods = 1000
        start_price = 100.0
        prices = [start_price]
        for _ in range(periods):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=periods),
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
        df['symbol'] = self.config.symbols[0] if self.config.symbols else "BTCUSDT"
        df = df.set_index('timestamp')
        return df

    def generate_initial_population(self, size: int) -> List[Strategy]:
        population = []
        for _ in range(size):
            params = StrategyParameters()
            strategy = Strategy(parameters=params)
            population.append(strategy)
        return population

    def evaluate_strategy(self, strategy: Strategy) -> StrategyPerformance:
        # Простейшая симуляция: случайные метрики
        total_trades = np.random.randint(10, 100)
        total_pnl = Money(Decimal(str(np.random.normal(1000, 500))), Currency.USD)
        max_drawdown = Money(Decimal(str(np.random.uniform(0, 0.2))), Currency.USD)
        sharpe_ratio = Decimal(str(np.random.normal(1.5, 0.5)))
        return StrategyPerformance(
            strategy_id=strategy.id,
            total_trades=total_trades,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )

    def mutate_population(self, population: List[Strategy], size: int) -> List[Strategy]:
        mutated = []
        for _ in range(size):
            parent = np.random.choice(len(population))
            parent_strategy = population[parent]
            new_params = StrategyParameters()
            mutated.append(Strategy(parameters=new_params))
        return mutated


async def main():
    config = ExampleConfig(
        symbols=["BTCUSDT"],
        mode=ExampleMode.DEMO,
        risk_level=0.5,
        max_positions=5,
        enable_logging=True
    )
    example = StrategyEvolutionExample(config)
    result = await example.execute_with_metrics()
    example.log_performance_summary(result)
    return result

if __name__ == "__main__":
    asyncio.run(main()) 