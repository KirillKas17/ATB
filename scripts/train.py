import asyncio
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles  # type: ignore
import joblib
import pandas as pd
import psutil
import yaml
from data_loader import DataLoader
from dotenv import load_dotenv
from loguru import logger
from ml.adaptive_strategy_generator import AdaptiveStrategyGenerator
from ml.meta_learning import MetaLearning
from ml.pattern_discovery import PatternDiscovery
from simulation.backtester import BacktestConfig, Backtester
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""

    data_dir: Path
    models_dir: Path
    initial_capital: float
    market_regimes: List[str]
    validation_interval: int
    batch_size: int = 1000
    num_workers: int = -1
    early_stopping_rounds: int = 10
    min_samples: int = 100
    max_samples: int = 10000
    metrics_window: int = 24
    backup_dir: str = "backups"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "TrainingConfig":
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            return cls(
                data_dir=Path(config["data_dir"]),
                models_dir=Path(config["models_dir"]),
                initial_capital=config["training"]["initial_capital"],
                market_regimes=config["training"]["market_regimes"],
                validation_interval=config["training"]["validation_interval"],
                batch_size=config["training"].get("batch_size", 1000),
                num_workers=config["training"].get("num_workers", -1),
                early_stopping_rounds=config["training"].get(
                    "early_stopping_rounds", 10
                ),
                min_samples=config["training"].get("min_samples", 100),
                max_samples=config["training"].get("max_samples", 10000),
                metrics_window=config["training"].get("metrics_window", 24),
                backup_dir=config["training"].get("backup_dir", "backups"),
                log_dir=config["training"].get("log_dir", "logs"),
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise


@dataclass
class TrainingMetrics:
    """Метрики обучения"""

    regime: str
    start_time: datetime
    end_time: datetime
    training_time: float
    memory_usage: float
    cpu_usage: float
    patterns_count: int
    strategies_count: int
    validation_metrics: Dict
    success: bool
    error: Optional[str] = None


class ModelTrainer:
    """Тренер моделей"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Инициализация тренера"""
        self.config = config or TrainingConfig.from_yaml()
        self.metrics_history: List[TrainingMetrics] = []
        self._train_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # Создание директорий
        for dir_path in [
            self.config.models_dir,
            self.config.backup_dir,
            self.config.log_dir,
        ]:
            Path(str(dir_path)).mkdir(parents=True, exist_ok=True)

    def _get_system_metrics(self) -> Dict:
        """Получение системных метрик"""
        return {
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
        }

    def _handle_shutdown(self, signum, frame):
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения")
        self._executor.shutdown(wait=False)

    async def train_regime(
        self, regime: str, market_data: pd.DataFrame
    ) -> TrainingMetrics:
        """Обучение для конкретного режима"""
        try:
            start_time = datetime.now()
            start_metrics = self._get_system_metrics()

            logger.info(f"Обучение для режима {regime}")

            # Фильтрация данных по режиму
            regime_data = market_data[market_data["regime"] == regime]

            if len(regime_data) < self.config.min_samples:
                raise ValueError(f"Недостаточно данных для режима {regime}")

            # Ограничение размера выборки
            if len(regime_data) > self.config.max_samples:
                regime_data = regime_data.sample(
                    n=self.config.max_samples, random_state=42
                )

            # Инициализация компонентов
            pattern_discovery = PatternDiscovery()
            meta_learning = MetaLearning()
            strategy_generator = AdaptiveStrategyGenerator()
            backtester = Backtester(
                config=BacktestConfig(
                    start_date=regime_data.index[0],
                    end_date=regime_data.index[-1],
                    initial_balance=self.config.initial_capital,
                    position_size=self.config.batch_size,
                    max_positions=10,
                    stop_loss=0.02,
                    take_profit=0.04,
                    commission=0.001,
                    slippage=0.001,
                    data_dir=self.config.data_dir,
                    models_dir=self.config.models_dir,
                    log_dir=self.config.log_dir,
                    backup_dir=self.config.backup_dir,
                    metrics_window=self.config.metrics_window,
                    min_samples=self.config.min_samples,
                    max_samples=self.config.max_samples,
                    num_threads=self.config.num_workers,
                    validation_split=0.2,
                    use_patterns=True,
                    use_adaptation=True,
                    use_regime_detection=True,
                )
            )

            # Обучение паттернов
            patterns = await pattern_discovery.discover_patterns(
                data=regime_data, regime=regime
            )

            # Обучение мета-модели
            meta_model = await meta_learning.train(
                data=regime_data, patterns=patterns, regime=regime
            )

            # Генерация стратегий
            strategies = await strategy_generator.generate_strategies(
                data=regime_data,
                patterns=patterns,
                meta_model=meta_model,
                regime=regime,
            )

            # Валидация стратегий
            validation_results = await backtester.validate_strategies(
                strategies=strategies, data=regime_data, regime=regime
            )

            # Сохранение результатов
            await self.save_training_results(
                regime=regime,
                patterns=patterns,
                meta_model=meta_model,
                strategies=strategies,
                validation_results=validation_results,
            )

            # Расчет метрик
            end_time = datetime.now()
            end_metrics = self._get_system_metrics()
            training_time = (end_time - start_time).total_seconds()

            metrics = TrainingMetrics(
                regime=regime,
                start_time=start_time,
                end_time=end_time,
                training_time=training_time,
                memory_usage=end_metrics["memory_usage"]
                - start_metrics["memory_usage"],
                cpu_usage=end_metrics["cpu_usage"],
                patterns_count=len(patterns),
                strategies_count=len(strategies),
                validation_metrics=validation_results,
                success=True,
            )

            self.metrics_history.append(metrics)

            # Вывод статистики
            self.print_training_statistics(validation_results, regime)

            return metrics

        except Exception as e:
            logger.error(f"Ошибка обучения для режима {regime}: {str(e)}")
            metrics = TrainingMetrics(
                regime=regime,
                start_time=start_time,
                end_time=datetime.now(),
                training_time=(datetime.now() - start_time).total_seconds(),
                memory_usage=0,
                cpu_usage=0,
                patterns_count=0,
                strategies_count=0,
                validation_metrics={},
                success=False,
                error=str(e),
            )
            self.metrics_history.append(metrics)
            raise

    async def save_training_results(
        self,
        regime: str,
        patterns: Dict,
        meta_model: Dict,
        strategies: List[Dict],
        validation_results: Dict,
    ):
        """Сохранение результатов обучения"""
        try:
            async with self._train_lock:
                regime_dir = self.config.models_dir / regime
                regime_dir.mkdir(exist_ok=True)

                # Создание бэкапа
                backup_dir = Path(
                    f"{self.config.backup_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                backup_dir.mkdir(parents=True, exist_ok=True)

                # Сохранение паттернов
                patterns_path = regime_dir / "patterns.joblib"
                joblib.dump(patterns, patterns_path)
                joblib.dump(patterns, backup_dir / "patterns.joblib")

                # Сохранение мета-модели
                meta_model_path = regime_dir / "meta_model.joblib"
                joblib.dump(meta_model, meta_model_path)
                joblib.dump(meta_model, backup_dir / "meta_model.joblib")

                # Сохранение стратегий
                strategies_path = regime_dir / "strategies.joblib"
                joblib.dump(strategies, strategies_path)
                joblib.dump(strategies, backup_dir / "strategies.joblib")

                # Сохранение результатов валидации
                validation_path = regime_dir / "validation_results.json"
                async with aiofiles.open(validation_path, "w") as f:
                    await f.write(json.dumps(validation_results, indent=2))
                async with aiofiles.open(
                    backup_dir / "validation_results.json", "w"
                ) as f:
                    await f.write(json.dumps(validation_results, indent=2))

                logger.info(f"Результаты сохранены для режима {regime}")

        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {str(e)}")
            raise

    def print_training_statistics(self, validation_results: Dict, regime: str):
        """Вывод статистики обучения"""
        print(f"\nСтатистика обучения для режима {regime}:")
        print(f"Всего стратегий: {len(validation_results['strategies'])}")
        print(f"Средний винрейт: {validation_results['avg_win_rate']:.2%}")
        print(f"Средний профит-фактор: {validation_results['avg_profit_factor']:.2f}")
        print(f"Лучший винрейт: {validation_results['best_win_rate']:.2%}")
        print(f"Лучший профит-фактор: {validation_results['best_profit_factor']:.2f}")
        print(f"Всего сделок: {validation_results['total_trades']}")
        print(
            f"Средняя длительность сделки: {validation_results['avg_trade_duration']}"
        )

    async def train(self):
        """Основная функция обучения"""
        try:
            # Регистрация обработчиков сигналов
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            # Инициализация загрузчика данных
            data_loader = DataLoader(self.config.data_dir)

            # Загрузка данных
            logger.info("Загрузка рыночных данных...")
            market_data = await data_loader.load_all_data()

            # Обучение для каждого режима
            for regime in tqdm(
                self.config.market_regimes, desc="Обучение для рыночных режимов"
            ):
                await self.train_regime(regime, market_data)

            # Сохранение метрик
            metrics_file = Path(f"{self.config.log_dir}/training_metrics.json")
            async with aiofiles.open(metrics_file, "w") as f:
                await f.write(
                    json.dumps(
                        [m.__dict__ for m in self.metrics_history],
                        indent=2,
                        default=str,
                    )
                )

            logger.info("Обучение успешно завершено")

        except Exception as e:
            logger.error(f"Ошибка обучения: {str(e)}")
            raise
        finally:
            self._executor.shutdown(wait=True)


async def main():
    """Основная функция"""
    try:
        # Загрузка переменных окружения
        load_dotenv()

        # Инициализация тренера
        trainer = ModelTrainer()

        # Запуск обучения
        await trainer.train()

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/training_{time}.log", rotation="1 day", retention="7 days", level="INFO"
    )

    # Запуск асинхронного main
    asyncio.run(main())
