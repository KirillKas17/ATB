import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from loguru import logger

from agents.agent_market_maker_model import MarketMakerAgent
from agents.agent_market_regime import MarketRegimeAgent
from agents.agent_meta_learning import MetaLearningAgent
from agents.agent_transformer_predictor import TransformerPredictor
from ml.transformer_predictor import TransformerPredictor
from strategies.adaptive_strategy_generator import AdaptiveStrategyGenerator
from strategies.manipulation_strategies import ManipulationStrategy
from strategies.trend_strategies import TrendStrategy
from utils.feature_engineering import generate_features


@dataclass
class TradingConfig:
    """Конфигурация торговли"""

    exchange: str
    symbol: str
    timeframe: str
    initial_capital: float
    max_position_size: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    max_open_trades: int
    max_daily_trades: int
    max_daily_loss: float
    max_drawdown: float
    data_dir: Path
    models_dir: Path
    log_dir: str = "logs"
    backup_dir: str = "backups"
    websocket_url: str = "ws://localhost:5000/ws"
    api_url: str = "http://localhost:5000/api"

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "TradingConfig":
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            return cls(
                exchange=config["exchange"]["name"],
                symbol=config["exchange"]["symbol"],
                timeframe=config["exchange"]["timeframe"],
                initial_capital=config["trading"]["initial_capital"],
                max_position_size=config["trading"]["max_position_size"],
                risk_per_trade=config["trading"]["risk_per_trade"],
                stop_loss=config["trading"]["stop_loss"],
                take_profit=config["trading"]["take_profit"],
                trailing_stop=config["trading"]["trailing_stop"],
                max_open_trades=config["trading"]["max_open_trades"],
                max_daily_trades=config["trading"]["max_daily_trades"],
                max_daily_loss=config["trading"]["max_daily_loss"],
                max_drawdown=config["trading"]["max_drawdown"],
                data_dir=Path(config["data_dir"]),
                models_dir=Path(config["models_dir"]),
                log_dir=config.get("log_dir", "logs"),
                backup_dir=config.get("backup_dir", "backups"),
                websocket_url=config.get("websocket_url", "ws://localhost:5000/ws"),
                api_url=config.get("api_url", "http://localhost:5000/api"),
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise


class TradingMetrics:
    """Класс для хранения метрик торговли"""

    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0

    def update(self, trade_result: Dict[str, Any]) -> None:
        """
        Обновление метрик после сделки.

        Args:
            trade_result: Результат сделки
        """
        try:
            self.total_trades += 1
            profit = float(trade_result.get("profit", 0.0))
            self.total_profit += profit

            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            self.current_balance += profit
            self.peak_balance = max(self.peak_balance, self.current_balance)
            self.current_drawdown = float(
                (self.peak_balance - self.current_balance) / self.peak_balance
            )
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

            self.win_rate = float(self.winning_trades / self.total_trades)
            self.profit_factor = float(
                abs(self.total_profit) / abs(self.losing_trades)
                if self.losing_trades > 0
                else 0
            )

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение текущих метрик.

        Returns:
            Dict[str, Any]: Словарь с метриками
        """
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_profit": float(self.total_profit),
            "max_drawdown": float(self.max_drawdown),
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "sharpe_ratio": float(self.sharpe_ratio),
            "current_drawdown": float(self.current_drawdown),
            "current_balance": float(self.current_balance),
        }


class TradingEngine:
    """Основной класс торгового движка"""

    def __init__(
        self,
        exchange: Any,
        market_maker: MarketMakerAgent,
        meta_learning: MetaLearningAgent,
        market_regime: MarketRegimeAgent,
        transformer_predictor: TransformerPredictor,
        strategy_generator: AdaptiveStrategyGenerator,
        manipulation_strategy: ManipulationStrategy,
        trend_strategy: TrendStrategy,
        window_size: int = 100,
        log_dir: str = "logs/trading",
    ):
        """
        Инициализация торгового движка.

        Args:
            exchange: Объект биржи
            market_maker: Агент маркет-мейкера
            meta_learning: Агент мета-обучения
            market_regime: Агент определения режима рынка
            transformer_predictor: Трансформер для предсказаний
            strategy_generator: Генератор стратегий
            manipulation_strategy: Стратегия манипуляций
            trend_strategy: Трендовая стратегия
            window_size: Размер окна для анализа
            log_dir: Директория для логов
        """
        self.exchange = exchange
        self.market_maker = market_maker
        self.meta_learning = meta_learning
        self.market_regime = market_regime
        self.transformer_predictor = transformer_predictor
        self.strategy_generator = strategy_generator
        self.manipulation_strategy = manipulation_strategy
        self.trend_strategy = trend_strategy
        self.window_size = window_size
        self.log_dir = log_dir
        self.metrics = TradingMetrics()
        self.current_strategy = None
        self.market_data = pd.DataFrame()
        self.is_running = False

    async def connect(self) -> bool:
        """
        Подключение к бирже.

        Returns:
            bool: True если подключение успешно
        """
        try:
            await self.exchange.connect()
            return True
        except Exception as e:
            logger.error(f"Error connecting to exchange: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """Отключение от биржи"""
        try:
            await self.exchange.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from exchange: {str(e)}")

    async def _process_market_data(self, data: Dict[str, Any]) -> None:
        """
        Обработка рыночных данных.

        Args:
            data: Рыночные данные
        """
        try:
            # Обновляем данные
            self.market_data = pd.DataFrame(data)
            if len(self.market_data) < self.window_size:
                return

            # Генерируем признаки
            features = generate_features(self.market_data)

            # Определяем режим рынка
            regime = await self.market_regime.predict(features)

            # Получаем предсказания
            predictions = await self.transformer_predictor.predict(
                features, self.window_size
            )

            # Обновляем мета-обучение
            await self.meta_learning.update(features, predictions, regime)

            # Выбираем стратегию
            self.current_strategy = self.strategy_generator.get_best_strategy(
                self.market_data
            )

            # Проверяем на манипуляции
            manipulation = self.manipulation_strategy.detect_manipulation(
                self.market_data
            )
            if manipulation:
                logger.warning("Market manipulation detected!")

            # Генерируем сигнал
            if self.current_strategy:
                signal = self.current_strategy.generate_signal(self.market_data)
                if signal:
                    await self._execute_trade(signal)

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    async def _execute_trade(self, signal: Dict[str, Any]) -> None:
        """
        Исполнение торгового сигнала.

        Args:
            signal: Торговый сигнал
        """
        try:
            # Проверяем сигнал
            if not signal or "action" not in signal:
                return

            # Получаем параметры
            action = str(signal["action"])
            price = float(signal["entry_price"])
            amount = float(signal["amount"])
            stop_loss = float(signal.get("stop_loss", 0.0))
            take_profit = float(signal.get("take_profit", 0.0))

            # Исполняем ордер
            order = await self.exchange.create_order(
                symbol="BTC/USDT",
                type="limit",
                side=action,
                amount=amount,
                price=price,
            )

            # Устанавливаем стоп-лосс и тейк-профит
            if stop_loss > 0:
                await self.exchange.create_order(
                    symbol="BTC/USDT",
                    type="stop",
                    side="sell" if action == "buy" else "buy",
                    amount=amount,
                    price=stop_loss,
                )

            if take_profit > 0:
                await self.exchange.create_order(
                    symbol="BTC/USDT",
                    type="limit",
                    side="sell" if action == "buy" else "buy",
                    amount=amount,
                    price=take_profit,
                )

            # Обновляем метрики
            self.metrics.update({"profit": 0.0})  # Временное значение

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")

    async def start(self) -> None:
        """Запуск торгового движка"""
        try:
            self.is_running = True
            await self.connect()

            while self.is_running:
                # Получаем данные
                data = await self.exchange.fetch_ohlcv(
                    "BTC/USDT", "1m", limit=self.window_size
                )
                await self._process_market_data(data)

        except Exception as e:
            logger.error(f"Error in trading engine: {str(e)}")
            self.is_running = False

    async def stop(self) -> None:
        """Остановка торгового движка"""
        self.is_running = False
        await self.disconnect()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение текущих метрик.

        Returns:
            Dict[str, Any]: Словарь с метриками
        """
        return self.metrics.get_metrics()


async def main() -> None:
    """Основная функция"""
    try:
        # Инициализация компонентов
        exchange = None  # TODO: Инициализировать биржу
        market_maker = MarketMakerAgent()
        meta_learning = MetaLearningAgent()
        market_regime = MarketRegimeAgent()
        transformer_predictor = TransformerPredictor()
        strategy_generator = AdaptiveStrategyGenerator([])
        manipulation_strategy = ManipulationStrategy()
        trend_strategy = TrendStrategy()

        # Создание торгового движка
        engine = TradingEngine(
            exchange=exchange,
            market_maker=market_maker,
            meta_learning=meta_learning,
            market_regime=market_regime,
            transformer_predictor=transformer_predictor,
            strategy_generator=strategy_generator,
            manipulation_strategy=manipulation_strategy,
            trend_strategy=trend_strategy,
        )

        # Запуск движка
        await engine.start()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
