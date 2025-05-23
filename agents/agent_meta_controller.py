import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

from agents.agent_market_maker_model import MarketMakerModelAgent
from agents.agent_market_regime import MarketRegimeAgent
from agents.agent_news import NewsAgent
from agents.agent_risk import RiskAgent
from agents.agent_whales import WhalesAgent
from core.strategy import Signal
from core.types import TradeDecision as CoreTradeDecision
from simulation.backtester import Backtester
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StrategyMetrics:
    """Класс для хранения метрик стратегии"""

    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade: float
    total_trades: int
    confidence: float
    last_update: datetime


@dataclass
class TradeDecision:
    """Структура торгового решения"""

    action: str  # buy/sell/hold
    confidence: float  # уверенность в решении
    position_size: float  # размер позиции
    stop_loss: float  # уровень стоп-лосса
    take_profit: float  # уровень тейк-профита
    source: str  # источник сигнала
    timestamp: datetime  # время принятия решения
    explanation: str  # объяснение решения


class MetaControllerAgent:
    """
    Мета-контроллер для управления стратегиями и агрегации сигналов.
    TODO: Вынести работу с файлами, стратегиями, сигналами, ретренингом в отдельные классы/модули (SRP).
    TODO: Проверить потокобезопасность асинхронных методов.
    """

    config: Dict[str, Any]
    market_regime_agent: MarketRegimeAgent
    risk_agent: RiskAgent
    whales_agent: WhalesAgent
    news_agent: NewsAgent
    market_maker_agent: MarketMakerModelAgent
    backtester: Optional[Backtester]
    strategies: Dict[str, Dict]
    active_strategies: Dict[str, str]
    last_retrain: Dict[str, datetime]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация мета-контроллера.
        :param config: конфигурация агента
        """
        self.config = config or {
            "min_win_rate": 0.55,
            "min_profit_factor": 1.5,
            "min_sharpe": 1.0,
            "min_trades": 100,
            "retrain_interval": 24,
            "max_drawdown": 0.15,
            "confidence_threshold": 0.7,
        }
        self.market_regime_agent = MarketRegimeAgent()
        self.risk_agent = RiskAgent()
        self.whales_agent = WhalesAgent()
        self.news_agent = NewsAgent()
        self.market_maker_agent = MarketMakerModelAgent()
        self.backtester: Optional[Backtester] = None
        self.strategies: Dict[str, Dict] = {}
        self.active_strategies: Dict[str, str] = {}
        self.last_retrain: Dict[str, datetime] = {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            "logs/meta_controller_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    async def initialize(self) -> None:
        """Инициализация бэктестера"""
        if self.backtester is None:
            from simulation.backtester import Backtester

            self.backtester = Backtester()
            # Backtester не требует асинхронной инициализации

    async def evaluate_strategies(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Оценка стратегий для торговой пары.
        :param symbol: торговая пара
        :return: решение о торговле
        """
        try:
            if not self._is_pair_ready(symbol):
                return None
            active_strategy = self.active_strategies.get(symbol)
            if not active_strategy:
                return None
            strategy_metrics = self.strategies.get(symbol, {}).get(active_strategy)
            if not strategy_metrics or not self._check_metrics(strategy_metrics):
                logger.warning(
                    f"Strategy {active_strategy} for {symbol} has poor metrics"
                )
                return None
            signal = await self._get_strategy_signal(symbol, active_strategy)
            if not signal:
                return None
            if signal.get("confidence", 0) < self.config["confidence_threshold"]:
                return None
            return signal
        except Exception as e:
            logger.error(f"Error evaluating strategies for {symbol}: {str(e)}")
            return None

    def _is_pair_ready(self, symbol: str) -> bool:
        """
        Проверка готовности торговой пары.
        :param symbol: торговая пара
        :return: True если пара готова к торговле
        """
        try:
            if symbol not in self.strategies:
                return False
            if symbol not in self.active_strategies:
                return False
            last_retrain = self.last_retrain.get(symbol)
            if not last_retrain:
                return False
            hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
            if hours_since_retrain >= self.config["retrain_interval"]:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking pair readiness for {symbol}: {str(e)}")
            return False

    def _check_metrics(self, metrics: StrategyMetrics) -> bool:
        """
        Проверка метрик стратегии.
        :param metrics: метрики стратегии
        :return: True если метрики удовлетворительны
        """
        try:
            return (
                metrics.win_rate >= self.config["min_win_rate"]
                and metrics.profit_factor >= self.config["min_profit_factor"]
                and metrics.sharpe_ratio >= self.config["min_sharpe"]
                and metrics.total_trades >= self.config["min_trades"]
                and metrics.max_drawdown <= self.config["max_drawdown"]
            )
        except Exception as e:
            logger.error(f"Error checking metrics: {str(e)}")
            return False

    async def _get_strategy_signal(
        self, symbol: str, strategy: str
    ) -> Optional[Dict[str, Any]]:
        """
        Получить сигнал от стратегии.
        :param symbol: торговая пара
        :param strategy: стратегия
        :return: сигнал
        """
        try:
            if self.backtester is None:
                await self.initialize()

            if self.backtester is None:
                logger.error("Failed to initialize backtester")
                return None

            # Получаем сигнал от соответствующего агента
            if strategy == "market_maker":
                return await self.market_maker_agent.predict_next_move(symbol)
            elif strategy == "market_regime":
                signals = await self.market_regime_agent.get_signals()
                return self._convert_signals_to_dict(signals)
            elif strategy == "whales":
                # Временно возвращаем пустой словарь, пока не реализован метод get_signals
                return {}
            elif strategy == "news":
                # Временно возвращаем пустой словарь, пока не реализован метод get_signals
                return {}
            else:
                logger.warning(f"Unknown strategy type: {strategy}")
                return None
        except Exception as e:
            logger.error(f"Error getting strategy signal for {symbol}: {str(e)}")
            return None

    def _convert_signals_to_dict(self, signals: List[Signal]) -> Dict[str, Any]:
        """
        Конвертирует список сигналов в словарь.
        :param signals: список сигналов
        :return: словарь с сигналами
        """
        if not signals:
            return {}

        latest_signal = signals[-1]
        return {
            "action": latest_signal.action,
            "price": latest_signal.price,
            "size": latest_signal.size,
            "stop_loss": latest_signal.stop_loss,
            "take_profit": latest_signal.take_profit,
            "metadata": latest_signal.metadata,
        }

    async def retrain_if_needed(self, symbol: str) -> None:
        """
        Ретрейнинг стратегий при необходимости.
        :param symbol: торговая пара
        """
        try:
            last_retrain = self.last_retrain.get(symbol)
            if last_retrain:
                hours_since_retrain = (
                    datetime.now() - last_retrain
                ).total_seconds() / 3600
                if hours_since_retrain < self.config["retrain_interval"]:
                    return
            await self._retrain_strategies(symbol)
            self.last_retrain[symbol] = datetime.now()
        except Exception as e:
            logger.error(f"Error retraining strategies for {symbol}: {str(e)}")

    async def _retrain_strategies(self, symbol: str) -> None:
        """
        Ретрейнинг всех стратегий для пары.
        :param symbol: торговая пара
        """
        try:
            historical_data = await self._get_historical_data(symbol)
            for strategy in self.strategies.get(symbol, {}):
                results = await self.backtester.run_backtest(
                    symbol=symbol, strategy=strategy, data=historical_data
                )
                # TODO: Привести структуру StrategyMetrics к единому виду по всему проекту
                self.strategies[symbol][strategy] = StrategyMetrics(
                    win_rate=results["win_rate"],
                    profit_factor=results["profit_factor"],
                    sharpe_ratio=results["sharpe_ratio"],
                    total_trades=results["total_trades"],
                    max_drawdown=results["max_drawdown"],
                    avg_trade=results.get("avg_trade", 0.0),
                    confidence=results.get("confidence", 0.0),
                    last_update=datetime.now(),
                )
            await self.activate_best_strategy(symbol)
        except Exception as e:
            logger.error(f"Error retraining strategies for {symbol}: {str(e)}")

    async def _get_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Получение исторических данных.
        :param symbol: торговая пара
        :return: DataFrame с историческими данными
        TODO: Реализовать получение исторических данных
        """
        try:
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def activate_best_strategy(self, symbol: str) -> None:
        """
        Активация лучшей стратегии для пары.
        :param symbol: торговая пара
        """
        try:
            if symbol not in self.strategies:
                return
            best_strategy = None
            best_score = -float("inf")
            for strategy, metrics in self.strategies[symbol].items():
                if not self._check_metrics(metrics):
                    continue
                score = (
                    metrics.win_rate * 0.3
                    + metrics.profit_factor * 0.3
                    + metrics.sharpe_ratio * 0.2
                    + (1 - metrics.max_drawdown) * 0.2
                )
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            if best_strategy:
                self.active_strategies[symbol] = best_strategy
                logger.info(f"Activated strategy {best_strategy} for {symbol}")
        except Exception as e:
            logger.error(f"Error activating best strategy for {symbol}: {str(e)}")

    def is_pair_ready_to_trade(self, symbol: str) -> bool:
        """
        Проверка готовности пары к торговле.
        :param symbol: торговая пара
        :return: True если пара готова к торговле
        """
        try:
            return (
                symbol in self.strategies
                and symbol in self.active_strategies
                and self._check_metrics(
                    self.strategies[symbol][self.active_strategies[symbol]]
                )
            )
        except Exception as e:
            logger.error(f"Error checking pair readiness for {symbol}: {str(e)}")
            return False

    def get_strategy_metrics(
        self, symbol: str, strategy: str
    ) -> Optional[StrategyMetrics]:
        """
        Получение метрик стратегии.
        :param symbol: торговая пара
        :param strategy: название стратегии
        :return: метрики стратегии
        """
        try:
            return self.strategies.get(symbol, {}).get(strategy)
        except Exception as e:
            logger.error(f"Error getting strategy metrics for {symbol}: {str(e)}")
            return None

    def get_active_strategy(self, symbol: str) -> Optional[str]:
        """
        Получение активной стратегии для пары.
        :param symbol: торговая пара
        :return: название активной стратегии
        """
        try:
            return self.active_strategies.get(symbol)
        except Exception as e:
            logger.error(f"Error getting active strategy for {symbol}: {str(e)}")
            return None

    def _create_hold_signal(self) -> Dict[str, Any]:
        """
        Создание сигнала о удержании позиции.
        :return: словарь сигнала
        """
        return {
            "action": "hold",
            "confidence": 0.0,
            "source": "bayesian_ensemble",
            "regime": "unknown",
            "strength": 0.0,
        }


class PairManager:
    """Manager for handling trading pairs and their configurations."""

    def __init__(self, config_path: str = "config/allowed_pairs.yaml"):
        """Initialize PairManager with configuration.

        Args:
            config_path: Path to the allowed pairs configuration file
        """
        self.config_path = config_path
        self.base_dir = Path("data/pairs")
        self._load_allowed_pairs()

    def _load_allowed_pairs(self) -> None:
        """Load allowed pairs from configuration file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                self.allowed_pairs = config.get("allowed_pairs", [])
            logger.info(f"Loaded {len(self.allowed_pairs)} allowed pairs")
        except Exception as e:
            logger.error(f"Error loading allowed pairs: {str(e)}")
            self.allowed_pairs = []

    def create_base_strategy(self, pair: str) -> None:
        """Create base strategy configuration for a trading pair.

        Args:
            pair: Trading pair symbol
        """
        try:
            base_strategy = {
                "regime": "trend",
                "entry_signal": "ema_crossover & rsi_oversold",
                "exit_signal": "rsi_overbought | price_below_ema",
                "stop_loss": "atr_trailing",
                "take_profit": "risk_reward_ratio: 2.0",
                "confidence_score": 0.3,
            }

            strategy_path = self.base_dir / pair / "strategy_profile.json"
            with open(strategy_path, "w") as f:
                json.dump(base_strategy, f, indent=2)

            logger.info(f"Created base strategy for pair {pair}")

        except Exception as e:
            logger.error(f"Error creating base strategy for pair {pair}: {str(e)}")

    def create_base_indicators(self, pair: str) -> None:
        """Create base indicators configuration for a trading pair.

        Args:
            pair: Trading pair symbol
        """
        try:
            indicators_config = {
                "timeframes": {"main": "1h", "confirm": "4h", "entry": "15m"},
                "indicators": [
                    {"name": "EMA", "params": [50, 200]},
                    {"name": "RSI", "params": [14]},
                    {"name": "ATR", "params": [14]},
                    {"name": "OBV", "params": []},
                ],
            }

            indicators_path = self.base_dir / pair / "indicators_config.yaml"
            with open(indicators_path, "w") as f:
                yaml.dump(indicators_config, f)

            logger.info(f"Created base indicators for pair {pair}")

        except Exception as e:
            logger.error(f"Error creating base indicators for pair {pair}: {str(e)}")

    def init_pair_structure(self, pair: str) -> None:
        """Initialize directory structure and files for a trading pair.

        Args:
            pair: Trading pair symbol (e.g., 'BTCUSDT')
        """
        try:
            # Create pair directory
            pair_dir = self.base_dir / pair
            pair_dir.mkdir(parents=True, exist_ok=True)

            # Create and initialize files
            if not (pair_dir / "indicators_config.yaml").exists():
                self.create_base_indicators(pair)

            if not (pair_dir / "strategy_profile.json").exists():
                self.create_base_strategy(pair)

            if not (pair_dir / "model_state.pkl").exists():
                (pair_dir / "model_state.pkl").touch()

            if not (pair_dir / "backtest_report.json").exists():
                with open(pair_dir / "backtest_report.json", "w") as f:
                    json.dump({}, f)

            if not (pair_dir / "meta_status.json").exists():
                meta_status = {
                    "WR": 0.0,
                    "is_trade_ready": False,
                    "is_trained": False,
                    "strategy_defined": True,
                    "last_update": None,
                    "strategy_status": "not_initialized",
                    "model_status": "not_initialized",
                    "backtest_status": "not_initialized",
                }
                with open(pair_dir / "meta_status.json", "w") as f:
                    json.dump(meta_status, f, indent=2)

            logger.info(f"Initialized structure for pair {pair}")

        except Exception as e:
            logger.error(f"Error initializing structure for pair {pair}: {str(e)}")

    def get_pair_status(self, pair: str) -> Dict[str, Any]:
        """
        Получить текущий статус торговой пары.
        :param pair: тикер пары
        :return: словарь со статусом
        """
        try:
            meta_path = self.base_dir / pair / "meta_status.json"
            if not meta_path.exists():
                return {
                    "WR": 0.0,
                    "is_trade_ready": False,
                    "is_trained": False,
                    "strategy_defined": False,
                }
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error getting status for pair {pair}: {str(e)}")
            return {
                "WR": 0.0,
                "is_trade_ready": False,
                "is_trained": False,
                "strategy_defined": False,
            }

    def get_active_pairs(self) -> List[str]:
        """
        Получить список пар, готовых к торговле.
        :return: список тикеров
        """
        active_pairs: List[str] = []
        for pair in self.allowed_pairs:
            try:
                self.init_pair_structure(pair)
                status = self.get_pair_status(pair)
                if (
                    status.get("WR", 0) >= 0.55
                    and status.get("is_trade_ready", False)
                    and status.get("is_trained", False)
                    and status.get("strategy_defined", False)
                    and status.get("strategy_status") == "ready"
                    and status.get("model_status") == "ready"
                    and status.get("backtest_status") == "completed"
                ):
                    active_pairs.append(pair)
            except Exception as e:
                logger.error(f"Error checking pair {pair}: {str(e)}")
                continue
        logger.info(f"Found {len(active_pairs)} active pairs")
        return active_pairs

    def update_pair_status(self, pair: str, status_updates: Dict[str, Any]) -> None:
        """
        Обновить статус торговой пары.
        :param pair: тикер пары
        :param status_updates: словарь с обновлениями статуса
        """
        try:
            meta_path = self.base_dir / pair / "meta_status.json"
            current_status = self.get_pair_status(pair)
            current_status.update(status_updates)
            current_status["last_update"] = datetime.now().isoformat()
            with open(meta_path, "w") as f:
                json.dump(current_status, f, indent=2)
            logger.info(f"Updated status for pair {pair}")
        except Exception as e:
            logger.error(f"Error updating status for pair {pair}: {str(e)}")

    def check_pair_requirements(self, pair: str) -> Dict[str, bool]:
        """
        Проверить, удовлетворяет ли пара всем требованиям для торговли.
        :param pair: тикер пары
        :return: словарь с результатами проверки
        """
        try:
            pair_dir = self.base_dir / pair
            status = self.get_pair_status(pair)
            requirements = {
                "has_model": (pair_dir / "model_state.pkl").exists(),
                "has_strategy": (pair_dir / "strategy_profile.json").exists(),
                "has_backtest": (pair_dir / "backtest_report.json").exists(),
                "has_indicators": (pair_dir / "indicators_config.yaml").exists(),
                "win_rate_ok": status.get("WR", 0) >= 0.55,
                "is_trade_ready": status.get("is_trade_ready", False),
                "is_trained": status.get("is_trained", False),
                "strategy_defined": status.get("strategy_defined", False),
            }
            return requirements
        except Exception as e:
            logger.error(f"Error checking requirements for pair {pair}: {str(e)}")
            return {}


class BayesianMetaController:
    """Контроллер для агрегации сигналов с использованием байесовского подхода."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация байесовского мета-контроллера.
        :param config: конфигурация
        """
        self.config = config or {}
        self.backtester = None
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.history: List[CoreTradeDecision] = []
        self.metadata: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Инициализация бэктестера"""
        if self.backtester is None:
            from simulation.backtester import Backtester

            self.backtester = Backtester()

    async def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Получение исторических данных.
        :param symbol: торговая пара
        :return: DataFrame с историческими данными
        """
        try:
            # Здесь должна быть реализация получения исторических данных
            # Например, через data_provider или другой источник данных
            return None
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None

    async def run_backtest(self, symbol: str, strategy: str) -> Dict[str, Any]:
        """
        Запуск бэктеста для стратегии.
        :param symbol: торговая пара
        :param strategy: стратегия
        :return: результаты бэктеста
        """
        try:
            if self.backtester is None:
                await self.initialize()

            if self.backtester is None:
                raise RuntimeError("Failed to initialize backtester")

            # Получаем исторические данные
            data = await self._get_historical_data(symbol)
            if data is None or data.empty:
                return {}

            # Запускаем бэктест
            if not hasattr(self.backtester, "run_backtest"):
                logger.error("Backtester does not have run_backtest method")
                return {}

            # Создаем конфигурацию для бэктеста
            config = {
                "symbol": symbol,
                "strategy": strategy,
                "start_date": self.config.get("start_date"),
                "end_date": self.config.get("end_date"),
            }

            results = await self.backtester.run_backtest(data, config)
            if results is None:
                return {}

            return dict(results) if isinstance(results, dict) else {}
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {str(e)}")
            return {}

    def evaluate_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Оценка производительности стратегий."""
        performance = {}
        for strategy, metrics in self.performance_metrics.items():
            if isinstance(metrics, dict) and "pnl" in metrics:
                pnl = float(metrics["pnl"])
                if pnl > 0:
                    performance[strategy] = {
                        "pnl": pnl,
                        "win_rate": float(metrics.get("win_rate", 0.0)),
                        "sharpe": float(metrics.get("sharpe", 0.0)),
                    }
        return performance
