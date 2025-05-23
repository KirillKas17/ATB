import pickle
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StrategyMetrics:
    """Метрики стратегии"""

    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    volatility: float = 0.0
    mar_ratio: float = 0.0
    ulcer_index: float = 0.0
    omega_ratio: float = 0.0
    gini_coefficient: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    drawdown_duration: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    median_trade: float = 0.0
    median_duration: float = 0.0
    profit_streak: int = 0
    loss_streak: int = 0
    stability: float = 0.0
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Торговый сигнал"""

    direction: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Базовый класс для всех торговых стратегий"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        self.config = config or {}
        self.metrics = StrategyMetrics()
        self._initialize_parameters()
        self._setup_logger()
        self._setup_state()
        self._setup_threading()

    def _setup_logger(self):
        """Настройка логгера"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / f"strategy_{self.__class__.__name__}_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )

    def _setup_state(self):
        """Настройка состояния стратегии"""
        self.state_dir = Path("state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / f"{self.__class__.__name__}_state.pkl"
        self.load_state()

    def _setup_threading(self):
        """Настройка многопоточности"""
        self.signal_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()

    def _initialize_parameters(self):
        """Инициализация параметров стратегии"""
        self.required_columns = ["open", "high", "low", "close", "volume"]
        self.timeframes = self.config.get("timeframes", ["1h"])
        self.symbols = self.config.get("symbols", [])
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.max_position_size = self.config.get("max_position_size", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.use_stop_loss = self.config.get("use_stop_loss", True)
        self.use_take_profit = self.config.get("use_take_profit", True)
        self.trailing_stop = self.config.get("trailing_stop", False)
        self.trailing_stop_activation = self.config.get(
            "trailing_stop_activation", 0.02
        )
        self.trailing_stop_distance = self.config.get("trailing_stop_distance", 0.01)

    def save_state(self):
        """Сохранение состояния стратегии"""
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump({"metrics": self.metrics, "config": self.config}, f)
            logger.info("Strategy state saved successfully")
        except Exception as e:
            logger.error(f"Error saving strategy state: {str(e)}")

    def load_state(self):
        """Загрузка состояния стратегии"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "rb") as f:
                    state = pickle.load(f)
                    self.metrics = state.get("metrics", StrategyMetrics())
                    self.config.update(state.get("config", {}))
                logger.info("Strategy state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading strategy state: {str(e)}")

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Optional[Signal] с сигналом или None
        """

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Валидация входных данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Tuple[bool, Optional[str]]: (валидность, сообщение об ошибке)
        """
        try:
            if data.empty:
                return False, "Empty dataset"
            missing_columns = [
                col for col in self.required_columns if col not in data.columns
            ]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            if data.isnull().any().any():
                return False, "Dataset contains missing values"
            if (data[["open", "high", "low", "close", "volume"]] < 0).any().any():
                return False, "Dataset contains negative values"
            if not (data["high"] >= data["low"]).all():
                return False, "High price is less than low price"
            if (
                not (data["high"] >= data["open"]).all()
                or not (data["high"] >= data["close"]).all()
            ):
                return False, "High price is less than open or close"
            if (
                not (data["low"] <= data["open"]).all()
                or not (data["low"] <= data["close"]).all()
            ):
                return False, "Low price is greater than open or close"
            return True, None
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False, str(e)

    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Расчет размера позиции.

        Args:
            signal: Торговый сигнал
            account_balance: Баланс счета

        Returns:
            float: Размер позиции
        """
        try:
            # Базовая позиция на основе риска
            risk_amount = account_balance * self.risk_per_trade
            if signal.stop_loss:
                risk_per_unit = abs(signal.entry_price - signal.stop_loss)
                position_size = risk_amount / risk_per_unit
            else:
                position_size = account_balance * self.max_position_size

            # Ограничиваем размер позиции
            position_size = min(position_size, account_balance * self.max_position_size)

            # Учитываем уверенность в сигнале
            position_size *= signal.confidence

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет расширенных метрик риска.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с метриками риска
        """
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) < 2:
                return {}

            # Базовые метрики
            volatility = returns.std()
            max_drawdown = self._calculate_max_drawdown(data)
            sharpe_ratio = self._calculate_sharpe_ratio(data)
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # Продвинутые метрики
            mar_ratio = self._calculate_mar_ratio(returns, max_drawdown)
            ulcer_index = self._calculate_ulcer_index(data)
            omega_ratio = self._calculate_omega_ratio(returns)
            gini = self._calculate_gini_coefficient(returns)
            tail_ratio = self._calculate_tail_ratio(returns)

            # Статистические метрики
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()

            # Дополнительные метрики
            drawdown_duration = self._calculate_drawdown_duration(data)
            stability = self._calculate_stability(data)

            # Новые метрики
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            treynor_ratio = self._calculate_treynor_ratio(returns, data)
            information_ratio = self._calculate_information_ratio(returns, data)
            kappa_ratio = self._calculate_kappa_ratio(returns)
            gain_loss_ratio = self._calculate_gain_loss_ratio(returns)

            return {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "mar_ratio": mar_ratio,
                "ulcer_index": ulcer_index,
                "omega_ratio": omega_ratio,
                "gini_coefficient": gini,
                "tail_ratio": tail_ratio,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "drawdown_duration": drawdown_duration,
                "stability": stability,
                "calmar_ratio": calmar_ratio,
                "treynor_ratio": treynor_ratio,
                "information_ratio": information_ratio,
                "kappa_ratio": kappa_ratio,
                "gain_loss_ratio": gain_loss_ratio,
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Расчет Calmar Ratio"""
        if max_drawdown == 0:
            return 0.0
        return returns.mean() * 252 / max_drawdown

    def _calculate_treynor_ratio(self, returns: pd.Series, data: pd.DataFrame) -> float:
        """Расчет Treynor Ratio"""
        try:
            market_returns = data["close"].pct_change()
            beta = returns.cov(market_returns) / market_returns.var()
            if beta == 0:
                return 0.0
            return returns.mean() * 252 / beta
        except:
            return 0.0

    def _calculate_information_ratio(
        self, returns: pd.Series, data: pd.DataFrame
    ) -> float:
        """Расчет Information Ratio"""
        try:
            market_returns = data["close"].pct_change()
            excess_returns = returns - market_returns
            if len(excess_returns) < 2:
                return 0.0
            return excess_returns.mean() / excess_returns.std() * (252**0.5)
        except:
            return 0.0

    def _calculate_kappa_ratio(self, returns: pd.Series) -> float:
        """Расчет Kappa Ratio"""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            return returns.mean() / (downside_returns.std() ** 2) * (252**0.5)
        except:
            return 0.0

    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Расчет Gain/Loss Ratio"""
        try:
            gains = returns[returns > 0]
            losses = returns[returns < 0]
            if len(losses) == 0:
                return 0.0
            return gains.mean() / abs(losses.mean())
        except:
            return 0.0

    def update_metrics(self, signal: Signal, result: Dict[str, Any]):
        """
        Обновление метрик стратегии.

        Args:
            signal: Торговый сигнал
            result: Результат торговли
        """
        try:
            with self.lock:
                self.metrics.total_signals += 1

                if result.get("profit", 0) > 0:
                    self.metrics.successful_signals += 1
                    self.metrics.profit_streak += 1
                    self.metrics.loss_streak = 0
                else:
                    self.metrics.failed_signals += 1
                    self.metrics.loss_streak += 1
                    self.metrics.profit_streak = 0

                # Обновляем средние значения
                if self.metrics.total_signals > 0:
                    self.metrics.win_rate = (
                        self.metrics.successful_signals / self.metrics.total_signals
                    )

                if self.metrics.successful_signals > 0:
                    self.metrics.avg_profit = (
                        self.metrics.avg_profit * (self.metrics.successful_signals - 1)
                        + result.get("profit", 0)
                    ) / self.metrics.successful_signals

                if self.metrics.failed_signals > 0:
                    self.metrics.avg_loss = (
                        self.metrics.avg_loss * (self.metrics.failed_signals - 1)
                        + result.get("profit", 0)
                    ) / self.metrics.failed_signals

                # Обновляем дополнительные метрики
                self.metrics.additional.update(result.get("additional_metrics", {}))

                # Сохраняем состояние
                self.save_state()

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик стратегии.

        Returns:
            Dict с метриками
        """
        try:
            return {
                "total_signals": self.metrics.total_signals,
                "successful_signals": self.metrics.successful_signals,
                "failed_signals": self.metrics.failed_signals,
                "win_rate": self.metrics.win_rate,
                "avg_profit": self.metrics.avg_profit,
                "avg_loss": self.metrics.avg_loss,
                "profit_streak": self.metrics.profit_streak,
                "loss_streak": self.metrics.loss_streak,
                "additional_metrics": self.metrics.additional,
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def __del__(self):
        """Деструктор"""
        try:
            self.executor.shutdown(wait=True)
            self.save_state()
        except:
            pass
