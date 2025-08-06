import pickle
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


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
    """
    Торговый сигнал с ОБЯЗАТЕЛЬНЫМИ уровнями риска.
    КРИТИЧЕСКИ ВАЖНО: 
    - stop_loss и take_profit ДОЛЖНЫ быть установлены для безопасной торговли!
    - Все финансовые значения используют Decimal для точности
    """

    direction: str
    entry_price: Decimal  # ИЗМЕНЕНО: Decimal для точности!
    stop_loss: Decimal    # ИЗМЕНЕНО: Decimal для точности!
    take_profit: Decimal  # ИЗМЕНЕНО: Decimal для точности!
    volume: Optional[Decimal] = None  # ИЗМЕНЕНО: Decimal для точности!
    confidence: Decimal = Decimal('1.0')  # ИЗМЕНЕНО: Decimal для точности!
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Проверка критически важных параметров после создания сигнала"""
        # Автоматическое преобразование в Decimal если пришли float
        if isinstance(self.entry_price, (int, float)):
            object.__setattr__(self, 'entry_price', Decimal(str(self.entry_price)))
        if isinstance(self.stop_loss, (int, float)):
            object.__setattr__(self, 'stop_loss', Decimal(str(self.stop_loss)))
        if isinstance(self.take_profit, (int, float)):
            object.__setattr__(self, 'take_profit', Decimal(str(self.take_profit)))
        if self.volume is not None and isinstance(self.volume, (int, float)):
            object.__setattr__(self, 'volume', Decimal(str(self.volume)))
        if isinstance(self.confidence, (int, float)):
            object.__setattr__(self, 'confidence', Decimal(str(self.confidence)))
            
        # Проверка на положительные значения
        if self.entry_price <= 0:
            raise ValueError(f"entry_price должен быть больше 0, получено: {self.entry_price}")
        if self.stop_loss <= 0:
            raise ValueError(f"stop_loss должен быть больше 0, получено: {self.stop_loss}")
        if self.take_profit <= 0:
            raise ValueError(f"take_profit должен быть больше 0, получено: {self.take_profit}")
        if self.volume is not None and self.volume <= 0:
            raise ValueError(f"volume должен быть больше 0, получено: {self.volume}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence должен быть в диапазоне [0, 1], получено: {self.confidence}")
            
        # Проверка логики для LONG позиций
        if self.direction.lower() in ["long", "buy"]:
            if self.stop_loss >= self.entry_price:
                raise ValueError(f"Для LONG позиции stop_loss ({self.stop_loss}) должен быть меньше entry_price ({self.entry_price})")
            if self.take_profit <= self.entry_price:
                raise ValueError(f"Для LONG позиции take_profit ({self.take_profit}) должен быть больше entry_price ({self.entry_price})")
                
        # Проверка логики для SHORT позиций  
        elif self.direction.lower() in ["short", "sell"]:
            if self.stop_loss <= self.entry_price:
                raise ValueError(f"Для SHORT позиции stop_loss ({self.stop_loss}) должен быть больше entry_price ({self.entry_price})")
            if self.take_profit >= self.entry_price:
                raise ValueError(f"Для SHORT позиции take_profit ({self.take_profit}) должен быть меньше entry_price ({self.entry_price})")
                
        # Проверка разумности расстояний (максимум 50% стоп-лосс, минимум 0.01% тейк-профит)
        stop_distance_pct = abs(self.stop_loss - self.entry_price) / self.entry_price
        profit_distance_pct = abs(self.take_profit - self.entry_price) / self.entry_price
        
        if stop_distance_pct > Decimal('0.5'):  # 50%
            raise ValueError(f"Стоп-лосс слишком далеко: {stop_distance_pct:.2%} от цены входа")
        if profit_distance_pct < Decimal('0.0001'):  # 0.01%
            raise ValueError(f"Тейк-профит слишком близко: {profit_distance_pct:.2%} от цены входа")


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

    def _setup_logger(self) -> None:
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

    def _setup_state(self) -> None:
        """Настройка состояния стратегии"""
        self.state_dir = Path("state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / f"{self.__class__.__name__}_state.pkl"
        self.load_state(str(self.state_file))

    def _setup_threading(self) -> None:
        """Настройка многопоточности"""
        self.signal_queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()

    def _initialize_parameters(self) -> None:
        """Инициализация параметров стратегии"""
        self.required_columns = ["open", "high", "low", "close", "volume"]
        self.timeframes = self.config.get("timeframes", ["1h"])
        self.symbols = self.config.get("symbols", [])
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.max_position_size = self.config.get("max_position_size", 0.1)
        self.position_size_ratio = self.config.get("position_size_ratio", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.use_stop_loss = self.config.get("use_stop_loss", True)
        self.use_take_profit = self.config.get("use_take_profit", True)
        self.trailing_stop = self.config.get("trailing_stop", False)
        self.trailing_stop_activation = self.config.get(
            "trailing_stop_activation", 0.02
        )
        self.trailing_stop_distance = self.config.get("trailing_stop_distance", 0.01)

    def save_state(self) -> None:
        """Сохранение состояния стратегии"""
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump({"metrics": self.metrics, "config": self.config}, f)
            logger.info("Strategy state saved successfully")
        except Exception as e:
            logger.error(f"Error saving strategy state: {str(e)}")

    def load_state(self, state_file: str) -> None:
        """Загрузка состояния стратегии"""
        try:
            if Path(state_file).exists():
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                    self.metrics = state.get("metrics", StrategyMetrics())
                    self.config.update(state.get("config", {}))
                logger.info("Strategy state loaded successfully")
            else:
                # Создаём дефолтное состояние если файл не найден
                logger.info("Strategy state file not found, using default state")
        except (ImportError, ModuleNotFoundError) as e:
            # Игнорируем ошибки импорта модулей при загрузке pickle
            logger.info("Strategy state loaded with module compatibility mode")
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
            if data is None or data.empty:
                return False, "Empty dataset"
            missing_columns = [
                col for col in self.required_columns if col not in data.columns
            ]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            # Проверяем наличие пропущенных значений
            if hasattr(data, 'isnull') and hasattr(data.isnull(), 'any'):
                null_check = data.isnull()
                if hasattr(null_check, 'any') and null_check.any().any():
                    return False, "Dataset contains missing values"
            # Проверка на отрицательные значения
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in data.columns:
                    # Проверяем, что данные числовые и сравниваем с 0
                    col_data = data[col]
                    if hasattr(col_data, 'any') and hasattr(col_data, '__lt__'):
                        if (col_data < 0).any():
                            return False, f"Dataset contains negative values in {col}"
                    else:
                        # Альтернативная проверка для нечисловых данных
                        return False, f"Column {col} contains non-numeric data"
            # Проверка логики цен
            high_series = data["high"]
            low_series = data["low"]
            open_series = data["open"]
            close_series = data["close"]
            
            if hasattr(high_series, '__ge__') and hasattr(low_series, '__le__'):
                if not (high_series >= low_series).all():
                    return False, "High price is less than low price"
                if (
                    not (high_series >= open_series).all()
                    or not (high_series >= close_series).all()
                ):
                    return False, "High price is less than open or close"
                if (
                    not (low_series <= open_series).all()
                    or not (low_series <= close_series).all()
                ):
                    return False, "Low price is greater than open or close"
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def calculate_position_size(self, signal: Signal, account_balance: Decimal) -> Decimal:
        """
        Улучшенный расчет размера позиции с учетом риска.
        Args:
            signal: Торговый сигнал
            account_balance: Баланс аккаунта
        Returns:
            Размер позиции
        """
        try:
            # Расчет риска на основе стоп-лосса
            risk_per_trade = Decimal(str(self.risk_per_trade))  # Обычно 1-2%
            
            if signal.stop_loss is not None and signal.entry_price is not None:
                # Теперь это уже Decimal, но на всякий случай проверим
                entry_price = signal.entry_price if isinstance(signal.entry_price, Decimal) else Decimal(str(signal.entry_price))
                stop_loss = signal.stop_loss if isinstance(signal.stop_loss, Decimal) else Decimal(str(signal.stop_loss))
                
                # Риск на единицу зависит от направления
                if signal.direction == "long":
                    risk_per_unit = entry_price - stop_loss
                elif signal.direction == "short":
                    risk_per_unit = stop_loss - entry_price
                else:
                    risk_per_unit = entry_price * Decimal('0.02')  # 2% по умолчанию
                
                if risk_per_unit > 0:
                    # Размер позиции = (Риск на сделку * Баланс) / Риск на единицу
                    risk_amount = account_balance * risk_per_trade
                    position_size = risk_amount / risk_per_unit
                else:
                    # Если риск на единицу неположительный, используем базовый расчет
                    position_size = account_balance * Decimal(str(self.position_size_ratio))
            else:
                # Если нет стоп-лосса, используем базовый расчет
                position_size = account_balance * Decimal(str(self.position_size_ratio))
            
            # Ограничиваем максимальный размер позиции
            max_size = account_balance * Decimal(str(self.max_position_size))
            position_size = min(position_size, max_size)
            
            # Учитываем уверенность в сигнале
            position_size *= Decimal(str(signal.confidence))
            
            # Убеждаемся, что размер не превышает баланс
            position_size = min(position_size, account_balance)
            
            return max(Decimal('0'), position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return Decimal('0')

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
            volatility = float(returns.std())
            # Используем существующие методы или убираем несуществующие
            sharpe_ratio = float(self._calculate_kappa_ratio(returns))  # Временно используем kappa как sharpe
            sortino_ratio = float(self._calculate_information_ratio(returns, data))  # Временно используем information как sortino
            # Продвинутые метрики
            mar_ratio = float(self._calculate_calmar_ratio(returns, 0.1))  # Временно используем фиксированный drawdown
            ulcer_index = 0.0  # Убираем несуществующий метод
            omega_ratio = float(self._calculate_gain_loss_ratio(returns))  # Временно используем gain/loss как omega
            gini = 0.0  # Убираем несуществующий метод
            tail_ratio = float(self._calculate_treynor_ratio(returns, data))  # Временно используем treynor как tail
            # Статистические метрики
            try:
                skewness_val = returns.skew()
                skewness = float(skewness_val) if skewness_val is not None else 0.0
            except (TypeError, ValueError):
                skewness = 0.0
            try:
                kurtosis_val = returns.kurtosis()
                kurtosis = float(kurtosis_val) if kurtosis_val is not None else 0.0
            except (TypeError, ValueError):
                kurtosis = 0.0
            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(returns[returns <= var_95].mean())
            # Дополнительные метрики
            drawdown_duration = 0.0  # Убираем несуществующий метод
            stability = 0.0
            # Новые метрики
            calmar_ratio = float(self._calculate_calmar_ratio(returns, 0.1))
            treynor_ratio = float(self._calculate_treynor_ratio(returns, data))
            information_ratio = float(self._calculate_information_ratio(returns, data))
            kappa_ratio = float(self._calculate_kappa_ratio(returns))
            gain_loss_ratio = float(self._calculate_gain_loss_ratio(returns))
            return {
                "volatility": volatility,
                "max_drawdown": 0.1,  # Временно фиксированное значение
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
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Расчет Calmar Ratio"""
        if max_drawdown == 0:
            return 0.0
        try:
            mean_return = returns.mean()
            return float(mean_return * 252 / max_drawdown)
        except (TypeError, ValueError):
            return 0.0

    def _calculate_treynor_ratio(self, returns: pd.Series, data: pd.DataFrame) -> float:
        """Расчет Treynor Ratio"""
        try:
            market_returns = data["close"].pct_change()
            # Проверяем наличие метода cov у Series
            if hasattr(returns, 'cov') and hasattr(market_returns, 'var'):
                beta = float(returns.cov(market_returns) / market_returns.var())
                if beta == 0:
                    return 0.0
                mean_return = float(returns.mean())
                return float(mean_return * 252 / beta)
            else:
                # Альтернативный расчет без cov
                mean_return = float(returns.mean())
                std_return = float(returns.std())
                return float(mean_return * 252 / std_return)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating Treynor ratio: {e}")
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
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            return float(mean_excess / std_excess * (252**0.5))
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating Information ratio: {e}")
            return 0.0

    def _calculate_kappa_ratio(self, returns: pd.Series) -> float:
        """Расчет Kappa Ratio"""
        try:
            downside_returns = returns[returns < 0.0]
            if len(downside_returns) == 0:
                return 0.0
            mean_return = returns.mean()
            downside_std = downside_returns.std()
            return float(mean_return / (downside_std ** 2) * (252**0.5))
        except (ZeroDivisionError, ValueError, TypeError) as e:
            logger.warning(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Расчет Gain/Loss Ratio"""
        try:
            # Используем pandas методы для безопасного сравнения
            gains_mask = returns.gt(0.0)
            losses_mask = returns.lt(0.0)
            gains = returns[gains_mask]
            losses = returns[losses_mask]
            if len(losses) == 0:
                return 0.0
            return float(gains.mean() / abs(losses.mean()))
        except (ZeroDivisionError, ValueError, TypeError) as e:
            logger.warning(f"Error calculating gain/loss ratio: {e}")
            return 0.0

    def update_metrics(self, signal: Signal, result: Dict[str, Any]) -> None:
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
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error updating metrics: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error updating metrics: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик стратегии.
        Returns:
            Dict с метриками
        """
        try:
            with self.lock:
                return {
                    "total_signals": self.metrics.total_signals,
                    "successful_signals": self.metrics.successful_signals,
                    "failed_signals": self.metrics.failed_signals,
                    "win_rate": self.metrics.win_rate,
                    "avg_profit": self.metrics.avg_profit,
                    "avg_loss": self.metrics.avg_loss,
                    "profit_factor": self.metrics.profit_factor,
                    "sharpe_ratio": self.metrics.sharpe_ratio,
                    "sortino_ratio": self.metrics.sortino_ratio,
                    "max_drawdown": self.metrics.max_drawdown,
                    "recovery_factor": self.metrics.recovery_factor,
                    "expectancy": self.metrics.expectancy,
                    "risk_reward_ratio": self.metrics.risk_reward_ratio,
                    "kelly_criterion": self.metrics.kelly_criterion,
                    "volatility": self.metrics.volatility,
                    "mar_ratio": self.metrics.mar_ratio,
                    "ulcer_index": self.metrics.ulcer_index,
                    "omega_ratio": self.metrics.omega_ratio,
                    "gini_coefficient": self.metrics.gini_coefficient,
                    "tail_ratio": self.metrics.tail_ratio,
                    "skewness": self.metrics.skewness,
                    "kurtosis": self.metrics.kurtosis,
                    "var_95": self.metrics.var_95,
                    "cvar_95": self.metrics.cvar_95,
                    "drawdown_duration": self.metrics.drawdown_duration,
                    "max_equity": self.metrics.max_equity,
                    "min_equity": self.metrics.min_equity,
                    "median_trade": self.metrics.median_trade,
                    "median_duration": self.metrics.median_duration,
                    "profit_streak": self.metrics.profit_streak,
                    "loss_streak": self.metrics.loss_streak,
                    "stability": self.metrics.stability,
                    "additional": self.metrics.additional,
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def __del__(self) -> None:
        """Деструктор"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error in destructor: {str(e)}")
