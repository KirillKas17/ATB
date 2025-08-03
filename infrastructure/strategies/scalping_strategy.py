from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from loguru import logger

from .base_strategy import BaseStrategy, Signal

# --- Импорт pandas
import pandas as pd


@dataclass
class ScalpingConfig:
    """Конфигурация скальпинг-стратегии"""

    # Параметры входа
    entry_threshold: float = 0.001  # Порог для входа
    min_volume: float = 1000.0  # Минимальный объем
    min_volatility: float = 0.0005  # Минимальная волатильность
    max_spread: float = 0.0003  # Максимальный спред
    min_tick_size: float = 0.0001  # Минимальный размер тика
    # Параметры выхода
    take_profit: float = 0.002  # Тейк-профит
    stop_loss: float = 0.001  # Стоп-лосс
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_step: float = 0.0005  # Шаг трейлинг-стопа
    max_holding_time: int = 300  # Максимальное время удержания (сек)
    # Параметры управления рисками
    max_position_size: float = 1.0  # Максимальный размер позиции
    max_daily_trades: int = 100  # Максимальное количество сделок в день
    max_daily_loss: float = 0.02  # Максимальный дневной убыток
    risk_per_trade: float = 0.01  # Риск на сделку
    # Параметры мониторинга
    price_deviation_threshold: float = 0.0005  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности
    order_book_depth: int = 10  # Глубина стакана
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1m"])
    log_dir: str = "logs"


class ScalpingStrategy(BaseStrategy):
    """Стратегия скальпинга"""

    def __init__(
        self, config: Optional[Union[Dict[str, Any], ScalpingConfig]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект ScalpingConfig
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, ScalpingConfig):
            config_dict = {
                "entry_threshold": config.entry_threshold,
                "min_volume": config.min_volume,
                "min_volatility": config.min_volatility,
                "max_spread": config.max_spread,
                "take_profit": config.take_profit,
                "stop_loss": config.stop_loss,
                "risk_per_trade": config.risk_per_trade,
                "max_position_size": config.max_position_size,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию для этого класса
        if isinstance(config, ScalpingConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = ScalpingConfig(**config)
        else:
            self._config = ScalpingConfig()
            
        # Состояние стратегии
        self.position: Optional[str] = None
        self.entry_time: Optional[datetime] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.total_position: float = 0.0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

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
                return False, "Data is None or empty"
            
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            
            if len(data) < 20:  # Минимальное количество данных для анализа
                return False, f"Insufficient data: {len(data)} < 20"
            
            # Проверка на NaN значения
            for col in required_columns:
                if data[col].isna().any():
                    return False, f"NaN values found in column: {col}"
            
            # Проверка на отрицательные цены
            for col in ["open", "high", "low", "close"]:
                if (data[col] <= 0).any():
                    return False, f"Non-positive values found in column: {col}"
            
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет метрик риска.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с метриками риска
        """
        try:
            # Расчет волатильности
            returns = data["close"].pct_change()
            volatility = returns.std()
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0 
            
            # Расчет максимальной просадки
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            max_drawdown = float(max_drawdown) if max_drawdown is not None and not pd.isna(max_drawdown) else 0.0 
            
            # Расчет VaR (Value at Risk)
            var_95 = returns.quantile(0.05)
            var_95 = float(var_95) if var_95 is not None and not pd.isna(var_95) else 0.0 
            
            # Расчет Sharpe ratio
            mean_return = returns.mean()
            mean_return = float(mean_return) if mean_return is not None and not pd.isna(mean_return) else 0.0 
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
            
            return {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "sharpe_ratio": sharpe_ratio,
                "mean_return": mean_return,
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "sharpe_ratio": 0.0,
                "mean_return": 0.0,
            }

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/scalping_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с результатами анализа
        """
        try:
            is_valid, error = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")
            # Расчет волатильности
            volatility = self._calculate_volatility(data)
            # Расчет спреда
            spread = self._calculate_spread(data)
            # Анализ ликвидности
            liquidity = self._analyze_liquidity(data)
            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)
            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "risk_metrics": risk_metrics,
                "timestamp": data.index[-1],
            }
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {}

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            analysis = self.analyze(data)
            if not analysis:
                return None
            volatility = analysis["volatility"]
            spread = analysis["spread"]
            liquidity = analysis["liquidity"]
            # Проверяем базовые условия
            if not self._check_basic_conditions(data, volatility, spread, liquidity):
                return None
            # Генерируем сигнал
            signal = self._generate_trading_signal(data, volatility, spread, liquidity)
            if signal:
                self._update_position_state(signal, data)
            return signal
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
    ) -> bool:
        """
        Проверка базовых условий для торговли.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка волатильности
            if volatility < self._config.min_volatility:
                return False
            # Проверка спреда
            if spread > self._config.max_spread:
                return False
            # Проверка ликвидности
            volume = liquidity.get("volume", 0.0)
            if volume < self._config.min_volume:
                return False
            depth = liquidity.get("depth", 0.0)
            if depth < self._config.liquidity_threshold:
                return False
            # Проверка размера позиции
            if self.total_position >= self._config.max_position_size:
                return False
            # Проверка количества сделок
            if self.daily_trades >= self._config.max_daily_trades:
                return False
            # Проверка дневного убытка
            if self.daily_pnl <= -self._config.max_daily_loss:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Расчет волатильности.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            float: Текущая волатильность
        """
        try:
            # Расчет волатильности как стандартного отклонения доходности
            returns = data["close"].pct_change()
            volatility = returns.std()
            return float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0 
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_spread(self, data: pd.DataFrame) -> float:
        """
        Расчет спреда.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            float: Текущий спред
        """
        try:
            # Расчет спреда как разницы между high и low
            high = data["high"].iloc[-1]
            low = data["low"].iloc[-1]
            close = data["close"].iloc[-1]
            
            high = float(high) if high is not None and not pd.isna(high) else 0.0 
            low = float(low) if low is not None and not pd.isna(low) else 0.0 
            close = float(close) if close is not None and not pd.isna(close) else 0.0 
            
            if close > 0:
                spread = (high - low) / close
            else:
                spread = 0.0
            return spread
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Анализ ликвидности.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с показателями ликвидности
        """
        try:
            # Расчет объема
            volume = data["volume"].iloc[-1]
            volume = float(volume) if volume is not None and not pd.isna(volume) else 0.0 
            
            # Расчет глубины рынка (используем объем как приближение)
            depth = volume * 2  # Приблизительная оценка
            
            # Расчет спреда объема (используем стандартное отклонение объема)
            volume_std = data["volume"].rolling(window=20).std().iloc[-1]
            volume_std = float(volume_std) if volume_std is not None and not pd.isna(volume_std) else 0.0 
            
            return {
                "volume": volume,
                "depth": depth,
                "volume_spread": volume_std,
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"volume": 0.0, "depth": 0.0, "volume_spread": 0.0}

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0 
            if current_price <= 0:
                return None
                
            # Проверяем вход в позицию
            if not self.position:
                return self._generate_entry_signal(data, volatility, spread, liquidity)
            # Проверяем выход из позиции
            return self._generate_exit_signal(data, volatility, spread, liquidity)
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0 
            if current_price <= 0:
                return None
                
            # Проверяем отклонение цены
            price_ma = data["close"].rolling(window=20).mean().iloc[-1]
            price_ma = float(price_ma) if price_ma is not None and not pd.isna(price_ma) else current_price 
            price_deviation = abs(current_price - price_ma) / current_price
            if price_deviation > self._config.price_deviation_threshold:
                return None
                
            # Проверяем отклонение объема
            current_volume = data["volume"].iloc[-1]
            current_volume = float(current_volume) if current_volume is not None and not pd.isna(current_volume) else 0.0 
            volume_ma = data["volume"].rolling(window=20).mean().iloc[-1]
            volume_ma = float(volume_ma) if volume_ma is not None and not pd.isna(volume_ma) else current_volume 
            
            if current_volume > 0:
                volume_deviation = abs(current_volume - volume_ma) / current_volume
            else:
                volume_deviation = 0.0
                
            if volume_deviation > self._config.volume_deviation_threshold:
                return None
                
            # Рассчитываем размер позиции
            volume = self._calculate_position_size(current_price, volatility)
            # Устанавливаем стоп-лосс и тейк-профит
            stop_loss = current_price * (1 - self._config.stop_loss)
            take_profit = current_price * (1 + self._config.take_profit)
            # Определяем направление
            prev_price = data["close"].iloc[-2]
            prev_price = float(prev_price) if prev_price is not None and not pd.isna(prev_price) else current_price 
            direction = "long" if current_price > prev_price else "short"
            
            return Signal(
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                confidence=min(1.0, volatility / self._config.min_volatility),
                timestamp=datetime.now(),
                metadata={
                    "volatility": volatility,
                    "spread": spread,
                    "liquidity": liquidity,
                    "price_deviation": price_deviation,
                    "volume_deviation": volume_deviation,
                },
            )
        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0 
            if current_price <= 0:
                return None
                
            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= (self.stop_loss or 0):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                        },
                    )
                elif current_price >= (self.take_profit or float('inf')):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                        },
                    )
            elif self.position == "short":
                if current_price >= (self.stop_loss or float('inf')):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                        },
                    )
                elif current_price <= (self.take_profit or 0):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                        },
                    )
            # Проверяем трейлинг-стоп
            if self._config.trailing_stop:
                if self.position == "long" and current_price > (self.take_profit or 0):
                    self.take_profit = current_price * (1 - self._config.trailing_step)
                elif self.position == "short" and current_price < (self.take_profit or float('inf')):
                    self.take_profit = current_price * (1 + self._config.trailing_step)
            # Проверяем время удержания
            if (
                self.entry_time
                and (datetime.now() - self.entry_time).total_seconds()
                > self._config.max_holding_time
            ):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "max_holding_time",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _update_position_state(self, signal: Signal, data: pd.DataFrame):
        """
        Обновление состояния позиции.
        Args:
            signal: Торговый сигнал
            data: DataFrame с OHLCV данными
        """
        try:
            if signal.direction in ["long", "short"]:
                self.position = signal.direction
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                self.entry_price = signal.entry_price
                if signal.volume is not None:
                    self.total_position += signal.volume
                self.entry_time = datetime.now()
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
            elif signal.direction == "close":
                # Обновляем дневной P&L
                if self.position == "long" and self.entry_price is not None:
                    current_price = data["close"].iloc[-1]
                    current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0 
                    self.daily_pnl += (current_price - self.entry_price) * self.total_position
                elif self.position == "short" and self.entry_price is not None:
                    current_price = data["close"].iloc[-1]
                    current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0 
                    self.daily_pnl += (self.entry_price - current_price) * self.total_position
                self.position = None
                self.stop_loss = None
                self.take_profit = None
                self.entry_price = None
                self.total_position = 0.0
                self.entry_time = None
        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def _calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Расчет размера позиции.
        Args:
            price: Текущая цена
            volatility: Текущая волатильность
        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self._config.risk_per_trade
            # Корректировка на волатильность
            volatility_factor = 1 / (1 + volatility) if volatility > 0 else 1.0
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(
                position_size, self._config.max_position_size - self.total_position
            )
            return max(0.0, position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
