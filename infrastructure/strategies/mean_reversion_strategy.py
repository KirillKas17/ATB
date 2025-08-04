from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class MeanReversionConfig:
    """Конфигурация стратегии возврата к среднему"""

    # Параметры возврата к среднему
    mean_period: int = 20  # Период для расчета среднего
    std_period: int = 20  # Период для расчета стандартного отклонения
    z_score_threshold: float = 2.0  # Порог Z-оценки
    min_reversion_periods: int = 3  # Минимальное количество периодов для подтверждения
    max_reversion_periods: int = (
        10  # Максимальное количество периодов для подтверждения
    )
    # Параметры входа
    entry_threshold: float = 0.01  # Порог для входа
    min_volume: float = 1000.0  # Минимальный объем
    min_volatility: float = 0.01  # Минимальная волатильность
    max_spread: float = 0.001  # Максимальный спред
    # Параметры выхода
    take_profit: float = 0.03  # Тейк-профит
    stop_loss: float = 0.015  # Стоп-лосс
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_step: float = 0.005  # Шаг трейлинг-стопа
    # Параметры управления рисками
    max_position_size: float = 1.0  # Максимальный размер позиции
    max_daily_trades: int = 10  # Максимальное количество сделок в день
    max_daily_loss: float = 0.02  # Максимальный дневной убыток
    risk_per_trade: float = 0.02  # Риск на сделку
    # Параметры мониторинга
    price_deviation_threshold: float = 0.002  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности
    reversion_strength_threshold: float = 0.7  # Порог силы возврата
    # Адаптивные параметры
    adaptive_mean: bool = True  # Адаптивное среднее
    adaptive_std: bool = True  # Адаптивное стандартное отклонение
    adaptive_z_score: bool = True  # Адаптивный Z-score
    adaptive_volatility: bool = True  # Адаптивная волатильность
    adaptive_position_sizing: bool = True  # Адаптивный размер позиции
    # Параметры для адаптации
    adaptation_window: int = 100  # Окно для адаптации
    adaptation_threshold: float = 0.1  # Порог для адаптации
    adaptation_speed: float = 0.1  # Скорость адаптации
    adaptation_method: str = "ewm"  # Метод адаптации (ewm, kalman, particle)
    # Параметры для фильтрации сигналов
    use_trend_filter: bool = True  # Использовать фильтр тренда
    use_volume_filter: bool = True  # Использовать фильтр объема
    use_volatility_filter: bool = True  # Использовать фильтр волатильности
    use_correlation_filter: bool = True  # Использовать фильтр корреляции
    # Параметры для фильтров
    trend_period: int = 50  # Период для определения тренда
    trend_threshold: float = 0.01  # Порог тренда
    volume_ma_period: int = 20  # Период для скользящего среднего объема
    volatility_ma_period: int = 20  # Период для скользящего среднего волатильности
    correlation_period: int = 20  # Период для расчета корреляции
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class MeanReversionStrategy(BaseStrategy):
    """Стратегия возврата к среднему (расширенная)"""

    def __init__(
        self, config: Optional[Union[Dict[str, Any], MeanReversionConfig]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        if isinstance(config, MeanReversionConfig):
            super().__init__(asdict(config))
            self._config = config
        elif isinstance(config, dict):
            super().__init__(config)
            self._config = MeanReversionConfig(**config)
        else:
            super().__init__(None)
            self._config = MeanReversionConfig()
        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.total_position: float = 0.0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных для стратегии возврата к среднему.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            dict[str, Any]: Результат анализа
        """
        try:
            is_valid, error = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")
            
            # Расчет базовых показателей
            volatility = self._calculate_volatility(data)
            spread = self._calculate_spread(data)
            liquidity = self._analyze_liquidity(data)
            reversion = self._analyze_reversion(data)
            
            # Расчет адаптивных параметров
            adaptive_params = self._calculate_adaptive_parameters(data)
            
            # Применение фильтров
            filters_passed = self._apply_filters(data, {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "reversion": reversion,
                "adaptive_params": adaptive_params
            })
            
            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "reversion": reversion,
                "adaptive_params": adaptive_params,
                "filters_passed": filters_passed,
                "timestamp": data.index[-1] if len(data) > 0 else None,
            }
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            return {}

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала на основе стратегии возврата к среднему.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[Signal]: Торговый сигнал или None
        """
        try:
            analysis = self.analyze(data)
            if not analysis:
                return None
            
            volatility = analysis["volatility"]
            spread = analysis["spread"]
            liquidity = analysis["liquidity"]
            reversion = analysis["reversion"]
            
            # Проверяем базовые условия
            if not self._check_basic_conditions(data, volatility, spread, liquidity, reversion):
                return None
            
            # Генерируем сигнал
            signal = self._generate_trading_signal(data, volatility, spread, liquidity, reversion)
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
        reversion: Dict[str, Any],
    ) -> bool:
        """
        Проверка базовых условий для торговли.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему
        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка волатильности
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            if volatility < self._config.min_volatility:
                return False
            # Проверка спреда
            spread = float(spread) if spread is not None and not pd.isna(spread) else 0.0
            if spread > self._config.max_spread:
                return False
            # Проверка ликвидности
            volume = liquidity.get("volume", 0.0)
            volume = float(volume) if volume is not None and not pd.isna(volume) else 0.0
            if volume < self._config.min_volume:
                return False
            depth = liquidity.get("depth", 0.0)
            depth = float(depth) if depth is not None and not pd.isna(depth) else 0.0
            if depth < self._config.liquidity_threshold:
                return False
            # Проверка силы возврата
            strength = reversion.get("strength", 0.0)
            strength = float(strength) if strength is not None and not pd.isna(strength) else 0.0
            if strength < self._config.reversion_strength_threshold:
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
        try:
            returns = data["close"].pct_change()
            volatility = returns.std()
            return float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_spread(self, data: pd.DataFrame) -> float:
        try:
            spread = (data["ask"] - data["bid"]).iloc[-1]
            return float(spread) if spread is not None and not pd.isna(spread) else 0.0
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            volume = data["volume"].iloc[-1]
            volume = float(volume) if volume is not None and not pd.isna(volume) else 0.0
            depth = (data["ask_volume"] + data["bid_volume"]).iloc[-1]  # type: ignore
            depth = float(depth) if depth is not None and not pd.isna(depth) else 0.0
            volume_spread = abs(data["ask_volume"] - data["bid_volume"]).iloc[-1]  # type: ignore
            volume_spread = float(volume_spread) if volume_spread is not None and not pd.isna(volume_spread) else 0.0
            return {"volume": volume, "depth": depth, "volume_spread": volume_spread}
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"volume": 0.0, "depth": 0.0, "volume_spread": 0.0}

    def _analyze_reversion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ возврата к среднему.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с показателями возврата к среднему
        """
        try:
            # Расчет скользящего среднего
            mean = data["close"].rolling(window=self._config.mean_period).mean()
            # Расчет стандартного отклонения
            std = data["close"].rolling(window=self._config.std_period).std()
            # Расчет Z-оценки
            z_score = (data["close"] - mean) / std
            current_z_score = z_score.iloc[-1] if hasattr(z_score, "iloc") else z_score[-1]  # type: ignore
            # Расчет силы возврата
            reversion_strength = abs(current_z_score) / z_score.std() if z_score.std() > 0 else 0.0
            # Расчет направления возврата
            reversion_direction = "up" if current_z_score < 0 else "down"
            # Расчет скорости возврата
            reversion_speed = abs(z_score.diff().iloc[-1]) if len(z_score) > 1 else 0.0  # type: ignore
            # Расчет количества периодов отклонения
            deviation_periods = self._calculate_deviation_periods(z_score)
            current_z_score = z_score.iloc[-1] if hasattr(z_score, "iloc") else z_score[-1]  # type: ignore
            current_mean = mean.iloc[-1] if hasattr(mean, "iloc") else mean[-1]  # type: ignore
            current_std = std.iloc[-1] if hasattr(std, "iloc") else std[-1]  # type: ignore
            
            return {
                "z_score": float(current_z_score) if current_z_score is not None and not pd.isna(current_z_score) else 0.0,
                "mean": float(current_mean) if current_mean is not None and not pd.isna(current_mean) else 0.0,
                "std": float(current_std) if current_std is not None and not pd.isna(current_std) else 0.0,
                "strength": float(reversion_strength) if reversion_strength is not None and not pd.isna(reversion_strength) else 0.0,
                "direction": reversion_direction,
                "speed": float(reversion_speed) if reversion_speed is not None and not pd.isna(reversion_speed) else 0.0,  # type: ignore
                "deviation_periods": deviation_periods,
            }
        except Exception as e:
            logger.error(f"Error analyzing reversion: {str(e)}")
            return {}

    def _calculate_deviation_periods(self, z_score: pd.Series) -> int:
        """
        Расчет количества периодов отклонения.
        Args:
            z_score: Серия Z-оценок
        Returns:
            int: Количество периодов отклонения
        """
        try:
            current_z = z_score.iloc[-1] if hasattr(z_score, "iloc") else z_score[-1]  # type: ignore
            if pd.isna(current_z):
                return 0
            periods = 0
            for z in reversed(z_score[:-1]):
                if pd.isna(z):
                    continue
                if (current_z > 0 and z > 0) or (current_z < 0 and z < 0):
                    periods += 1
                else:
                    break
            return periods
        except Exception as e:
            logger.error(f"Error calculating deviation periods: {str(e)}")
            return 0

    def _calculate_adaptive_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет адаптивных параметров.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с адаптивными параметрами
        """
        try:
            window = self._config.adaptation_window
            returns = data["close"].pct_change().dropna()
            # Адаптивное среднее
            if self._config.adaptive_mean:
                mean = returns.ewm(span=window, adjust=False).mean()
            else:
                mean = returns.rolling(window=self._config.mean_period).mean()
            # Адаптивное стандартное отклонение
            if self._config.adaptive_std:
                std = returns.ewm(span=window, adjust=False).std()
            else:
                std = returns.rolling(window=self._config.std_period).std()
            # Адаптивный Z-score
            if self._config.adaptive_z_score:
                z_score = (returns - mean) / std
                z_score = z_score.ewm(span=window, adjust=False).mean()
            else:
                z_score = (returns - mean) / std
            # Адаптивная волатильность
            if self._config.adaptive_volatility:
                volatility = returns.ewm(span=window, adjust=False).std()
            else:
                volatility = returns.rolling(window=self._config.std_period).std()
            return {
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "volatility": volatility,
            }
        except Exception as e:
            logger.error(f"Error calculating adaptive parameters: {str(e)}")
            return {}

    def _apply_filters(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> bool:
        """
        Применение фильтров к сигналу.
        Args:
            data: DataFrame с OHLCV данными
            analysis: Результаты анализа
        Returns:
            bool: Результат фильтрации
        """
        try:
            # Фильтр тренда
            if self._config.use_trend_filter:
                trend = self._calculate_trend(data)
                if trend["strength"] > self._config.trend_threshold:
                    return False
            # Фильтр объема
            if self._config.use_volume_filter:
                volume_ma = data["volume"].rolling(window=self._config.volume_ma_period).mean()
                current_volume = data["volume"].iloc[-1]
                volume_ma_last = volume_ma.iloc[-1] if hasattr(volume_ma, "iloc") else volume_ma[-1]
                if (current_volume is not None and not pd.isna(current_volume) and 
                    volume_ma_last is not None and not pd.isna(volume_ma_last) and 
                    current_volume < volume_ma_last * 0.8):
                    return False
            # Фильтр волатильности
            if self._config.use_volatility_filter:
                volatility_series = analysis.get("volatility")
                if isinstance(volatility_series, pd.Series):
                    volatility_ma = volatility_series.rolling(window=self._config.volatility_ma_period).mean()
                    current_volatility = volatility_series.iloc[-1]
                    volatility_ma_last = volatility_ma.iloc[-1]
                    if (current_volatility is not None and not pd.isna(current_volatility) and 
                        volatility_ma_last is not None and not pd.isna(volatility_ma_last) and 
                        current_volatility < volatility_ma_last * 0.5):
                        return False
                elif callable(volatility_series):
                    # Если volatility_series - это функция, вызываем её
                    volatility_data = volatility_series()
                    if isinstance(volatility_data, dict) and 'volatility' in volatility_data:
                        volatility_series_data: pd.Series = volatility_data['volatility']
                        if isinstance(volatility_series_data, pd.Series) and len(volatility_series_data) > 0:
                            volatility_ma: pd.Series = volatility_series_data.rolling(window=self._config.volatility_ma_period).mean()
                            current_volatility = volatility_series_data.iloc[-1] if hasattr(volatility_series_data, "iloc") else volatility_series_data.values[-1]
                            volatility_ma_last = volatility_ma.iloc[-1] if hasattr(volatility_ma, "iloc") else volatility_ma.values[-1]
                            if (current_volatility is not None and not pd.isna(current_volatility) and 
                                volatility_ma_last is not None and not pd.isna(volatility_ma_last) and 
                                current_volatility < volatility_ma_last * 0.5):
                                return False
            # Фильтр корреляции
            if self._config.use_correlation_filter:
                correlation = self._calculate_correlation(data)
                if abs(correlation) > 0.7:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return False

    def _calculate_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет тренда.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с параметрами тренда
        """
        try:
            # Расчет скользящего среднего
            ma = data["close"].rolling(window=self._config.trend_period).mean()
            # Расчет наклона тренда
            slope = (ma - ma.shift(1)) / ma.shift(1)
            # Расчет силы тренда
            trend_strength = abs(slope).mean()
            current_slope = slope.iloc[-1] if hasattr(slope, "iloc") else slope[-1]  # type: ignore
            # Определение направления тренда
            trend_direction = "up" if current_slope > 0 else "down"
            return {
                "direction": trend_direction,
                "strength": float(trend_strength) if trend_strength is not None and not pd.isna(trend_strength) else 0.0,
                "slope": float(current_slope) if current_slope is not None and not pd.isna(current_slope) else 0.0,
            }
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return {"direction": "unknown", "strength": 0.0, "slope": 0.0}

    def _calculate_correlation(self, data: pd.DataFrame) -> float:
        """
        Расчет корреляции.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            float: Коэффициент корреляции
        """
        try:
            # Расчет корреляции между ценой и объемом
            returns = data["close"].pct_change()
            volume_change = data["volume"].pct_change()
            correlation = returns.corr(volume_change)
            return float(correlation) if correlation is not None and not pd.isna(correlation) else 0.0
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала с учетом фильтров.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            # Получаем адаптивные параметры
            adaptive_params = self._calculate_adaptive_parameters(data)
            # Применяем фильтры
            if not self._apply_filters(data, {"volatility": volatility}):
                return None
            # Генерируем сигнал входа
            if not self.position:
                signal = self._generate_entry_signal(
                    data, volatility, spread, liquidity, reversion
                )
                if signal:
                    # Корректируем сигнал с учетом адаптивных параметров
                    if self._config.adaptive_position_sizing:
                        signal.volume = self._calculate_adaptive_position_size(
                            signal, adaptive_params
                        )
                    return signal
            # Генерируем сигнал выхода
            else:
                signal = self._generate_exit_signal(
                    data, volatility, spread, liquidity, reversion
                )
                if signal:
                    return signal
            return None
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1] if len(data["close"]) > 0 else 0.0  # type: ignore
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
            if current_price <= 0:
                return None
            # Проверяем условия для длинной позиции
            if (
                reversion["z_score"] < -self._config.z_score_threshold
                and reversion["direction"] == "up"
                and reversion["deviation_periods"] >= self._config.min_reversion_periods
                and reversion["deviation_periods"] <= self._config.max_reversion_periods
            ):
                # Проверяем объем
                current_volume = data["volume"].iloc[-1]
                volume_ma = data["volume"].rolling(window=20).mean().iloc[-1]
                if (current_volume is not None and not pd.isna(current_volume) and 
                    volume_ma is not None and not pd.isna(volume_ma) and 
                    current_volume > volume_ma):
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)
                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 - self._config.stop_loss)
                    take_profit = reversion["mean"]  # Цель - возврат к среднему
                    return Signal(
                        direction="long",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, reversion["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "reversion": reversion,
                        },
                    )
            # Проверяем условия для короткой позиции
            elif (
                reversion["z_score"] > self._config.z_score_threshold
                and reversion["direction"] == "down"
                and reversion["deviation_periods"] >= self._config.min_reversion_periods
                and reversion["deviation_periods"] <= self._config.max_reversion_periods
            ):
                # Проверяем объем
                current_volume = data["volume"].iloc[-1]
                volume_ma = data["volume"].rolling(window=20).mean().iloc[-1]
                if (current_volume is not None and not pd.isna(current_volume) and 
                    volume_ma is not None and not pd.isna(volume_ma) and 
                    current_volume > volume_ma):
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)
                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 + self._config.stop_loss)
                    take_profit = reversion["mean"]  # Цель - возврат к среднему
                    return Signal(
                        direction="short",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, reversion["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "reversion": reversion,
                        },
                    )
            return None
        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.
        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1] if len(data["close"]) > 0 else 0.0
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
            if current_price <= 0:
                return None
            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if self.stop_loss and current_price <= self.stop_loss:
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
                            "reversion": reversion,
                        },
                    )
                elif self.take_profit and current_price >= self.take_profit:
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
                            "reversion": reversion,
                        },
                    )
            elif self.position == "short":
                if self.stop_loss and current_price >= self.stop_loss:
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
                            "reversion": reversion,
                        },
                    )
                elif self.take_profit and current_price <= self.take_profit:
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
                            "reversion": reversion,
                        },
                    )
            # Проверяем трейлинг-стоп
            if self._config.trailing_stop and self.take_profit:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self._config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self._config.trailing_step)
            # Проверяем ослабление возврата
            if self._check_reversion_weakening(reversion):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "reversion_weakening",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                        "reversion": reversion,
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _check_reversion_weakening(self, reversion: Dict[str, Any]) -> bool:
        """
        Проверка ослабления возврата.
        Args:
            reversion: Показатели возврата к среднему
        Returns:
            bool: Результат проверки
        """
        try:
            if self.position == "long":
                # Проверяем ослабление восходящего возврата
                if reversion["direction"] == "down" or reversion["speed"] < 0:
                    return True
            else:
                # Проверяем ослабление нисходящего возврата
                if reversion["direction"] == "up" or reversion["speed"] < 0:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking reversion weakening: {str(e)}")
            return False

    def _update_position_state(self, signal: Signal, data: pd.DataFrame) -> None:
        """
        Обновление состояния позиции.
        Args:
            signal: Торговый сигнал
            data: DataFrame с OHLCV данными
        """
        try:
            if signal.direction in ["long", "short"]:
                self.position = signal.direction
                self.entry_price = signal.entry_price
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                if signal.volume is not None:
                    self.total_position += signal.volume
                self.daily_trades += 1
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                # Обновляем дневной P&L
                current_price = data["close"].iloc[-1]
                current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
                if self.position == "long" and self.entry_price is not None and current_price > 0:
                    self.daily_pnl += (current_price - self.entry_price) * self.total_position
                elif self.position == "short" and self.entry_price is not None and current_price > 0:
                    self.daily_pnl += (self.entry_price - current_price) * self.total_position
                self.position = None
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                self.total_position = 0.0
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
            volatility_factor = 1 / (1 + volatility)
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(
                position_size, self._config.max_position_size - self.total_position
            )
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _calculate_adaptive_position_size(
        self, signal: Signal, adaptive_params: Dict[str, Any]
    ) -> float:
        """
        Расчет адаптивного размера позиции.
        Args:
            signal: Торговый сигнал
            adaptive_params: Адаптивные параметры
        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            volatility_series = adaptive_params.get("volatility")
            if isinstance(volatility_series, pd.Series) and len(volatility_series) > 0:
                volatility = volatility_series.iloc[-1] if hasattr(volatility_series, "iloc") else volatility_series[-1]  # type: ignore
                volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            else:
                volatility = 0.0
                
            position_size = self._calculate_position_size(signal.entry_price, volatility)
            # Корректировка на Z-score
            z_score_series = adaptive_params.get("z_score")
            if isinstance(z_score_series, pd.Series) and len(z_score_series) > 0:
                z_score = z_score_series.iloc[-1] if hasattr(z_score_series, "iloc") else z_score_series[-1]  # type: ignore
                z_score = float(z_score) if z_score is not None and not pd.isna(z_score) else 0.0
                position_size *= 1 - abs(z_score) / 3.0
            # Корректировка на волатильность
            position_size *= 1 - volatility
            # Ограничение размера позиции
            position_size = min(position_size, self._config.max_position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating adaptive position size: {str(e)}")
            return 0.0
