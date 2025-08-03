"""
Промышленный симулятор рынка.
Полная реализация всех абстрактных методов.
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .types import (
    BaseMarketSimulator,
    MarketRegimeType,
    MarketSimulationConfig,
    SimulationMarketData,
    SimulationPrice,
    SimulationTimestamp,
    SimulationVolume,
    Symbol,
)


class MarketSimulator(BaseMarketSimulator):
    """Симулятор рынка с профессиональной реализацией."""

    def __init__(self, config: MarketSimulationConfig) -> None:
        super().__init__(config)
        self._setup_logger()
        self._setup_market_state()
        self._setup_random_seed()

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        log_path = self.config.logs_dir / "market_simulator.log"
        logger.add(
            log_path,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    def _setup_market_state(self) -> None:
        """Инициализация состояния рынка."""
        self.market_state: Dict[Symbol, Dict[str, Any]] = {}

    def _setup_random_seed(self) -> None:
        """Настройка случайного зерна."""
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    async def initialize(self) -> None:
        """Инициализация симулятора."""
        self.logger.info("Initializing MarketSimulator")
        # Создание директорий
        for dir_path in [
            self.config.data_dir,
            self.config.results_dir,
            self.config.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        # Инициализация состояния рынка
        await self._initialize_market_state()
        self.logger.info("MarketSimulator initialized successfully")

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        self.logger.info("Cleaning up MarketSimulator")
        # Сохранение финального состояния
        await self._save_market_state()
        # Очистка кэшей
        self.market_state.clear()
        self.logger.info("MarketSimulator cleanup completed")

    async def generate_market_data(
        self, symbol: Symbol, start_time: datetime, end_time: datetime
    ) -> List[SimulationMarketData]:
        """Генерация рыночных данных с профессиональной реализацией."""
        self.logger.info(
            f"Generating market data for {symbol} from {start_time} to {end_time}"
        )
        market_data = []
        current_time = start_time
        
        # Инициализируем состояние для символа, если его нет
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                "current_price": float(self.config.initial_price.value),  # type: ignore
                "current_volume": 1000.0,
                "current_regime": MarketRegimeType.UNKNOWN,
                "volatility": getattr(self.config, 'volatility', 0.02),
                "trend_strength": getattr(self.config, 'trend_strength', 0.01),
                "mean_reversion": getattr(self.config, 'mean_reversion', 0.1),
                "noise_level": getattr(self.config, 'noise_level', 0.005),
                "regime_counter": 0.0,
                "price_history": [],
                "volume_history": [],
                "regime_history": [],
            }
        
        # Генерация данных по временным интервалам
        while current_time <= end_time:
            # Обновление состояния рынка
            await self._update_market_state_internal(symbol)
            # Генерация OHLCV данных
            open_price = self._generate_open_price(symbol)
            high_price = self._generate_high_price(symbol, open_price)
            low_price = self._generate_low_price(symbol, open_price, high_price)
            close_price = self._generate_close_price(symbol, open_price, high_price, low_price)
            volume = self._generate_volume(symbol)
            # Создание рыночных данных
            market_point = SimulationMarketData(
                symbol=symbol,
                timestamp=SimulationTimestamp(current_time),
                open=SimulationPrice(Decimal(str(open_price))),
                high=SimulationPrice(Decimal(str(high_price))),
                low=SimulationPrice(Decimal(str(low_price))),
                close=SimulationPrice(Decimal(str(close_price))),
                volume=SimulationVolume(Decimal(str(volume))),
                regime=self.market_state[symbol]["current_regime"],
                volatility=self.market_state[symbol]["volatility"],
                trend_strength=self.market_state[symbol]["trend_strength"],
                momentum=self._calculate_momentum(symbol),
                metadata={
                    "regime_counter": self.market_state[symbol]["regime_counter"],
                    "noise_level": self.market_state[symbol]["noise_level"],
                    "mean_reversion": self.market_state[symbol]["mean_reversion"],
                },
            )
            market_data.append(market_point)
            # Обновление истории
            self.market_state[symbol]["price_history"].append(close_price)
            self.market_state[symbol]["volume_history"].append(volume)
            self.market_state[symbol]["regime_history"].append(
                self.market_state[symbol]["current_regime"]
            )
            # Ограничение размера истории
            max_history = 1000
            if len(self.market_state[symbol]["price_history"]) > max_history:
                self.market_state[symbol]["price_history"] = self.market_state[symbol]["price_history"][
                    -max_history:
                ]
                self.market_state[symbol]["volume_history"] = self.market_state[symbol][
                    "volume_history"
                ][-max_history:]
                self.market_state[symbol]["regime_history"] = self.market_state[symbol][
                    "regime_history"
                ][-max_history:]
            # Переход к следующему временному интервалу
            current_time += timedelta(minutes=1)
            # Проверка смены режима
            if getattr(self.config, 'regime_switching', False):
                await self._check_regime_switch(symbol)
        self.logger.info(
            f"Generated {len(market_data)} market data points for {symbol}"
        )
        return market_data

    async def update_market_state(
        self, symbol: Symbol, market_data: SimulationMarketData
    ) -> None:
        """Обновление состояния рынка."""
        await self._update_market_state_internal(symbol, market_data)

    async def get_market_regime(self, symbol: Symbol) -> MarketRegimeType:
        """Получение текущего режима рынка."""
        if symbol not in self.market_state:
            return MarketRegimeType.UNKNOWN
        regime = self.market_state[symbol]["current_regime"]
        if isinstance(regime, MarketRegimeType):
            return regime
        return MarketRegimeType.UNKNOWN

    async def _initialize_market_state(self) -> None:
        """Инициализация состояния рынка."""
        # Инициализируем состояние для каждого символа
        for symbol in self.config.symbols:
            self.market_state[symbol] = {
                "current_price": float(self.config.initial_price.value),  # type: ignore
                "current_volume": 1000.0,
                "current_regime": MarketRegimeType.SIDEWAYS,
                "volatility": getattr(self.config, 'volatility', 0.02),
                "trend_strength": getattr(self.config, 'trend_strength', 0.01),
                "mean_reversion": getattr(self.config, 'mean_reversion', 0.1),
                "noise_level": getattr(self.config, 'noise_level', 0.005),
                "regime_counter": 0.0,
                "price_history": [float(self.config.initial_price.value)],  # type: ignore
                "volume_history": [1000.0],
                "regime_history": [MarketRegimeType.SIDEWAYS],
            }
        self.logger.info("Market state initialized")

    async def _update_market_state_internal(
        self, symbol: Symbol, market_data: Optional[SimulationMarketData] = None
    ) -> None:
        """Внутреннее обновление состояния рынка."""
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                "current_price": float(self.config.initial_price.value),  # type: ignore
                "current_volume": 1000.0,
                "current_regime": MarketRegimeType.UNKNOWN,
                "volatility": getattr(self.config, 'volatility', 0.02),
                "trend_strength": getattr(self.config, 'trend_strength', 0.01),
                "mean_reversion": getattr(self.config, 'mean_reversion', 0.1),
                "noise_level": getattr(self.config, 'noise_level', 0.005),
                "regime_counter": 0.0,
                "price_history": [],
                "volume_history": [],
                "regime_history": [],
            }
            
        if market_data:
            # Обновление на основе внешних данных
            self.market_state[symbol]["current_price"] = float(market_data.close)
            self.market_state[symbol]["current_volume"] = float(market_data.volume)
            self.market_state[symbol]["current_regime"] = market_data.regime
            self.market_state[symbol]["volatility"] = market_data.volatility
            self.market_state[symbol]["trend_strength"] = market_data.trend_strength
        else:
            # Автоматическое обновление состояния
            await self._update_price_dynamics(symbol)
            await self._update_volume_dynamics(symbol)
            await self._update_regime_dynamics(symbol)

    async def _update_price_dynamics(self, symbol: Symbol) -> None:
        """Обновление динамики цен."""
        current_price = self.market_state[symbol]["current_price"]
        # Базовое движение цены
        price_change = self._calculate_price_change(symbol)
        # Применение тренда
        if self.market_state[symbol]["current_regime"] in [
            MarketRegimeType.TRENDING_UP,
            MarketRegimeType.TRENDING_DOWN,
        ]:
            trend_direction = (
                1
                if self.market_state[symbol]["current_regime"] == MarketRegimeType.TRENDING_UP
                else -1
            )
            price_change += (
                trend_direction * self.market_state[symbol]["trend_strength"] * 0.001
            )
        # Применение mean reversion
        if self.market_state[symbol]["current_regime"] == MarketRegimeType.SIDEWAYS:
            mean_price = float(self.config.initial_price.value)  # type: ignore
            reversion_force = (
                (mean_price - current_price)
                * self.market_state[symbol]["mean_reversion"]
                * 0.001
            )
            price_change += reversion_force
        # Применение волатильности
        volatility_impact = np.random.normal(0, self.market_state[symbol]["volatility"]) * 0.01
        price_change += volatility_impact
        # Применение шума
        noise_impact = np.random.normal(0, self.market_state[symbol]["noise_level"]) * 0.005
        price_change += noise_impact
        # Обновление цены
        new_price = current_price * (1 + price_change)
        self.market_state[symbol]["current_price"] = max(
            new_price, 0.000001
        )  # Минимальная цена

    async def _update_volume_dynamics(self, symbol: Symbol) -> None:
        """Обновление динамики объема."""
        current_volume = self.market_state[symbol]["current_volume"]
        # Базовое изменение объема
        volume_change = np.random.normal(0, 0.1)  # 10% стандартное отклонение
        # Влияние волатильности на объем
        if self.market_state[symbol]["volatility"] > 0.02:
            volume_change += 0.2  # Увеличение объема при высокой волатильности
        # Влияние режима на объем
        if self.market_state[symbol]["current_regime"] in [
            MarketRegimeType.BREAKOUT,
            MarketRegimeType.VOLATILE,
        ]:
            volume_change += 0.3  # Увеличение объема при важных событиях
        # Обновление объема
        new_volume = current_volume * (1 + volume_change)
        self.market_state[symbol]["current_volume"] = max(new_volume, 1.0)  # Минимальный объем

    async def _update_regime_dynamics(self, symbol: Symbol) -> None:
        """Обновление динамики режимов."""
        self.market_state[symbol]["regime_counter"] += 1
        # Проверка необходимости смены режима
        if self.market_state[symbol]["regime_counter"] >= getattr(self.config, 'regime_duration', 100):
            await self._switch_regime(symbol)
            # Обновление счетчика режима
            self.market_state[symbol]["regime_counter"] = 0.0

    async def _check_regime_switch(self, symbol: Symbol) -> None:
        """Проверка смены режима."""
        # Выбор нового режима на основе вероятности
        if random.random() < getattr(self.config, 'regime_probability', 0.1):
            await self._switch_regime(symbol)

    async def _switch_regime(self, symbol: Symbol) -> None:
        """Смена режима рынка."""
        current_regime = self.market_state[symbol]["current_regime"]
        # Определяем новый режим на основе вероятностей
        regime_weights = getattr(self.config, 'regime_transition_weights', {
            MarketRegimeType.SIDEWAYS: 0.4,
            MarketRegimeType.TRENDING_UP: 0.2,
            MarketRegimeType.TRENDING_DOWN: 0.2,
            MarketRegimeType.VOLATILE: 0.1,
            MarketRegimeType.BREAKOUT: 0.1,
        })
        # Исключение текущего режима для разнообразия
        if current_regime in regime_weights:
            del regime_weights[current_regime]
        # Нормализация вероятностей
        total_prob = sum(regime_weights.values())
        if total_prob > 0:
            regime_weights = {
                k: v / total_prob for k, v in regime_weights.items()
            }
            # Выбор нового режима
            rand_val = random.random()
            cumulative_prob = 0
            for regime, prob in regime_weights.items():
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    self.market_state[symbol]["current_regime"] = regime
                    break
        # Обновление параметров режима
        await self._update_regime_parameters(symbol)
        self.logger.info(
            f"Market regime switched to: {self.market_state[symbol]['current_regime']}"
        )

    async def _update_regime_parameters(self, symbol: Symbol) -> None:
        """Обновление параметров режима."""
        regime = self.market_state[symbol]["current_regime"]
        # Настройка параметров в зависимости от режима
        if regime == MarketRegimeType.TRENDING_UP:
            self.market_state[symbol]["trend_strength"] = min(
                self.market_state[symbol]["trend_strength"] * 1.5, 0.3
            )
            self.market_state[symbol]["volatility"] = max(
                self.market_state[symbol]["volatility"] * 0.8, 0.01
            )
        elif regime == MarketRegimeType.TRENDING_DOWN:
            self.market_state[symbol]["trend_strength"] = min(
                self.market_state[symbol]["trend_strength"] * 1.5, 0.3
            )
            self.market_state[symbol]["volatility"] = max(
                self.market_state[symbol]["volatility"] * 0.8, 0.01
            )
        elif regime == MarketRegimeType.SIDEWAYS:
            self.market_state[symbol]["trend_strength"] = max(
                self.market_state[symbol]["trend_strength"] * 0.5, 0.01
            )
            self.market_state[symbol]["volatility"] = min(
                self.market_state[symbol]["volatility"] * 1.2, 0.05
            )
        elif regime == MarketRegimeType.VOLATILE:
            self.market_state[symbol]["volatility"] = min(
                self.market_state[symbol]["volatility"] * 2.0, 0.1
            )
            self.market_state[symbol]["trend_strength"] = max(
                self.market_state[symbol]["trend_strength"] * 0.3, 0.01
            )
        elif regime == MarketRegimeType.BREAKOUT:
            self.market_state[symbol]["volatility"] = min(
                self.market_state[symbol]["volatility"] * 1.5, 0.08
            )
            self.market_state[symbol]["trend_strength"] = min(
                self.market_state[symbol]["trend_strength"] * 1.2, 0.25
            )

    def _calculate_price_change(self, symbol: Symbol) -> float:
        """Расчет изменения цены."""
        # Базовое случайное движение
        base_change = np.random.normal(0, 0.001)  # 0.1% стандартное отклонение
        # Влияние текущей волатильности
        volatility_impact = np.random.normal(0, self.market_state[symbol]["volatility"]) * 0.01
        return base_change + volatility_impact

    def _calculate_momentum(self, symbol: Symbol) -> float:
        """Расчет моментума."""
        price_history = self.market_state[symbol]["price_history"]
        if len(price_history) < 10:
            return 0.0
        # Моментум за последние 10 периодов
        return (price_history[-1] - price_history[-10]) / price_history[-10]

    def _generate_open_price(self, symbol: Symbol) -> float:
        """Генерация цены открытия."""
        current_price = self.market_state[symbol]["current_price"]
        # Цена открытия близка к предыдущей цене закрытия
        price_change = np.random.normal(0, self.market_state[symbol]["volatility"]) * 0.005
        result = current_price * (1 + price_change)
        return float(result)

    def _generate_high_price(self, symbol: Symbol, open_price: float) -> float:
        """Генерация максимальной цены."""
        # Максимальная цена выше цены открытия
        high_range = open_price * self.market_state[symbol]["volatility"] * 0.5
        high_addition = np.random.uniform(0, high_range)
        result = open_price + high_addition
        return float(result)

    def _generate_low_price(self, symbol: Symbol, open_price: float, high_price: float) -> float:
        """Генерация минимальной цены."""
        # Минимальная цена ниже цены открытия
        low_range = open_price * self.market_state[symbol]["volatility"] * 0.5
        low_subtraction = np.random.uniform(0, low_range)
        result = max(open_price - low_subtraction, 0.000001)
        return float(result)

    def _generate_close_price(
        self, symbol: Symbol, open_price: float, high_price: float, low_price: float
    ) -> float:
        """Генерация цены закрытия."""
        # Цена закрытия в диапазоне между минимальной и максимальной
        if high_price == low_price:
            return float(open_price)
        # Вероятность направления движения
        if self.market_state[symbol]["current_regime"] == MarketRegimeType.TRENDING_UP:
            up_prob = 0.7
        elif self.market_state[symbol]["current_regime"] == MarketRegimeType.TRENDING_DOWN:
            up_prob = 0.3
        else:
            up_prob = 0.5
        # Генерация цены закрытия
        if random.random() < up_prob:
            # Закрытие выше открытия
            close_price = open_price + np.random.uniform(0, high_price - open_price)
        else:
            # Закрытие ниже открытия
            close_price = open_price - np.random.uniform(0, open_price - low_price)
        result = max(close_price, low_price)
        return float(result)

    def _generate_volume(self, symbol: Symbol) -> float:
        """Генерация объема."""
        base_volume = self.market_state[symbol]["current_volume"]
        # Случайное изменение объема
        volume_change = np.random.normal(0, 0.2)  # 20% стандартное отклонение
        # Влияние волатильности
        if self.market_state[symbol]["volatility"] > 0.03:
            volume_change += 0.3
        # Влияние режима
        if self.market_state[symbol]["current_regime"] in [
            MarketRegimeType.BREAKOUT,
            MarketRegimeType.VOLATILE,
        ]:
            volume_change += 0.4
        new_volume = base_volume * (1 + volume_change)
        result = max(new_volume, 1.0)
        return float(result)

    async def _save_market_state(self) -> None:
        """Сохранение состояния рынка."""
        state_file = self.config.data_dir / "market_state.json"
        state_data = {}
        
        for symbol, state in self.market_state.items():
            state_data[str(symbol)] = {
                "current_price": state["current_price"],
                "current_volume": state["current_volume"],
                "current_regime": state["current_regime"].value,
                "volatility": state["volatility"],
                "trend_strength": state["trend_strength"],
                "regime_counter": state["regime_counter"],
                "timestamp": datetime.now().isoformat(),
            }
        
        import json
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        self.logger.info(f"Market state saved to {state_file}")
