"""
Резервная копия агента рыночного режима.
Включает:
- Классификацию рыночных режимов
- Анализ волатильности
- Определение трендов
- Сохранение состояния
"""

import json
import time
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from loguru import logger

from domain.types import Symbol
from domain.value_objects.timestamp import Timestamp


class MarketRegime(Enum):
    """Типы рыночных режимов."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


class VolatilityLevel(Enum):
    """Уровни волатильности."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RegimeAnalysis:
    """Результат анализа рыночного режима."""

    regime: MarketRegime
    confidence: float
    volatility_level: VolatilityLevel
    trend_strength: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeState:
    """Состояние рыночного режима."""

    symbol: Symbol
    timestamp: Timestamp
    current_regime: MarketRegime
    previous_regime: Optional[MarketRegime]
    regime_duration: int
    volatility_history: List[float]
    price_history: List[float]
    volume_history: List[float]


@runtime_checkable
class DataProvider(Protocol):
    """Протокол для провайдера данных."""

    def get_historical_data(self, symbol: Symbol, periods: int) -> pd.DataFrame:
        """Получить исторические данные."""
        ...

    def get_current_price(self, symbol: Symbol) -> float:
        """Получить текущую цену."""
        ...

    def get_volume(self, symbol: Symbol) -> float:
        """Получить объем."""
        ...


class RegimeClassifier(ABC):
    """Базовый класс для классификации рыночных режимов."""

    @abstractmethod
    def classify_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Классифицировать рыночный режим."""


class TechnicalRegimeClassifier(RegimeClassifier):
    """Технический классификатор рыночных режимов."""

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period

    def classify_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Классифицировать режим на основе технических индикаторов."""
        if len(data) < self.lookback_period:
            return self._default_analysis()
        # Расчет технических индикаторов
        sma_short = data["close"].rolling(20).mean()
        sma_long = data["close"].rolling(50).mean()
        volatility = data["close"].pct_change().rolling(20).std()
        rsi = self._calculate_rsi(data["close"])
        # Определение тренда
        current_price = data["close"].iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        # Определение режима
        if current_sma_short > current_sma_long and current_price > current_sma_short:
            regime = MarketRegime.TRENDING_UP
            trend_strength = (
                abs(current_sma_short - current_sma_long) / current_sma_long
            )
        elif current_sma_short < current_sma_long and current_price < current_sma_short:
            regime = MarketRegime.TRENDING_DOWN
            trend_strength = (
                abs(current_sma_short - current_sma_long) / current_sma_long
            )
        else:
            regime = MarketRegime.SIDEWAYS
            trend_strength = 0.0
        # Определение волатильности
        current_volatility = volatility.iloc[-1]
        if current_volatility < 0.01:
            volatility_level = VolatilityLevel.VERY_LOW
        elif current_volatility < 0.02:
            volatility_level = VolatilityLevel.LOW
        elif current_volatility < 0.04:
            volatility_level = VolatilityLevel.MEDIUM
        elif current_volatility < 0.08:
            volatility_level = VolatilityLevel.HIGH
        else:
            volatility_level = VolatilityLevel.VERY_HIGH
        # Расчет уверенности
        confidence = self._calculate_confidence(data, regime, volatility_level)
        # Определение уровней поддержки и сопротивления
        support, resistance = self._calculate_support_resistance(data)
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            support_level=support,
            resistance_level=resistance,
            metadata={
                "rsi": float(rsi.iloc[-1] if not rsi.empty else 50.0),
                "volatility": float(current_volatility),
                "sma_short": float(current_sma_short),
                "sma_long": float(current_sma_long),
            },
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитать RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # type: ignore
        loss = (delta.where(delta < 0, 0)).rolling(window=period).mean()  # type: ignore
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        regime: MarketRegime,
        volatility_level: VolatilityLevel,
    ) -> float:
        """Рассчитать уверенность в классификации."""
        # Базовая уверенность
        confidence = 0.7
        # Корректировка по волатильности
        if volatility_level in [VolatilityLevel.VERY_LOW, VolatilityLevel.LOW]:
            confidence += 0.1
        elif volatility_level in [VolatilityLevel.HIGH, VolatilityLevel.VERY_HIGH]:
            confidence -= 0.1
        # Корректировка по объему
        volume_ratio = (
            data["volume"].tail(10).mean() / data["volume"].tail(50).mean()
        )
        if volume_ratio > 1.2:
            confidence += 0.1
        elif volume_ratio < 0.8:
            confidence -= 0.1
        return min(1.0, max(0.0, confidence))

    def _calculate_support_resistance(
        self, data: pd.DataFrame
    ) -> tuple[Optional[float], Optional[float]]:
        """Рассчитать уровни поддержки и сопротивления."""
        if len(data) < 20:
            return None, None
        # Простой метод: минимумы и максимумы
        if hasattr(data, 'tail') and callable(data.tail):
            recent_data = data.tail(20)
            if hasattr(recent_data, 'min') and callable(recent_data.min):
                support = float(recent_data["low"].min())
                resistance = float(recent_data["high"].max())
                return support, resistance
        return None, None

    def _default_analysis(self) -> RegimeAnalysis:
        """Анализ по умолчанию."""
        return RegimeAnalysis(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            volatility_level=VolatilityLevel.MEDIUM,
            trend_strength=0.0,
        )


class MarketRegimeAgent:
    """Агент для анализа рыночных режимов."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.classifier = TechnicalRegimeClassifier()
        self.state: Dict[Symbol, RegimeState] = {}
        self.backup_path = Path("backups/market_regime_agent")
        self.backup_path.mkdir(parents=True, exist_ok=True)

    async def analyze_regime(
        self, symbol: Symbol, data_provider: DataProvider
    ) -> RegimeAnalysis:
        """Проанализировать рыночный режим."""
        try:
            # Получение данных
            historical_data = data_provider.get_historical_data(symbol, 100)
            # Классификация режима
            analysis = self.classifier.classify_regime(historical_data)
            # Обновление состояния
            await self._update_state(symbol, analysis, data_provider)
            # Сохранение резервной копии
            await self._save_backup()
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing regime for {symbol}: {e}")
            return self._default_analysis()

    async def _update_state(
        self, symbol: Symbol, analysis: RegimeAnalysis, data_provider: DataProvider
    ) -> None:
        """Обновить состояние агента."""
        current_price = data_provider.get_current_price(symbol)
        current_volume = data_provider.get_volume(symbol)
        if symbol not in self.state:
            self.state[symbol] = RegimeState(
                symbol=symbol,
                timestamp=Timestamp.now(),
                current_regime=analysis.regime,
                previous_regime=None,
                regime_duration=1,
                volatility_history=[analysis.metadata.get("volatility", 0.0)],
                price_history=[current_price],
                volume_history=[current_volume],
            )
        else:
            previous_state = self.state[symbol]
            # Проверка смены режима
            regime_duration = previous_state.regime_duration
            if analysis.regime != previous_state.current_regime:
                regime_duration = 1
            else:
                regime_duration += 1
            self.state[symbol] = RegimeState(
                symbol=symbol,
                timestamp=Timestamp.now(),
                current_regime=analysis.regime,
                previous_regime=previous_state.current_regime,
                regime_duration=regime_duration,
                volatility_history=previous_state.volatility_history[-49:]
                + [analysis.metadata.get("volatility", 0.0)],
                price_history=previous_state.price_history[-49:] + [current_price],
                volume_history=previous_state.volume_history[-49:] + [current_volume],
            )

    async def _save_backup(self) -> None:
        """Сохранить резервную копию состояния."""
        try:
            backup_data = {
                "agent_id": self.agent_id,
                "timestamp": Timestamp.now().isoformat(),  # type: ignore
                "state": {
                    symbol: {
                        "symbol": state.symbol,
                        "timestamp": state.timestamp.isoformat(),  # type: ignore[attr-defined]
                        "current_regime": state.current_regime.value,
                        "previous_regime": (
                            state.previous_regime.value
                            if state.previous_regime
                            else None
                        ),
                        "regime_duration": state.regime_duration,
                        "volatility_history": state.volatility_history,
                        "price_history": state.price_history,
                        "volume_history": state.volume_history,
                    }
                    for symbol, state in self.state.items()
                },
            }
            backup_file = (
                self.backup_path / f"backup_{self.agent_id}_{int(time.time())}.json"
            )
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Backup saved: {backup_file}")
        except Exception as e:
            logger.error(f"Error saving backup: {e}")

    async def load_backup(self, backup_file: Path) -> bool:
        """Загрузить резервную копию."""
        try:
            with open(backup_file, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
            # Восстановление состояния
            if hasattr(self.state, 'clear'):
                self.state.clear()
            for symbol_str, state_data in backup_data["state"].items():
                symbol: Symbol = Symbol(symbol_str)
                if hasattr(self.state, '__setitem__'):
                    self.state[symbol] = RegimeState(
                        symbol=symbol,
                        timestamp=Timestamp.from_iso_string(state_data["timestamp"]),
                        current_regime=MarketRegime(state_data["current_regime"]),
                        previous_regime=(
                            MarketRegime(state_data["previous_regime"])
                            if state_data["previous_regime"]
                            else None
                        ),
                        regime_duration=state_data["regime_duration"],
                        volatility_history=state_data["volatility_history"],
                        price_history=state_data["price_history"],
                        volume_history=state_data["volume_history"],
                    )
            logger.info(f"Backup loaded: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading backup: {e}")
            return False

    def get_regime_summary(self) -> Dict[str, Any]:
        """Получить сводку режимов."""
        summary: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "total_symbols": len(self.state),
            "regime_distribution": {},
            "average_confidence": 0.0,
            "symbols": {},
        }
        if not self.state:
            return summary
        # Распределение режимов
        regime_counts: Dict[str, int] = {}
        for symbol, state in self.state.items():
            regime = state.current_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            summary["symbols"][str(symbol)] = {
                "current_regime": regime,
                "regime_duration": state.regime_duration,
                "volatility": (
                    float(state.volatility_history[-1]) if state.volatility_history else 0.0
                ),
            }
        summary["regime_distribution"] = regime_counts
        return summary

    def _default_analysis(self) -> RegimeAnalysis:
        """Анализ по умолчанию."""
        return RegimeAnalysis(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            volatility_level=VolatilityLevel.MEDIUM,
            trend_strength=0.0,
        )

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        await self._save_backup()
        logger.info(f"Market Regime Agent {self.agent_id} cleanup completed")


# Фабричная функция для создания агента
def create_market_regime_agent(
    agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None
) -> MarketRegimeAgent:
    """Создать агента рыночного режима."""
    if agent_id is None:
        agent_id = str(uuid4())
    return MarketRegimeAgent(agent_id=agent_id, config=config)
