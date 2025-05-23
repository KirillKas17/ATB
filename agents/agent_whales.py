from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.agent_market_regime import MarketRegimeAgent
from core.types import Signal
from exchange.market_data import MarketData
from utils.indicators import calculate_volume_profile
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WhaleActivityType(Enum):
    ORDER_BOOK = "order_book"
    VOLUME = "volume"
    IMPULSE = "impulse"
    DOMINANCE = "dominance"


@dataclass
class WhaleActivity:
    """Класс для хранения информации о активности китов"""

    timestamp: pd.Timestamp
    pair: str
    activity_type: WhaleActivityType
    confidence: float
    impact: float
    details: Dict[str, Any]
    priority: int = 0


class IDataProvider(ABC):
    @abstractmethod
    async def get_market_data(self, pair: str, interval: str = "1m") -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_order_book(self, pair: str) -> Dict[str, Any]:
        pass


class DefaultDataProvider(IDataProvider):
    """Провайдер данных по умолчанию"""

    def __init__(self, symbol: str = "BTC/USDT", interval: str = "1h"):
        self.market_data = MarketData(symbol=symbol, interval=interval)
        self.whale_data = {}
        self.last_update = None

    async def get_market_data(self, pair: str, interval: str = "1m") -> pd.DataFrame:
        return self.market_data.df

    async def get_order_book(self, pair: str) -> Dict[str, Any]:
        return {"bids": [], "asks": []}  # TODO: Реализовать получение стакана


class WhaleActivityCache:
    def __init__(self):
        self.activities: Dict[str, List[WhaleActivity]] = {}

    def add(self, pair: str, activity: WhaleActivity):
        self.activities.setdefault(pair, []).append(activity)

    def get_recent(self, pair: str, lookback: int) -> List[WhaleActivity]:
        return self.activities.get(pair, [])[-lookback:]

    def clear(self):
        self.activities.clear()


class WhaleSignalAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze_order_book(
        self, order_book: Dict[str, Any], config: Dict[str, Any]
    ) -> Optional[WhaleActivity]:
        # ... кластеризация, spoofing/iceberg detection ...
        return None

    def analyze_volume(
        self, market_data: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[WhaleActivity]:
        # ... кумулятивные изменения, экстремумы ...
        return None

    def analyze_impulses(
        self, market_data: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[WhaleActivity]:
        # ... волновой анализ ...
        return None

    def analyze_dominance(
        self, market_data: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[WhaleActivity]:
        # ... кросс-активные корреляции ...
        return None


class WhalesAgent:
    """
    Агент анализа активности китов: асинхронный анализ, алерты, история, автоматическая калибровка порогов.
    """

    config: Dict[str, Any]
    data_provider: IDataProvider
    cache: WhaleActivityCache
    analyzer: WhaleSignalAnalyzer
    market_regime_agent: MarketRegimeAgent

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента анализа активности китов.
        :param config: словарь параметров
        """
        self.config = config or {
            "volume_threshold": 1000000,
            "price_impact_threshold": 0.02,
            "order_book_depth": 20,
            "min_whale_size": 100000,
            "max_spread": 0.01,
        }
        self.data_provider = DefaultDataProvider()
        self.cache = WhaleActivityCache()
        self.analyzer = WhaleSignalAnalyzer(self.config)
        self.market_regime_agent = MarketRegimeAgent()
        self.whale_activity = False
        self.last_analysis = datetime.now()
        self.activity_score = 0.0  # Используем float напрямую

    async def detect_whale_activity(self, pair: str) -> List[WhaleActivity]:
        """Определение активности китов."""
        try:
            # Получаем данные
            market_data = await self.data_provider.get_market_data(pair)
            order_book = await self.data_provider.get_order_book(pair)

            # Анализируем активность
            is_whale_activity = self._detect_whale_activity(market_data)
            order_book_analysis = await self._analyze_order_book(order_book)
            whale_score = self._calculate_whale_score(market_data)

            if is_whale_activity:
                return [
                    WhaleActivity(
                        pair=pair,
                        timestamp=pd.Timestamp.now(),
                        activity_type=WhaleActivityType.ORDER_BOOK,
                        confidence=float(whale_score),
                        impact=float(whale_score),
                        details={
                            "imbalance": float(order_book_analysis["imbalance"]),
                            "bid_volume": float(order_book_analysis["bid_volume"]),
                            "ask_volume": float(order_book_analysis["ask_volume"]),
                        },
                    )
                ]
            return []
        except Exception as e:
            logger.error(f"Error detecting whale activity: {str(e)}")
            return []

    def generate_alerts_for_market_regime(self) -> Dict[str, List[Dict[str, any]]]:
        """
        Генерация алертов для изменения рыночного режима.
        :return: словарь алертов по парам
        """
        try:
            alerts: Dict[str, List[Dict[str, any]]] = {}
            for pair, activities in self.cache.activities.items():
                pair_alerts: List[Dict[str, any]] = []
                recent_activities = activities[-self.config["lookback_period"] :]
                total_impact = sum(a.impact for a in recent_activities)
                total_confidence = sum(a.confidence for a in recent_activities)
                if (
                    total_impact > self.config["impact_threshold"]
                    and total_confidence > self.config["confidence_threshold"]
                ):
                    activity_types = [str(a.activity_type) for a in recent_activities]
                    dominant_type = max(set(activity_types), key=activity_types.count)
                    alert = {
                        "pair": pair,
                        "type": dominant_type,
                        "impact": total_impact,
                        "confidence": total_confidence,
                        "timestamp": pd.Timestamp.now(),
                        "suggested_action": self._get_suggested_action(dominant_type),
                    }
                    pair_alerts.append(alert)
                if pair_alerts:
                    alerts[pair] = pair_alerts
            return alerts
        except Exception as e:
            logger.error(f"Error generating market regime alerts: {str(e)}")
            return {}

    def whale_confidence_score(self, pair: str) -> float:
        """
        Расчет уверенности в активности китов.
        :param pair: тикер пары
        :return: float (0..1)
        """
        try:
            if pair not in self.cache.activities:
                return 0.0
            recent_activities = self.cache.activities[pair][
                -self.config["lookback_period"] :
            ]
            if not recent_activities:
                return 0.0
            weights = {
                "order_book": 0.4,
                "volume": 0.3,
                "impulse": 0.2,
                "dominance": 0.1,
            }
            weighted_confidence = sum(
                a.confidence * weights[str(a.activity_type)] for a in recent_activities
            )
            return min(weighted_confidence / len(recent_activities), 1.0)
        except Exception as e:
            logger.error(f"Error calculating whale confidence for {pair}: {str(e)}")
            return 0.0

    async def _analyze_order_book(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Анализ стакана заявок."""
        try:
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data)

            # Анализируем объемы на уровнях
            bid_volumes = df["bids"].apply(lambda x: sum(bid[1] for bid in x))
            ask_volumes = df["asks"].apply(lambda x: sum(ask[1] for ask in x))

            # Определяем дисбаланс
            volume_imbalance = (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes)

            return {
                "imbalance": float(volume_imbalance.iloc[-1]),
                "bid_volume": float(bid_volumes.iloc[-1]),
                "ask_volume": float(ask_volumes.iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error analyzing order book: {str(e)}")
            return {"imbalance": 0.0, "bid_volume": 0.0, "ask_volume": 0.0}

    def _detect_whale_activity(self, data: pd.DataFrame) -> bool:
        """Определение активности китов."""
        try:
            # Проверяем объемы
            volume_threshold = data["volume"].mean() * 3
            large_volumes = (data["volume"] > volume_threshold).values

            # Проверяем ценовые движения
            price_changes = data["close"].pct_change().abs()
            significant_moves = (price_changes > 0.02).values  # 2% изменение

            # Проверяем спред
            spread = (data["high"] - data["low"]) / data["low"]
            wide_spread = (spread > 0.01).values  # 1% спред

            # Если есть хотя бы одно условие
            return (
                bool(np.any(large_volumes))
                or bool(np.any(significant_moves))
                or bool(np.any(wide_spread))
            )
        except Exception as e:
            logger.error(f"Error detecting whale activity: {str(e)}")
            return False

    def _calculate_whale_score(self, data: pd.DataFrame) -> float:
        """Расчет скора активности китов."""
        try:
            # Преобразуем numpy array в Series
            volumes = pd.Series(data["volume"].values)
            price_changes = pd.Series(data["close"].pct_change().values)

            # Нормализуем метрики
            volume_score = min(
                1.0, float(volumes.iloc[-1]) / self.config["volume_threshold"]
            )
            price_score = min(
                1.0,
                abs(float(price_changes.iloc[-1]))
                / self.config["price_impact_threshold"],
            )

            return (volume_score + price_score) / 2
        except Exception as e:
            logger.error(f"Error calculating whale score: {str(e)}")
            return 0.0

    async def _analyze_volume(self, pair: str) -> Optional[WhaleActivity]:
        """Анализ необычного объема"""
        try:
            market_data = await self.data_provider.get_market_data(pair)
            if market_data.empty:
                return None
            volume_profile = calculate_volume_profile(market_data)
            if volume_profile is None:
                return None
            volume_series = pd.Series(volume_profile)
            if volume_series.empty:
                return None

            # Расчет среднего объема
            avg_volume = volume_series.rolling(20).mean().iloc[-1]
            current_volume = volume_series.iloc[-1]

            if current_volume < avg_volume * self.config["volume_threshold"]:
                return None

            # Расчет уверенности и влияния
            volume_ratio = current_volume / avg_volume
            confidence = min(volume_ratio / 5, 1.0)
            impact = min(volume_ratio / 3, 1.0)

            return WhaleActivity(
                timestamp=pd.Timestamp.now(),
                pair=pair,
                activity_type=WhaleActivityType.VOLUME,
                confidence=confidence,
                impact=impact,
                details={
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,
                    "ratio": volume_ratio,
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing volume for {pair}: {str(e)}")
            return None

    async def _analyze_impulses(self, pair: str) -> Optional[WhaleActivity]:
        """Анализ быстрых импульсов"""
        try:
            # Получение данных на низких таймфреймах
            market_data = await self.data_provider.get_market_data(pair, interval="1m")

            # Расчет импульсов
            price_change = market_data["close"].pct_change()
            impulse_mask = abs(price_change) > self.config["impulse_threshold"]

            if not impulse_mask.any():
                return None

            # Анализ последнего импульса
            last_impulse = price_change[impulse_mask].iloc[-1]

            # Расчет уверенности и влияния
            confidence = min(abs(last_impulse) / 0.05, 1.0)
            impact = min(abs(last_impulse) / 0.03, 1.0)

            return WhaleActivity(
                timestamp=pd.Timestamp.now(),
                pair=pair,
                activity_type=WhaleActivityType.IMPULSE,
                confidence=confidence,
                impact=impact,
                details={
                    "impulse_size": last_impulse,
                    "direction": "up" if last_impulse > 0 else "down",
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing impulses for {pair}: {str(e)}")
            return None

    async def _analyze_dominance(self, pair: str) -> Optional[WhaleActivity]:
        """Анализ доминирования BTC/ETH"""
        try:
            if pair not in ["BTCUSDT", "ETHUSDT"]:
                return None

            # Получение данных о доминировании
            market_data = await self.data_provider.get_market_data("BTCUSDT")
            total_market_cap = (
                market_data["close"].iloc[-1] * market_data["volume"].iloc[-1]
            )

            # Расчет доминирования
            if pair == "BTCUSDT":
                dominance = total_market_cap / self._get_total_crypto_market_cap()
            else:
                eth_data = await self.data_provider.get_market_data("ETHUSDT")
                eth_market_cap = (
                    eth_data["close"].iloc[-1] * eth_data["volume"].iloc[-1]
                )
                dominance = eth_market_cap / total_market_cap

            if dominance < self.config["dominance_threshold"]:
                return None

            # Расчет уверенности и влияния
            confidence = min(dominance / 0.8, 1.0)
            impact = min(dominance / 0.7, 1.0)

            return WhaleActivity(
                timestamp=pd.Timestamp.now(),
                pair=pair,
                activity_type=WhaleActivityType.DOMINANCE,
                confidence=confidence,
                impact=impact,
                details={"dominance": dominance, "market_cap": total_market_cap},
            )

        except Exception as e:
            logger.error(f"Error analyzing dominance for {pair}: {str(e)}")
            return None

    def _get_total_crypto_market_cap(self) -> float:
        """Получение общей капитализации крипторынка"""
        try:
            # Здесь должна быть реализация получения данных
            # о общей капитализации крипторынка
            return 1.0  # Заглушка
        except Exception as e:
            logger.error(f"Error getting total crypto market cap: {str(e)}")
            return 1.0

    def _get_suggested_action(self, activity_type: str) -> str:
        """Получение рекомендуемого действия на основе типа активности."""
        actions = {
            "order_book": "monitor",
            "volume": "alert",
            "impulse": "trade",
            "dominance": "analyze",
        }
        return actions.get(activity_type, "monitor")

    async def get_signals(self) -> List[Signal]:
        """
        Получение сигналов от агента.

        Returns:
            List[Signal]: Список сигналов
        """
        try:
            signals = []
            for pair in self.config.get("pairs", []):
                activities = await self.detect_whale_activity(pair)
                if activities:
                    for activity in activities:
                        signals.append(
                            Signal(
                                pair=pair,
                                action="monitor",
                                price=0.0,  # TODO: Получить текущую цену
                                size=0.0,  # TODO: Рассчитать размер
                                metadata={
                                    "type": "whale_activity",
                                    "strength": activity.impact,
                                    "confidence": activity.confidence,
                                    "timestamp": activity.timestamp,
                                    "source": "whale_agent",
                                },
                            )
                        )
            return signals
        except Exception as e:
            logger.error(f"Error getting whale signals: {str(e)}")
            return []
