"""
Распознавание рыночных паттернов.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd

from domain.entities.signal import Signal, SignalType
from domain.types import Symbol
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money

from loguru import logger

from domain.memory.pattern_memory import MarketFeatures
from domain.types.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp


@dataclass
class PatternDetection:
    """Результат обнаружения паттерна."""

    pattern_type: PatternType
    symbol: str
    timestamp: Timestamp
    confidence: float  # 0.0 - 1.0
    strength: float  # Сила сигнала
    direction: str  # "up", "down", "neutral"
    metadata: Dict[str, Any]
    # Ключевые метрики паттерна
    volume_anomaly: float
    price_impact: float
    order_book_imbalance: float
    spread_widening: float
    depth_absorption: float
    # Рыночные характеристики (для совместимости)
    features: Optional[MarketFeatures] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_type": self.pattern_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.to_iso(),
            "confidence": self.confidence,
            "strength": self.strength,
            "direction": self.direction,
            "metadata": self.metadata,
            "volume_anomaly": self.volume_anomaly,
            "price_impact": self.price_impact,
            "order_book_imbalance": self.order_book_imbalance,
            "spread_widening": self.spread_widening,
            "depth_absorption": self.depth_absorption,
        }


class MarketPatternRecognizer:
    """Распознаватель паттернов крупного капитала."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "volume_threshold": 1000000,  # Минимальный объем для кита
            "price_impact_threshold": 0.02,  # 2% изменение цены
            "spread_threshold": 0.001,  # 0.1% спред
            "depth_imbalance_threshold": 0.3,  # 30% дисбаланс
            "confidence_threshold": 0.7,  # Минимальная уверенность
            "lookback_periods": 50,  # Периоды для анализа
            "volume_sma_periods": 20,  # Периоды для SMA объема
        }
        # Кэш для оптимизации
        self._volume_cache: Dict[str, pd.Series] = {}
        self._price_cache: Dict[str, pd.Series] = {}
        self._last_analysis: Dict[str, datetime] = {}
        # Логгер
        self.logger = logger

    def detect_whale_absorption(
        self, symbol: str, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Optional[PatternDetection]:
        """
        Обнаружение поглощения ликвидности крупным капиталом.
        Признаки:
        - Большой объем
        - Дисбаланс стакана
        - Расширение спреда
        - Движение цены
        """
        try:
            if len(market_data) < self.config["lookback_periods"]:
                return self._create_default_pattern(symbol, "insufficient_data")
            # Анализ объема
            volume_analysis = self._analyze_volume_anomaly(market_data)
            # Анализ движения цены
            price_analysis = self._analyze_price_movement(market_data)
            # Анализ стакана
            orderbook_analysis = self._analyze_order_book_imbalance(order_book)
            # Анализ спреда
            spread_analysis = self._analyze_spread_widening(order_book)
            # Расчет уверенности
            confidence = self._calculate_absorption_confidence(
                volume_analysis, price_analysis, orderbook_analysis, spread_analysis
            )
            if confidence < self.config["confidence_threshold"]:
                return None
            # Определение направления
            direction = self._determine_absorption_direction(
                price_analysis, orderbook_analysis
            )
            return PatternDetection(
                pattern_type=PatternType.WHALE_ABSORPTION,
                symbol=symbol,
                timestamp=Timestamp.now(),
                confidence=confidence,
                strength=volume_analysis["anomaly_ratio"],
                direction=direction,
                metadata={
                    "volume_analysis": volume_analysis,
                    "price_analysis": price_analysis,
                    "orderbook_analysis": orderbook_analysis,
                    "spread_analysis": spread_analysis,
                },
                volume_anomaly=volume_analysis["anomaly_ratio"],
                price_impact=price_analysis["price_change"],
                order_book_imbalance=orderbook_analysis["imbalance_ratio"],
                spread_widening=spread_analysis["spread_change"],
                depth_absorption=orderbook_analysis["absorption_ratio"],
            )
        except Exception as e:
            logger.error(f"Error detecting whale absorption for {symbol}: {e}")
            return None

    def _create_default_pattern(self, symbol: str, reason: str) -> Optional[PatternDetection]:
        """Создает паттерн по умолчанию при недостатке данных."""
        return PatternDetection(
            pattern_type=PatternType.WHALE_ABSORPTION,
            symbol=symbol,
            timestamp=Timestamp.now(),
            confidence=0.0,
            strength=0.0,
            direction="neutral",
            metadata={"reason": reason},
            volume_anomaly=0.0,
            price_impact=0.0,
            order_book_imbalance=0.0,
            spread_widening=0.0,
            depth_absorption=0.0,
        )

    def detect_mm_spoofing(
        self, symbol: str, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Optional[PatternDetection]:
        """
        Обнаружение спуфинга маркет-мейкеров.
        Признаки:
        - Дисбаланс стакана
        - Резкое движение цены
        - Восстановление ликвидности
        """
        try:
            if len(market_data) < self.config["lookback_periods"]:
                return self._create_default_pattern(symbol, "insufficient_data")
            # Анализ дисбаланса стакана
            imbalance_analysis = self._analyze_order_book_imbalance(order_book)
            # Анализ движения цены
            price_analysis = self._analyze_price_movement(market_data)
            # Анализ ликвидности
            liquidity_analysis = self._analyze_liquidity_dynamics(order_book)
            # Анализ объема
            volume_analysis = self._analyze_volume_anomaly(market_data)
            # Расчет уверенности
            confidence = self._calculate_spoofing_confidence(
                imbalance_analysis, price_analysis, liquidity_analysis, volume_analysis
            )
            if confidence < self.config["confidence_threshold"]:
                return None
            # Определение направления
            direction = self._determine_spoofing_direction(
                imbalance_analysis, price_analysis
            )
            return PatternDetection(
                pattern_type=PatternType.MM_SPOOFING,
                symbol=symbol,
                timestamp=Timestamp.now(),
                confidence=confidence,
                strength=imbalance_analysis["imbalance_ratio"],
                direction=direction,
                metadata={
                    "imbalance_analysis": imbalance_analysis,
                    "price_analysis": price_analysis,
                    "liquidity_analysis": liquidity_analysis,
                    "volume_analysis": volume_analysis,
                },
                volume_anomaly=volume_analysis["anomaly_ratio"],
                price_impact=price_analysis["price_change"],
                order_book_imbalance=imbalance_analysis["imbalance_ratio"],
                spread_widening=liquidity_analysis["spread_change"],
                depth_absorption=liquidity_analysis["absorption_ratio"],
            )
        except Exception as e:
            logger.error(f"Error detecting MM spoofing for {symbol}: {e}")
            return None

    def detect_iceberg_detection(
        self, symbol: str, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> Optional[PatternDetection]:
        """
        Обнаружение айсберг-ордеров.
        Признаки:
        - Постоянные ордера одного размера
        - Медленное движение цены
        - Стабильный объем
        """
        try:
            if len(market_data) < self.config["lookback_periods"]:
                return self._create_default_pattern(symbol, "insufficient_data")
            # Анализ повторяющихся ордеров
            iceberg_analysis = self._analyze_iceberg_patterns(market_data)
            # Анализ движения цены
            price_analysis = self._analyze_price_movement(market_data)
            # Анализ объема
            volume_analysis = self._analyze_volume_consistency(market_data)
            # Расчет уверенности
            confidence = self._calculate_iceberg_confidence(
                iceberg_analysis, price_analysis, volume_analysis
            )
            if confidence < self.config["confidence_threshold"]:
                return None
            return PatternDetection(
                pattern_type=PatternType.ICEBERG_DETECTION,
                symbol=symbol,
                timestamp=Timestamp.now(),
                confidence=confidence,
                strength=iceberg_analysis["pattern_strength"],
                direction="neutral",  # Айсберги обычно нейтральны
                metadata={
                    "iceberg_analysis": iceberg_analysis,
                    "price_analysis": price_analysis,
                    "volume_analysis": volume_analysis,
                },
                volume_anomaly=volume_analysis["consistency_ratio"],
                price_impact=price_analysis["price_change"],
                order_book_imbalance=iceberg_analysis["imbalance_ratio"],
                spread_widening=0.0,  # Айсберги не влияют на спред
                depth_absorption=iceberg_analysis["absorption_ratio"],
            )
        except Exception as e:
            logger.error(f"Error detecting iceberg for {symbol}: {e}")
            return None

    def _analyze_volume_anomaly(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ аномалий объема."""
        try:
            # Текущий объем
            current_volume = market_data["volume"].iloc[-1]
            
            # Средний объем
            avg_volume = (
                market_data["volume"]
                .rolling(self.config["volume_sma_periods"])
                .mean()
                .iloc[-1]
            )
            
            # Изменение объема
            volume_change = (
                market_data["volume"].pct_change().rolling(5).sum().iloc[-1]
            )
            
            # Анализ аномалии
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > self.config["volume_anomaly_threshold"]:
                return {
                    "type": "volume_spike",
                    "confidence": min(0.9, volume_ratio / 3.0),
                    "severity": volume_ratio,
                    "change": volume_change,
                }
            elif volume_ratio < 1.0 / self.config["volume_anomaly_threshold"]:
                return {
                    "type": "volume_drop",
                    "confidence": min(0.9, (1.0 / volume_ratio) / 3.0),
                    "severity": 1.0 / volume_ratio,
                    "change": volume_change,
                }
            
            return {
                "type": "normal",
                "confidence": 0.0,
                "severity": 1.0,
                "change": volume_change,
            }
        except Exception as e:
            self.logger.error(f"Error analyzing volume anomaly: {e}")
            return {
                "type": "error",
                "confidence": 0.0,
                "severity": 0.0,
                "change": 0.0,
            }

    def _analyze_price_movement(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ движения цены."""
        try:
            # Изменения цены
            price_changes = market_data["close"].pct_change()
            
            # Краткосрочное изменение
            short_change = price_changes.rolling(3).sum().iloc[-1]
            
            # Среднесрочное изменение
            medium_change = price_changes.rolling(10).sum().iloc[-1]
            
            # Волатильность
            volatility = price_changes.rolling(20).std().iloc[-1]
            
            return {
                "short_term_change": short_change,
                "medium_term_change": medium_change,
                "volatility": volatility,
                "trend_direction": "up" if medium_change > 0 else "down",
                "trend_strength": abs(medium_change),
            }
        except Exception as e:
            self.logger.error(f"Error analyzing price movement: {e}")
            return {
                "short_term_change": 0.0,
                "medium_term_change": 0.0,
                "volatility": 0.0,
                "trend_direction": "sideways",
                "trend_strength": 0.0,
            }

    def _analyze_order_book_imbalance(
        self, order_book: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Анализ дисбаланса стакана."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return {"imbalance_ratio": 0.0, "absorption_ratio": 0.0}
            # Объемы по сторонам
            bid_volume = sum(bid["size"] for bid in bids)
            ask_volume = sum(ask["size"] for ask in asks)
            total_volume = bid_volume + ask_volume
            imbalance_ratio = (
                (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
            )
            # Анализ поглощения ликвидности
            absorption_ratio = self._calculate_absorption_ratio(order_book)
            return {
                "imbalance_ratio": imbalance_ratio,
                "absorption_ratio": absorption_ratio,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": total_volume,
                "is_imbalanced": abs(imbalance_ratio)
                > self.config["depth_imbalance_threshold"],
            }
        except Exception as e:
            self.logger.error(f"Error analyzing order book imbalance: {e}")
            return {"imbalance_ratio": 0.0, "absorption_ratio": 0.0}

    def _analyze_spread_widening(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ расширения спреда."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return {"spread_change": 0.0, "current_spread": 0.0}
            best_bid = max(bid["price"] for bid in bids)
            best_ask = min(ask["price"] for ask in asks)
            current_spread = (best_ask - best_bid) / best_bid
            spread_change = current_spread - self.config["spread_threshold"]
            return {
                "spread_change": spread_change,
                "current_spread": current_spread,
                "is_widening": spread_change > 0,
                "threshold_exceeded": current_spread > self.config["spread_threshold"],
            }
        except Exception as e:
            self.logger.error(f"Error analyzing spread widening: {e}")
            return {"spread_change": 0.0, "current_spread": 0.0}

    def _analyze_liquidity_dynamics(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ динамики ликвидности."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            # Анализ распределения ордеров
            bid_sizes = [bid["size"] for bid in bids]
            ask_sizes = [ask["size"] for ask in asks]
            # Коэффициент вариации размеров
            bid_cv = np.std(bid_sizes) / np.mean(bid_sizes) if bid_sizes else 0.0
            ask_cv = np.std(ask_sizes) / np.mean(ask_sizes) if ask_sizes else 0.0
            # Анализ поглощения
            absorption_ratio = self._calculate_absorption_ratio(order_book)
            return {
                "absorption_ratio": absorption_ratio,
                "bid_cv": bid_cv,
                "ask_cv": ask_cv,
                "spread_change": 0.0,  # Будет рассчитано отдельно
                "liquidity_quality": 1.0 / (1.0 + bid_cv + ask_cv),
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity dynamics: {e}")
            return {"absorption_ratio": 0.0, "spread_change": 0.0}

    def _analyze_iceberg_patterns(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ паттернов айсберг-ордеров."""
        try:
            # Анализ недавних объемов
            recent_volumes = market_data["volume"].tail(10)
            
            # Поиск повторяющихся объемов
            volume_counts = recent_volumes.value_counts()
            repeated_volumes = volume_counts[volume_counts > 2]
            
            if len(repeated_volumes) > 0:
                most_common_volume = repeated_volumes.index[0]
                frequency = repeated_volumes.iloc[0]
                
                return {
                    "detected": True,
                    "iceberg_volume": most_common_volume,
                    "frequency": frequency,
                    "confidence": min(0.9, frequency / 10.0),
                }
            
            return {"detected": False, "confidence": 0.0}
        except Exception as e:
            self.logger.error(f"Error analyzing iceberg patterns: {e}")
            return {"detected": False, "confidence": 0.0}

    def _analyze_volume_consistency(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ консистентности объема."""
        try:
            recent_volumes = market_data["volume"].tail(10)
            volume_std = recent_volumes.std()
            volume_mean = recent_volumes.mean()
            consistency_ratio = (
                1.0 - (volume_std / volume_mean) if volume_mean > 0 else 0.0
            )
            return {
                "consistency_ratio": consistency_ratio,
                "volume_std": volume_std,
                "volume_mean": volume_mean,
                "is_consistent": consistency_ratio > 0.7,
            }
        except Exception as e:
            logger.error(f"Error analyzing volume consistency: {e}")
            return {"consistency_ratio": 0.0, "is_consistent": False}

    def _calculate_absorption_ratio(self, order_book: Dict[str, Any]) -> float:
        """Расчет коэффициента поглощения ликвидности."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return 0.0
            # Анализ больших ордеров
            large_bids = [
                bid
                for bid in bids
                if bid["size"] > np.mean([b["size"] for b in bids]) * 2
            ]
            large_asks = [
                ask
                for ask in asks
                if ask["size"] > np.mean([a["size"] for a in asks]) * 2
            ]
            total_large = len(large_bids) + len(large_asks)
            total_orders = len(bids) + len(asks)
            return total_large / total_orders if total_orders > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating absorption ratio: {e}")
            return 0.0

    def _calculate_absorption_confidence(
        self,
        volume_analysis: Dict[str, Any],
        price_analysis: Dict[str, Any],
        orderbook_analysis: Dict[str, Any],
        spread_analysis: Dict[str, Any],
    ) -> float:
        """Расчет уверенности в поглощении."""
        try:
            confidence_factors = []
            # Фактор объема
            if volume_analysis["is_anomaly"]:
                confidence_factors.append(
                    min(1.0, volume_analysis["anomaly_ratio"] / 5.0)
                )
            # Фактор цены
            if price_analysis["is_significant"]:
                confidence_factors.append(0.8)
            # Фактор стакана
            if orderbook_analysis["is_imbalanced"]:
                confidence_factors.append(abs(orderbook_analysis["imbalance_ratio"]))
            # Фактор спреда
            if spread_analysis["is_widening"]:
                confidence_factors.append(0.7)
            return float(np.mean(confidence_factors) if confidence_factors else 0.0)
        except Exception as e:
            logger.error(f"Error calculating absorption confidence: {e}")
            return 0.0

    def _calculate_spoofing_confidence(
        self,
        imbalance_analysis: Dict[str, Any],
        price_analysis: Dict[str, Any],
        liquidity_analysis: Dict[str, Any],
        volume_analysis: Dict[str, Any],
    ) -> float:
        """Расчет уверенности в спуфинге."""
        try:
            confidence_factors = []
            # Фактор дисбаланса
            if imbalance_analysis["is_imbalanced"]:
                confidence_factors.append(abs(imbalance_analysis["imbalance_ratio"]))
            # Фактор цены
            if price_analysis["is_significant"]:
                confidence_factors.append(0.9)
            # Фактор ликвидности
            if liquidity_analysis["liquidity_quality"] < 0.5:
                confidence_factors.append(0.8)
            # Фактор объема
            if volume_analysis["is_anomaly"]:
                confidence_factors.append(0.6)
            return float(np.mean(confidence_factors) if confidence_factors else 0.0)
        except Exception as e:
            logger.error(f"Error calculating spoofing confidence: {e}")
            return 0.0

    def _calculate_iceberg_confidence(
        self,
        iceberg_analysis: Dict[str, Any],
        price_analysis: Dict[str, Any],
        volume_analysis: Dict[str, Any],
    ) -> float:
        """Расчет уверенности в айсберге."""
        try:
            confidence_factors = []
            # Фактор паттерна
            if iceberg_analysis["is_iceberg"]:
                confidence_factors.append(iceberg_analysis["pattern_strength"])
            # Фактор цены (айсберги обычно стабильны)
            if abs(price_analysis["price_change"]) < 0.01:
                confidence_factors.append(0.8)
            # Фактор объема
            if volume_analysis["is_consistent"]:
                confidence_factors.append(volume_analysis["consistency_ratio"])
            return float(np.mean(confidence_factors) if confidence_factors else 0.0)
        except Exception as e:
            logger.error(f"Error calculating iceberg confidence: {e}")
            return 0.0

    def _determine_absorption_direction(
        self, price_analysis: Dict[str, Any], orderbook_analysis: Dict[str, Any]
    ) -> str:
        """Определение направления поглощения."""
        try:
            # Если дисбаланс в сторону покупок и цена растет
            if (
                orderbook_analysis["imbalance_ratio"] > 0.1
                and price_analysis["trend_direction"] == "up"
            ):
                return "up"
            # Если дисбаланс в сторону продаж и цена падает
            elif (
                orderbook_analysis["imbalance_ratio"] < -0.1
                and price_analysis["trend_direction"] == "down"
            ):
                return "down"
            # Иначе нейтрально
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error determining absorption direction: {e}")
            return "neutral"

    def _determine_spoofing_direction(
        self, imbalance_analysis: Dict[str, Any], price_analysis: Dict[str, Any]
    ) -> str:
        """Определение направления спуфинга."""
        try:
            # Если большой дисбаланс в сторону покупок, но цена не растет
            if (
                imbalance_analysis["imbalance_ratio"] > 0.3
                and price_analysis["trend_direction"] != "up"
            ):
                return "down"  # Скорее всего фейк вверх
            # Если большой дисбаланс в сторону продаж, но цена не падает
            elif (
                imbalance_analysis["imbalance_ratio"] < -0.3
                and price_analysis["trend_direction"] != "down"
            ):
                return "up"  # Скорее всего фейк вниз
            # Иначе нейтрально
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error determining spoofing direction: {e}")
            return "neutral"

