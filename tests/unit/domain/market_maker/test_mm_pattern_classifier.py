"""
Unit тесты для mm_pattern_classifier.py.

Покрывает:
- OrderBookSnapshot - снимки ордербука
- TradeSnapshot - снимки сделок
- MarketMakerPatternClassifier - классификатор паттернов
- IPatternClassifier - интерфейс классификатора
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from collections import deque

from domain.market_maker.mm_pattern_classifier import (
    OrderBookSnapshot,
    TradeSnapshot,
    MarketMakerPatternClassifier,
    IPatternClassifier
)
from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    MarketMakerPatternType,
    PatternFeatures
)
from domain.types.market_maker_types import (
    Symbol,
    OrderBookLevel,
    TradeData,
    PatternClassifierConfig,
    Confidence,
    BookPressure,
    VolumeDelta,
    PriceReaction,
    SpreadChange,
    OrderImbalance,
    LiquidityDepth,
    TimeDuration,
    VolumeConcentration,
    PriceVolatility
)


class TestOrderBookSnapshot:
    """Тесты для OrderBookSnapshot."""
    
    @pytest.fixture
    def sample_order_book_data(self) -> Dict[str, Any]:
        """Тестовые данные для OrderBookSnapshot."""
        return {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "symbol": "BTC/USDT",
            "bids": [
                {"price": 50000.0, "size": 1.5, "orders_count": 10},
                {"price": 49999.0, "size": 2.0, "orders_count": 15},
                {"price": 49998.0, "size": 1.0, "orders_count": 8}
            ],
            "asks": [
                {"price": 50001.0, "size": 1.2, "orders_count": 12},
                {"price": 50002.0, "size": 1.8, "orders_count": 18},
                {"price": 50003.0, "size": 0.8, "orders_count": 6}
            ],
            "last_price": 50000.5,
            "volume_24h": 1000.0,
            "price_change_24h": 0.02
        }
    
    @pytest.fixture
    def order_book_snapshot(self, sample_order_book_data) -> OrderBookSnapshot:
        """Создает экземпляр OrderBookSnapshot."""
        return OrderBookSnapshot(**sample_order_book_data)
    
    def test_creation_with_valid_data(self, sample_order_book_data):
        """Тест создания с валидными данными."""
        snapshot = OrderBookSnapshot(**sample_order_book_data)
        
        assert snapshot.timestamp == sample_order_book_data["timestamp"]
        assert snapshot.symbol == sample_order_book_data["symbol"]
        assert snapshot.bids == sample_order_book_data["bids"]
        assert snapshot.asks == sample_order_book_data["asks"]
        assert snapshot.last_price == sample_order_book_data["last_price"]
        assert snapshot.volume_24h == sample_order_book_data["volume_24h"]
        assert snapshot.price_change_24h == sample_order_book_data["price_change_24h"]
    
    def test_get_bid_volume_default_levels(self, order_book_snapshot):
        """Тест получения объема бидов по умолчанию."""
        volume = order_book_snapshot.get_bid_volume()
        expected = 1.5 + 2.0 + 1.0 + 0.0 + 0.0  # 5 уровней, последние 2 пустые
        assert volume == expected
    
    def test_get_bid_volume_custom_levels(self, order_book_snapshot):
        """Тест получения объема бидов с пользовательским количеством уровней."""
        volume = order_book_snapshot.get_bid_volume(2)
        expected = 1.5 + 2.0  # Только первые 2 уровня
        assert volume == expected
    
    def test_get_ask_volume_default_levels(self, order_book_snapshot):
        """Тест получения объема асков по умолчанию."""
        volume = order_book_snapshot.get_ask_volume()
        expected = 1.2 + 1.8 + 0.8 + 0.0 + 0.0  # 5 уровней, последние 2 пустые
        assert volume == expected
    
    def test_get_ask_volume_custom_levels(self, order_book_snapshot):
        """Тест получения объема асков с пользовательским количеством уровней."""
        volume = order_book_snapshot.get_ask_volume(2)
        expected = 1.2 + 1.8  # Только первые 2 уровня
        assert volume == expected
    
    def test_get_mid_price(self, order_book_snapshot):
        """Тест получения средней цены."""
        mid_price = order_book_snapshot.get_mid_price()
        expected = (50000.0 + 50001.0) / 2  # (лучший бид + лучший аск) / 2
        assert mid_price == expected
    
    def test_get_mid_price_no_bids(self):
        """Тест получения средней цены без бидов."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[],
            asks=[{"price": 50001.0, "size": 1.0, "orders_count": 1}],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        mid_price = snapshot.get_mid_price()
        assert mid_price == 50000.5  # Должна вернуться last_price
    
    def test_get_mid_price_no_asks(self):
        """Тест получения средней цены без асков."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[{"price": 50000.0, "size": 1.0, "orders_count": 1}],
            asks=[],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        mid_price = snapshot.get_mid_price()
        assert mid_price == 50000.5  # Должна вернуться last_price
    
    def test_get_spread(self, order_book_snapshot):
        """Тест получения спреда."""
        spread = order_book_snapshot.get_spread()
        expected = 50001.0 - 50000.0  # Лучший аск - лучший бид
        assert spread == expected
    
    def test_get_spread_no_bids(self):
        """Тест получения спреда без бидов."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[],
            asks=[{"price": 50001.0, "size": 1.0, "orders_count": 1}],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        spread = snapshot.get_spread()
        assert spread == 0.0
    
    def test_get_spread_no_asks(self):
        """Тест получения спреда без асков."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[{"price": 50000.0, "size": 1.0, "orders_count": 1}],
            asks=[],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        spread = snapshot.get_spread()
        assert spread == 0.0
    
    def test_get_spread_percentage(self, order_book_snapshot):
        """Тест получения процентного спреда."""
        spread_percentage = order_book_snapshot.get_spread_percentage()
        mid_price = (50000.0 + 50001.0) / 2
        expected = ((50001.0 - 50000.0) / mid_price) * 100
        assert abs(spread_percentage - expected) < 0.001
    
    def test_get_spread_percentage_zero_mid_price(self):
        """Тест получения процентного спреда при нулевой средней цене."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[{"price": 0.0, "size": 1.0, "orders_count": 1}],
            asks=[{"price": 0.0, "size": 1.0, "orders_count": 1}],
            last_price=0.0,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        spread_percentage = snapshot.get_spread_percentage()
        assert spread_percentage == 0.0
    
    def test_get_order_imbalance_balanced(self, order_book_snapshot):
        """Тест получения дисбаланса ордеров (сбалансированный)."""
        imbalance = order_book_snapshot.get_order_imbalance()
        bid_volume = 1.5 + 2.0 + 1.0 + 0.0 + 0.0
        ask_volume = 1.2 + 1.8 + 0.8 + 0.0 + 0.0
        total_volume = bid_volume + ask_volume
        expected = (bid_volume - ask_volume) / total_volume
        assert abs(float(imbalance) - expected) < 0.001
    
    def test_get_order_imbalance_zero_volume(self):
        """Тест получения дисбаланса ордеров при нулевом объеме."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[],
            asks=[],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        imbalance = snapshot.get_order_imbalance()
        assert float(imbalance) == 0.0
    
    def test_get_order_imbalance_extreme_values(self):
        """Тест получения дисбаланса ордеров с экстремальными значениями."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[{"price": 50000.0, "size": 1000.0, "orders_count": 1}],
            asks=[{"price": 50001.0, "size": 0.1, "orders_count": 1}],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        imbalance = snapshot.get_order_imbalance()
        # Должен быть ограничен до 1.0
        assert float(imbalance) <= 1.0
    
    def test_get_liquidity_depth(self, order_book_snapshot):
        """Тест получения глубины ликвидности."""
        depth = order_book_snapshot.get_liquidity_depth()
        bid_volume = 1.5 + 2.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0
        ask_volume = 1.2 + 1.8 + 0.8 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0
        expected = bid_volume + ask_volume
        assert float(depth) == expected


class TestTradeSnapshot:
    """Тесты для TradeSnapshot."""
    
    @pytest.fixture
    def sample_trade_data(self) -> List[TradeData]:
        """Тестовые данные для TradeSnapshot."""
        return [
            {
                "price": 50000.0,
                "size": 1.0,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 0),
                "trade_id": "trade1",
                "maker": True
            },
            {
                "price": 50001.0,
                "size": 0.5,
                "side": "sell",
                "time": datetime(2024, 1, 1, 12, 0, 1),
                "trade_id": "trade2",
                "maker": False
            },
            {
                "price": 50002.0,
                "size": 1.5,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 2),
                "trade_id": "trade3",
                "maker": True
            }
        ]
    
    @pytest.fixture
    def trade_snapshot(self, sample_trade_data) -> TradeSnapshot:
        """Создает экземпляр TradeSnapshot."""
        return TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 2),
            symbol="BTC/USDT",
            trades=sample_trade_data
        )
    
    def test_creation_with_valid_data(self, sample_trade_data):
        """Тест создания с валидными данными."""
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 2),
            symbol="BTC/USDT",
            trades=sample_trade_data
        )
        
        assert snapshot.timestamp == datetime(2024, 1, 1, 12, 0, 2)
        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.trades == sample_trade_data
    
    def test_get_total_volume(self, trade_snapshot):
        """Тест получения общего объема."""
        total_volume = trade_snapshot.get_total_volume()
        expected = 1.0 + 0.5 + 1.5
        assert total_volume == expected
    
    def test_get_buy_volume(self, trade_snapshot):
        """Тест получения объема покупок."""
        buy_volume = trade_snapshot.get_buy_volume()
        expected = 1.0 + 1.5  # Только сделки со side="buy"
        assert buy_volume == expected
    
    def test_get_sell_volume(self, trade_snapshot):
        """Тест получения объема продаж."""
        sell_volume = trade_snapshot.get_sell_volume()
        expected = 0.5  # Только сделки со side="sell"
        assert sell_volume == expected
    
    def test_get_volume_delta_sufficient_trades(self, sample_trade_data):
        """Тест получения дельты объема при достаточном количестве сделок."""
        # Добавляем больше сделок для тестирования окна
        extended_trades = sample_trade_data + [
            {
                "price": 50003.0,
                "size": 0.8,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 3),
                "trade_id": "trade4",
                "maker": False
            },
            {
                "price": 50004.0,
                "size": 1.2,
                "side": "sell",
                "time": datetime(2024, 1, 1, 12, 0, 4),
                "trade_id": "trade5",
                "maker": True
            }
        ]
        
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 4),
            symbol="BTC/USDT",
            trades=extended_trades
        )
        
        volume_delta = snapshot.get_volume_delta(window=2)
        recent_volume = 0.8 + 1.2  # Последние 2 сделки
        older_volume = 1.5 + 0.8   # Предыдущие 2 сделки
        expected = (recent_volume - older_volume) / older_volume
        assert abs(float(volume_delta) - expected) < 0.001
    
    def test_get_volume_delta_insufficient_trades(self, trade_snapshot):
        """Тест получения дельты объема при недостаточном количестве сделок."""
        volume_delta = trade_snapshot.get_volume_delta(window=10)
        assert float(volume_delta) == 0.0
    
    def test_get_volume_delta_zero_older_volume(self, sample_trade_data):
        """Тест получения дельты объема при нулевом старом объеме."""
        # Создаем сделки с нулевым объемом в старом окне
        trades_with_zero_old = [
            {
                "price": 50000.0,
                "size": 0.0,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 0),
                "trade_id": "trade1",
                "maker": True
            },
            {
                "price": 50001.0,
                "size": 0.0,
                "side": "sell",
                "time": datetime(2024, 1, 1, 12, 0, 1),
                "trade_id": "trade2",
                "maker": False
            },
            {
                "price": 50002.0,
                "size": 1.5,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 2),
                "trade_id": "trade3",
                "maker": True
            }
        ]
        
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 2),
            symbol="BTC/USDT",
            trades=trades_with_zero_old
        )
        
        volume_delta = snapshot.get_volume_delta(window=1)
        assert float(volume_delta) == 0.0
    
    def test_get_price_reaction(self, trade_snapshot):
        """Тест получения ценовой реакции."""
        price_reaction = trade_snapshot.get_price_reaction()
        first_price = 50000.0
        last_price = 50002.0
        expected = (last_price - first_price) / first_price
        assert abs(float(price_reaction) - expected) < 0.001
    
    def test_get_price_reaction_insufficient_trades(self):
        """Тест получения ценовой реакции при недостаточном количестве сделок."""
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[{
                "price": 50000.0,
                "size": 1.0,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 0),
                "trade_id": "trade1",
                "maker": True
            }]
        )
        
        price_reaction = snapshot.get_price_reaction()
        assert float(price_reaction) == 0.0
    
    def test_get_volume_concentration(self, trade_snapshot):
        """Тест получения концентрации объема."""
        volume_concentration = trade_snapshot.get_volume_concentration()
        volumes = [1.0, 0.5, 1.5]
        total_volume = sum(volumes)
        mean_volume = total_volume / len(volumes)
        variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
        expected = (variance**0.5) / mean_volume
        assert abs(float(volume_concentration) - expected) < 0.001
    
    def test_get_volume_concentration_empty_trades(self):
        """Тест получения концентрации объема при пустых сделках."""
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[]
        )
        
        volume_concentration = snapshot.get_volume_concentration()
        assert float(volume_concentration) == 0.0
    
    def test_get_volume_concentration_zero_total_volume(self):
        """Тест получения концентрации объема при нулевом общем объеме."""
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[
                {
                    "price": 50000.0,
                    "size": 0.0,
                    "side": "buy",
                    "time": datetime(2024, 1, 1, 12, 0, 0),
                    "trade_id": "trade1",
                    "maker": True
                },
                {
                    "price": 50001.0,
                    "size": 0.0,
                    "side": "sell",
                    "time": datetime(2024, 1, 1, 12, 0, 1),
                    "trade_id": "trade2",
                    "maker": False
                }
            ]
        )
        
        volume_concentration = snapshot.get_volume_concentration()
        assert float(volume_concentration) == 0.0
    
    def test_get_price_volatility(self, trade_snapshot):
        """Тест получения волатильности цены."""
        price_volatility = trade_snapshot.get_price_volatility()
        prices = [50000.0, 50001.0, 50002.0]
        returns = [(50001.0 - 50000.0) / 50000.0, (50002.0 - 50001.0) / 50001.0]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        expected = variance**0.5
        assert abs(float(price_volatility) - expected) < 0.001
    
    def test_get_price_volatility_insufficient_trades(self):
        """Тест получения волатильности цены при недостаточном количестве сделок."""
        snapshot = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[{
                "price": 50000.0,
                "size": 1.0,
                "side": "buy",
                "time": datetime(2024, 1, 1, 12, 0, 0),
                "trade_id": "trade1",
                "maker": True
            }]
        )
        
        price_volatility = snapshot.get_price_volatility()
        assert float(price_volatility) == 0.0


class TestIPatternClassifier:
    """Тесты для IPatternClassifier."""
    
    def test_abstract_methods(self):
        """Тест абстрактных методов."""
        with pytest.raises(TypeError):
            IPatternClassifier()


class TestMarketMakerPatternClassifier:
    """Тесты для MarketMakerPatternClassifier."""
    
    @pytest.fixture
    def sample_order_book_snapshot(self) -> OrderBookSnapshot:
        """Создает тестовый снимок ордербука."""
        return OrderBookSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            bids=[
                {"price": 50000.0, "size": 1.5, "orders_count": 10},
                {"price": 49999.0, "size": 2.0, "orders_count": 15}
            ],
            asks=[
                {"price": 50001.0, "size": 1.2, "orders_count": 12},
                {"price": 50002.0, "size": 1.8, "orders_count": 18}
            ],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
    
    @pytest.fixture
    def sample_trade_snapshot(self) -> TradeSnapshot:
        """Создает тестовый снимок сделок."""
        return TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[
                {
                    "price": 50000.0,
                    "size": 1.0,
                    "side": "buy",
                    "time": datetime(2024, 1, 1, 12, 0, 0),
                    "trade_id": "trade1",
                    "maker": True
                },
                {
                    "price": 50001.0,
                    "size": 0.5,
                    "side": "sell",
                    "time": datetime(2024, 1, 1, 12, 0, 1),
                    "trade_id": "trade2",
                    "maker": False
                }
            ]
        )
    
    @pytest.fixture
    def classifier(self) -> MarketMakerPatternClassifier:
        """Создает экземпляр классификатора."""
        return MarketMakerPatternClassifier()
    
    def test_creation_with_default_config(self):
        """Тест создания с конфигурацией по умолчанию."""
        classifier = MarketMakerPatternClassifier()
        assert classifier.config is not None
        assert classifier.max_history_size == classifier.config.max_history_size
    
    def test_creation_with_custom_config(self):
        """Тест создания с пользовательской конфигурацией."""
        config = PatternClassifierConfig()
        config.min_confidence = Confidence(0.8)
        config.max_history_size = 500
        
        classifier = MarketMakerPatternClassifier(config)
        assert classifier.config == config
        assert classifier.max_history_size == 500
    
    def test_extract_features(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест извлечения признаков."""
        features = classifier.extract_features(sample_order_book_snapshot, sample_trade_snapshot)
        
        assert isinstance(features, PatternFeatures)
        assert isinstance(features.book_pressure, BookPressure)
        assert isinstance(features.volume_delta, VolumeDelta)
        assert isinstance(features.price_reaction, PriceReaction)
        assert isinstance(features.spread_change, SpreadChange)
        assert isinstance(features.order_imbalance, OrderImbalance)
        assert isinstance(features.liquidity_depth, LiquidityDepth)
        assert isinstance(features.time_duration, TimeDuration)
        assert isinstance(features.volume_concentration, VolumeConcentration)
        assert isinstance(features.price_volatility, PriceVolatility)
        assert isinstance(features.market_microstructure, dict)
    
    def test_classify_pattern_accumulation(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест классификации паттерна накопления."""
        # Настраиваем данные для паттерна накопления
        order_book = OrderBookSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            bids=[
                {"price": 50000.0, "size": 5.0, "orders_count": 10},
                {"price": 49999.0, "size": 3.0, "orders_count": 15}
            ],
            asks=[
                {"price": 50001.0, "size": 1.0, "orders_count": 12},
                {"price": 50002.0, "size": 1.0, "orders_count": 18}
            ],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        
        trades = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[
                {
                    "price": 50000.0,
                    "size": 2.0,
                    "side": "buy",
                    "time": datetime(2024, 1, 1, 12, 0, 0),
                    "trade_id": "trade1",
                    "maker": True
                },
                {
                    "price": 50001.0,
                    "size": 0.5,
                    "side": "sell",
                    "time": datetime(2024, 1, 1, 12, 0, 1),
                    "trade_id": "trade2",
                    "maker": False
                }
            ]
        )
        
        pattern = classifier.classify_pattern("BTC/USDT", order_book, trades)
        
        assert pattern is not None
        assert pattern.pattern_type == MarketMakerPatternType.ACCUMULATION
        assert pattern.symbol == "BTC/USDT"
        assert pattern.timestamp == order_book.timestamp
        assert isinstance(pattern.features, PatternFeatures)
        assert isinstance(pattern.confidence, Confidence)
        assert pattern.confidence >= classifier.config.min_confidence
    
    def test_classify_pattern_exit(self, classifier):
        """Тест классификации паттерна выхода."""
        # Настраиваем данные для паттерна выхода
        order_book = OrderBookSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            bids=[
                {"price": 50000.0, "size": 1.0, "orders_count": 10},
                {"price": 49999.0, "size": 1.0, "orders_count": 15}
            ],
            asks=[
                {"price": 50001.0, "size": 5.0, "orders_count": 12},
                {"price": 50002.0, "size": 3.0, "orders_count": 18}
            ],
            last_price=50000.5,
            volume_24h=1000.0,
            price_change_24h=0.02
        )
        
        trades = TradeSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC/USDT",
            trades=[
                {
                    "price": 50000.0,
                    "size": 0.5,
                    "side": "buy",
                    "time": datetime(2024, 1, 1, 12, 0, 0),
                    "trade_id": "trade1",
                    "maker": True
                },
                {
                    "price": 50001.0,
                    "size": 2.0,
                    "side": "sell",
                    "time": datetime(2024, 1, 1, 12, 0, 1),
                    "trade_id": "trade2",
                    "maker": False
                }
            ]
        )
        
        pattern = classifier.classify_pattern("BTC/USDT", order_book, trades)
        
        assert pattern is not None
        assert pattern.pattern_type == MarketMakerPatternType.EXIT
        assert pattern.symbol == "BTC/USDT"
        assert pattern.timestamp == order_book.timestamp
        assert isinstance(pattern.features, PatternFeatures)
        assert isinstance(pattern.confidence, Confidence)
        assert pattern.confidence >= classifier.config.min_confidence
    
    def test_classify_pattern_low_confidence(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест классификации паттерна с низкой уверенностью."""
        # Настраиваем классификатор с высокой минимальной уверенностью
        config = PatternClassifierConfig()
        config.min_confidence = Confidence(0.9)
        classifier = MarketMakerPatternClassifier(config)
        
        pattern = classifier.classify_pattern("BTC/USDT", sample_order_book_snapshot, sample_trade_snapshot)
        
        # Должен вернуть None из-за низкой уверенности
        assert pattern is None
    
    def test_update_history(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест обновления истории."""
        symbol = "BTC/USDT"
        
        # Проверяем, что история пуста изначально
        assert symbol not in classifier.order_book_history
        assert symbol not in classifier.trade_history
        
        # Обновляем историю
        classifier._update_history(symbol, sample_order_book_snapshot, sample_trade_snapshot)
        
        # Проверяем, что история обновлена
        assert symbol in classifier.order_book_history
        assert symbol in classifier.trade_history
        assert len(classifier.order_book_history[symbol]) == 1
        assert len(classifier.trade_history[symbol]) == 1
        assert classifier.order_book_history[symbol][0] == sample_order_book_snapshot
        assert classifier.trade_history[symbol][0] == sample_trade_snapshot
    
    def test_update_history_max_size(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест обновления истории с ограничением размера."""
        symbol = "BTC/USDT"
        max_size = 3
        
        # Создаем классификатор с ограниченным размером истории
        config = PatternClassifierConfig()
        config.max_history_size = max_size
        classifier = MarketMakerPatternClassifier(config)
        
        # Добавляем больше элементов, чем максимальный размер
        for i in range(max_size + 2):
            order_book = OrderBookSnapshot(
                timestamp=datetime(2024, 1, 1, 12, 0, i),
                symbol="BTC/USDT",
                bids=[{"price": 50000.0, "size": 1.0, "orders_count": 1}],
                asks=[{"price": 50001.0, "size": 1.0, "orders_count": 1}],
                last_price=50000.5,
                volume_24h=1000.0,
                price_change_24h=0.02
            )
            
            trades = TradeSnapshot(
                timestamp=datetime(2024, 1, 1, 12, 0, i),
                symbol="BTC/USDT",
                trades=[{
                    "price": 50000.0,
                    "size": 1.0,
                    "side": "buy",
                    "time": datetime(2024, 1, 1, 12, 0, i),
                    "trade_id": f"trade{i}",
                    "maker": True
                }]
            )
            
            classifier._update_history(symbol, order_book, trades)
        
        # Проверяем, что размер истории не превышает максимальный
        assert len(classifier.order_book_history[symbol]) <= max_size
        assert len(classifier.trade_history[symbol]) <= max_size
    
    def test_build_context(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест построения контекста."""
        symbol = "BTC/USDT"
        context = classifier._build_context(symbol, sample_order_book_snapshot, sample_trade_snapshot)
        
        assert context["symbol"] == symbol
        assert context["timestamp"] == sample_order_book_snapshot.timestamp.isoformat()
        assert context["last_price"] == sample_order_book_snapshot.last_price
        assert context["volume_24h"] == sample_order_book_snapshot.volume_24h
        assert context["price_change_24h"] == sample_order_book_snapshot.price_change_24h
    
    def test_determine_pattern_type_accumulation(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест определения типа паттерна (накопление)."""
        features = classifier.extract_features(sample_order_book_snapshot, sample_trade_snapshot)
        
        # Модифицируем признаки для паттерна накопления
        features = PatternFeatures(
            book_pressure=BookPressure(0.5),  # > 0.3
            volume_delta=VolumeDelta(0.3),    # > 0.2
            price_reaction=PriceReaction(0.1),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern_type, confidence = classifier._determine_pattern_type(features)
        
        assert pattern_type == MarketMakerPatternType.ACCUMULATION
        assert confidence == 0.8
    
    def test_determine_pattern_type_exit(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест определения типа паттерна (выход)."""
        features = classifier.extract_features(sample_order_book_snapshot, sample_trade_snapshot)
        
        # Модифицируем признаки для паттерна выхода
        features = PatternFeatures(
            book_pressure=BookPressure(-0.5),  # < -0.3
            volume_delta=VolumeDelta(0.3),     # > 0.2
            price_reaction=PriceReaction(0.1),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(-0.5),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern_type, confidence = classifier._determine_pattern_type(features)
        
        assert pattern_type == MarketMakerPatternType.EXIT
        assert confidence == 0.8
    
    def test_determine_pattern_type_default(self, classifier, sample_order_book_snapshot, sample_trade_snapshot):
        """Тест определения типа паттерна (по умолчанию)."""
        features = classifier.extract_features(sample_order_book_snapshot, sample_trade_snapshot)
        
        # Модифицируем признаки для паттерна по умолчанию
        features = PatternFeatures(
            book_pressure=BookPressure(0.1),   # Не подходит под накопление или выход
            volume_delta=VolumeDelta(0.1),     # Не подходит под накопление или выход
            price_reaction=PriceReaction(0.1),
            spread_change=SpreadChange(0.1),
            order_imbalance=OrderImbalance(0.1),
            liquidity_depth=LiquidityDepth(100.0),
            time_duration=TimeDuration(60),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.1)
        )
        
        pattern_type, confidence = classifier._determine_pattern_type(features)
        
        assert pattern_type == MarketMakerPatternType.ABSORPTION
        assert confidence == 0.5 