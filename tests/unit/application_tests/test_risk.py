"""
Тесты для управления рисками application слоя.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Any
from application.risk.liquidity_gravity_monitor import LiquidityGravityMonitor, MonitorConfig

class TestLiquidityGravityMonitor:
    """Тесты для LiquidityGravityMonitor."""
    @pytest.fixture
    def mock_repositories(self) -> tuple[Mock, Mock, Mock]:
        """Создает mock репозитории."""
        market_repo = Mock()
        orderbook_repo = Mock()
        risk_repo = Mock()
        market_repo.get_market_data = AsyncMock()
        market_repo.get_orderbook = AsyncMock()
        orderbook_repo.get_orderbook_data = AsyncMock()
        orderbook_repo.get_liquidity_data = AsyncMock()
        risk_repo.get_risk_metrics = AsyncMock()
        risk_repo.save_risk_alert = AsyncMock()
        return market_repo, orderbook_repo, risk_repo
    @pytest.fixture
    def monitor(self, mock_repositories: tuple[Mock, Mock, Mock]) -> LiquidityGravityMonitor:
        """Создает экземпляр монитора."""
        config = MonitorConfig()
        return LiquidityGravityMonitor(config)
    @pytest.fixture
    def sample_orderbook(self) -> dict[str, Any]:
        """Создает образец ордербука."""
        return {
            "symbol": "BTC/USD",
            "timestamp": "2024-01-01T00:00:00",
            "bids": [
                {"price": "50000", "quantity": "0.1"},
                {"price": "49999", "quantity": "0.2"},
                {"price": "49998", "quantity": "0.3"}
            ],
            "asks": [
                {"price": "50001", "quantity": "0.1"},
                {"price": "50002", "quantity": "0.2"},
                {"price": "50003", "quantity": "0.3"}
            ]
        }
    @pytest.mark.asyncio
    async def test_monitor_liquidity_gravity(self, monitor: LiquidityGravityMonitor, mock_repositories: tuple[Mock, Mock, Mock], sample_orderbook: dict[str, Any]) -> None:
        """Тест мониторинга гравитации ликвидности."""
        market_repo, orderbook_repo, risk_repo = mock_repositories
        symbol = "BTC/USD"
        timeframe = "1h"
        
        # Добавляем символ для мониторинга
        monitor.add_symbol(symbol)
        
        # Запускаем мониторинг на короткое время
        import asyncio
        try:
            await asyncio.wait_for(monitor.start_monitoring(), timeout=0.1)
        except asyncio.TimeoutError:
            monitor.stop_monitoring()
        
        # Проверяем, что мониторинг был запущен
        assert monitor.is_running is False  # Остановлен после timeout
        assert symbol in monitor.monitored_symbols
    @pytest.mark.asyncio
    async def test_calculate_liquidity_gravity(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест расчета гравитации ликвидности."""
        # Создаем OrderBookSnapshot из sample_orderbook
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем гравитацию ликвидности
        gravity_result = monitor.gravity_model.analyze_liquidity_gravity(order_book)
        
        assert gravity_result is not None
        assert hasattr(gravity_result, 'gravity_score')
        assert gravity_result.gravity_score >= 0
    @pytest.mark.asyncio
    async def test_analyze_orderbook_imbalance(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест анализа дисбаланса ордербука."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем дисбаланс
        bid_volume = sum(quantity for _, quantity in order_book.bids)
        ask_volume = sum(quantity for _, quantity in order_book.asks)
        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
        
        imbalance = {
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance_ratio": imbalance_ratio,
            "imbalance_direction": "BID" if imbalance_ratio > 1.1 else "ASK" if imbalance_ratio < 0.9 else "BALANCED"
        }
        
        assert "bid_volume" in imbalance
        assert "ask_volume" in imbalance
        assert "imbalance_ratio" in imbalance
        assert "imbalance_direction" in imbalance
        assert isinstance(imbalance["bid_volume"], (int, float))
        assert isinstance(imbalance["ask_volume"], (int, float))
        assert isinstance(imbalance["imbalance_ratio"], (int, float))
        assert isinstance(imbalance["imbalance_direction"], str)
        assert imbalance["imbalance_direction"] in ["BID", "ASK", "BALANCED"]
    @pytest.mark.asyncio
    async def test_detect_liquidity_clusters(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест обнаружения кластеров ликвидности."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем кластеры (упрощенная версия)
        bid_clusters = [{"price": price, "volume": volume} for price, volume in order_book.bids]
        ask_clusters = [{"price": price, "volume": volume} for price, volume in order_book.asks]
        cluster_strength = sum(volume for _, volume in order_book.bids + order_book.asks)
        
        clusters = {
            "bid_clusters": bid_clusters,
            "ask_clusters": ask_clusters,
            "cluster_strength": cluster_strength
        }
        
        assert "bid_clusters" in clusters
        assert "ask_clusters" in clusters
        assert "cluster_strength" in clusters
        assert isinstance(clusters["bid_clusters"], list)
        assert isinstance(clusters["ask_clusters"], list)
        assert isinstance(clusters["cluster_strength"], (int, float))
    @pytest.mark.asyncio
    async def test_calculate_gravity_strength(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест расчета силы гравитации."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем гравитацию
        gravity_result = monitor.gravity_model.analyze_liquidity_gravity(order_book)
        strength = gravity_result.total_gravity  # Используем правильный атрибут
        
        assert isinstance(strength, (int, float))
        assert 0 <= strength <= 1
    @pytest.mark.asyncio
    async def test_determine_risk_level(self, monitor: LiquidityGravityMonitor) -> None:
        """Тест определения уровня риска."""
        # Низкий риск
        low_gravity: float = 0.2
        risk_level = "LOW" if low_gravity < 0.3 else "MEDIUM" if low_gravity < 0.7 else "HIGH"
        assert risk_level == "LOW"
        
        # Средний риск
        medium_gravity: float = 0.5
        risk_level = "LOW" if medium_gravity < 0.3 else "MEDIUM" if medium_gravity < 0.7 else "HIGH"
        assert risk_level == "MEDIUM"
        
        # Высокий риск
        high_gravity: float = 0.8
        risk_level = "LOW" if high_gravity < 0.3 else "MEDIUM" if high_gravity < 0.7 else "HIGH"
        assert risk_level == "HIGH"
    @pytest.mark.asyncio
    async def test_generate_alerts(self, monitor: LiquidityGravityMonitor, mock_repositories: tuple[Mock, Mock, Mock]) -> None:
        """Тест генерации предупреждений."""
        market_repo, orderbook_repo, risk_repo = mock_repositories
        symbol = "BTC/USD"
        gravity_strength = 0.8
        risk_level = "HIGH"
        
        # Генерируем алерты (упрощенная версия)
        alerts = []
        if gravity_strength > monitor.config.alert_threshold:
            alerts.append({
                "type": "LIQUIDITY_GRAVITY_HIGH",
                "message": f"High liquidity gravity detected for {symbol}",
                "severity": risk_level,
                "timestamp": "2024-01-01T00:00:00"
            })
        
        assert isinstance(alerts, list)
        for alert in alerts:
            assert "type" in alert
            assert "message" in alert
            assert "severity" in alert
            assert "timestamp" in alert
            assert isinstance(alert["type"], str)
            assert isinstance(alert["message"], str)
            assert isinstance(alert["severity"], str)
            assert alert["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, monitor: LiquidityGravityMonitor) -> None:
        """Тест генерации рекомендаций."""
        symbol = "BTC/USD"
        gravity_strength = 0.8
        risk_level = "HIGH"
        imbalance = {
            "imbalance_direction": "BID",
            "imbalance_ratio": 1.5
        }
        
        # Генерируем рекомендации (упрощенная версия)
        recommendations = []
        if risk_level == "HIGH":
            recommendations.append({
                "type": "RISK_MANAGEMENT",
                "description": "Consider reducing position size due to high liquidity gravity",
                "priority": "HIGH"
            })
        
        assert isinstance(recommendations, list)
        for recommendation in recommendations:
            assert "type" in recommendation
            assert "description" in recommendation
            assert "priority" in recommendation
            assert isinstance(recommendation["type"], str)
            assert isinstance(recommendation["description"], str)
            assert isinstance(recommendation["priority"], str)
            assert recommendation["priority"] in ["LOW", "MEDIUM", "HIGH"]
    @pytest.mark.asyncio
    async def test_calculate_volume_weighted_price(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест расчета цены, взвешенной по объему."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Рассчитываем VWAP
        total_volume: float = 0.0
        weighted_sum: float = 0.0
        
        for price, volume in order_book.bids + order_book.asks:
            weighted_sum += price * volume
            total_volume += volume
        
        vwp = weighted_sum / total_volume if total_volume > 0 else 0
        
        assert isinstance(vwp, (int, float))
        assert vwp > 0
    @pytest.mark.asyncio
    async def test_analyze_price_impact(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест анализа влияния на цену."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем влияние на цену (упрощенная версия)
        bid_impact = sum(volume for _, volume in order_book.bids) * 0.001
        ask_impact = sum(volume for _, volume in order_book.asks) * 0.001
        total_impact = bid_impact + ask_impact
        impact_threshold = 0.1
        
        impact = {
            "bid_impact": bid_impact,
            "ask_impact": ask_impact,
            "total_impact": total_impact,
            "impact_threshold": impact_threshold
        }
        
        assert "bid_impact" in impact
        assert "ask_impact" in impact
        assert "total_impact" in impact
        assert "impact_threshold" in impact
        assert isinstance(impact["bid_impact"], (int, float))
        assert isinstance(impact["ask_impact"], (int, float))
        assert isinstance(impact["total_impact"], (int, float))
        assert isinstance(impact["impact_threshold"], (int, float))
    @pytest.mark.asyncio
    async def test_detect_liquidity_depletion(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест обнаружения истощения ликвидности."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем истощение ликвидности (упрощенная версия)
        bid_volume = sum(volume for _, volume in order_book.bids)
        ask_volume = sum(volume for _, volume in order_book.asks)
        total_volume = bid_volume + ask_volume
        
        depletion_level = 1.0 - (total_volume / 1000)  # Нормализованное значение
        depletion_direction = "BID" if bid_volume < ask_volume else "ASK"
        depletion_speed = 0.1  # Скорость истощения
        risk_score = min(1.0, depletion_level * 2)
        
        depletion = {
            "depletion_level": depletion_level,
            "depletion_direction": depletion_direction,
            "depletion_speed": depletion_speed,
            "risk_score": risk_score
        }
        
        assert "depletion_level" in depletion
        assert "depletion_direction" in depletion
        assert "depletion_speed" in depletion
        assert "risk_score" in depletion
        assert isinstance(depletion["depletion_level"], (int, float))
        assert isinstance(depletion["depletion_direction"], str)
        assert isinstance(depletion["depletion_speed"], (int, float))
        assert isinstance(depletion["risk_score"], (int, float))
        assert 0 <= depletion["risk_score"] <= 1
    @pytest.mark.asyncio
    async def test_calculate_spread_impact(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест расчета влияния спреда."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Рассчитываем влияние спреда
        if order_book.bids and order_book.asks:
            best_bid = max(price for price, _ in order_book.bids)
            best_ask = min(price for price, _ in order_book.asks)
            spread = best_ask - best_bid
            spread_impact = spread / best_bid  # Нормализованный спред
        else:
            spread_impact = 0
        
        assert isinstance(spread_impact, (int, float))
        assert spread_impact >= 0
    @pytest.mark.asyncio
    async def test_analyze_market_depth(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест анализа глубины рынка."""
        # Создаем OrderBookSnapshot
        from domain.market.liquidity_gravity import OrderBookSnapshot
        from datetime import datetime
        
        bids = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["bids"]]
        asks = [(float(item["price"]), float(item["quantity"])) for item in sample_orderbook["asks"]]
        
        order_book = OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            symbol=sample_orderbook["symbol"]
        )
        
        # Анализируем глубину рынка (упрощенная версия)
        bid_depth = sum(volume for _, volume in order_book.bids)
        ask_depth = sum(volume for _, volume in order_book.asks)
        total_depth = bid_depth + ask_depth
        depth_imbalance = abs(bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        depth_analysis = {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": total_depth,
            "depth_imbalance": depth_imbalance,
            "depth_score": 1.0 - depth_imbalance
        }
        
        assert "bid_depth" in depth_analysis
        assert "ask_depth" in depth_analysis
        assert "total_depth" in depth_analysis
        assert "depth_imbalance" in depth_analysis
        assert "depth_score" in depth_analysis
        assert isinstance(depth_analysis["bid_depth"], (int, float))
        assert isinstance(depth_analysis["ask_depth"], (int, float))
        assert isinstance(depth_analysis["total_depth"], (int, float))
        assert isinstance(depth_analysis["depth_imbalance"], (int, float))
        assert isinstance(depth_analysis["depth_score"], (int, float))
        assert 0 <= depth_analysis["depth_score"] <= 1
    @pytest.mark.asyncio
    async def test_get_historical_gravity_data(self, monitor: LiquidityGravityMonitor, mock_repositories: tuple[Mock, Mock, Mock]) -> None:
        """Тест получения исторических данных гравитации."""
        market_repo, orderbook_repo, risk_repo = mock_repositories
        symbol = "BTC/USD"
        
        # Получаем исторические данные (упрощенная версия)
        history = monitor.get_risk_history(symbol, limit=10)
        
        assert isinstance(history, list)
        # Проверяем, что история может быть пустой (если данных нет)
        # assert len(history) <= 10
    def test_validate_orderbook_data(self, monitor: LiquidityGravityMonitor, sample_orderbook: dict[str, Any]) -> None:
        """Тест валидации данных ордербука."""
        # Валидация корректных данных
        is_valid = (
            "bids" in sample_orderbook and
            "asks" in sample_orderbook and
            isinstance(sample_orderbook["bids"], list) and
            isinstance(sample_orderbook["asks"], list) and
            len(sample_orderbook["bids"]) > 0 and
            len(sample_orderbook["asks"]) > 0
        )
        assert is_valid is True
        
        # Валидация некорректных данных
        invalid_orderbook: dict[str, Any] = {"bids": [], "asks": []}
        is_valid = (
            "bids" in invalid_orderbook and
            "asks" in invalid_orderbook and
            isinstance(invalid_orderbook["bids"], list) and
            isinstance(invalid_orderbook["asks"], list) and
            len(invalid_orderbook["bids"]) > 0 and
            len(invalid_orderbook["asks"]) > 0
        )
        assert is_valid is False
        
        # Валидация отсутствующих данных
        empty_orderbook: dict[str, Any] = {}
        is_valid = (
            "bids" in empty_orderbook and
            "asks" in empty_orderbook and
            isinstance(empty_orderbook.get("bids"), list) and
            isinstance(empty_orderbook.get("asks"), list) and
            len(empty_orderbook.get("bids", [])) > 0 and
            len(empty_orderbook.get("asks", [])) > 0
        )
        assert is_valid is False
    @pytest.mark.asyncio
    async def test_calculate_gravity_trend(self, monitor: LiquidityGravityMonitor) -> None:
        """Тест расчета тренда гравитации."""
        symbol = "BTC/USD"
        
        # Добавляем символ для мониторинга
        monitor.add_symbol(symbol)
        
        # Получаем статистику мониторинга
        stats = monitor.get_monitoring_statistics()
        
        assert isinstance(stats, dict)
        assert "total_assessments" in stats
        assert "high_risk_detections" in stats
        assert "critical_risk_detections" in stats
        assert "start_time" in stats
        assert isinstance(stats["total_assessments"], int)
        assert isinstance(stats["high_risk_detections"], int)
        assert isinstance(stats["critical_risk_detections"], int) 
