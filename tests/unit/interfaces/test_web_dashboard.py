#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Web Dashboard Interface.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import json
from decimal import Decimal

from interfaces.web_dashboard import WebDashboard, DashboardAPI
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.position import Position
from domain.exceptions import ValidationError


class TestWebDashboard:
    """Тесты для Web Dashboard."""

    @pytest.fixture
    def dashboard(self) -> WebDashboard:
        """Фикстура веб-дашборда."""
        return WebDashboard(host="localhost", port=8000, debug=True)

    @pytest.fixture
    def mock_trading_data(self) -> Dict[str, Any]:
        """Фикстура торговых данных."""
        return {
            "total_pnl": "1250.75",
            "daily_pnl": "125.50",
            "active_positions": 3,
            "win_rate": "65.8",
            "total_trades": 150,
            "successful_trades": 98
        }

    @pytest.fixture
    def mock_position_data(self) -> List[Dict[str, Any]]:
        """Фикстура данных позиций."""
        return [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "size": "0.5",
                "entry_price": "45000.00",
                "current_price": "45500.00",
                "pnl": "250.00",
                "pnl_percentage": "1.11"
            },
            {
                "symbol": "ETHUSDT", 
                "side": "SHORT",
                "size": "2.0",
                "entry_price": "3200.00",
                "current_price": "3150.00",
                "pnl": "100.00",
                "pnl_percentage": "1.56"
            }
        ]

    @pytest.fixture
    def mock_analytics_data(self) -> Dict[str, Any]:
        """Фикстура аналитических данных."""
        return {
            "rsi": {"BTCUSDT": 62.5, "ETHUSDT": 45.8},
            "macd": {"BTCUSDT": 0.15, "ETHUSDT": -0.08},
            "bollinger": {"BTCUSDT": {"upper": 46000, "lower": 44000}, "ETHUSDT": {"upper": 3250, "lower": 3100}},
            "ai_signals": [
                {"symbol": "BTCUSDT", "signal": "BUY", "strength": 0.75, "confidence": 0.82},
                {"symbol": "ETHUSDT", "signal": "HOLD", "strength": 0.45, "confidence": 0.60}
            ]
        }

    def test_dashboard_initialization(self, dashboard: WebDashboard) -> None:
        """Тест инициализации дашборда."""
        assert dashboard.host == "localhost"
        assert dashboard.port == 8000
        assert dashboard.debug is True
        assert hasattr(dashboard, 'app')
        assert hasattr(dashboard, 'websocket_clients')

    def test_dashboard_configuration(self, dashboard: WebDashboard) -> None:
        """Тест конфигурации дашборда."""
        config = dashboard.get_configuration()
        
        assert config["host"] == "localhost"
        assert config["port"] == 8000
        assert config["debug"] is True
        assert "cors_enabled" in config
        assert "websocket_enabled" in config

    @pytest.mark.asyncio
    async def test_status_endpoint(self, dashboard: WebDashboard) -> None:
        """Тест endpoint статуса системы."""
        mock_status = {
            "status": "online",
            "uptime": "02:45:30",
            "cpu_usage": 45.2,
            "memory_usage": 68.5,
            "active_connections": 12
        }
        
        with patch.object(dashboard, '_get_system_status', return_value=mock_status):
            response = await dashboard.get_status()
            
            assert response["status"] == "online"
            assert response["uptime"] == "02:45:30"
            assert response["cpu_usage"] == 45.2
            assert response["memory_usage"] == 68.5
            assert response["active_connections"] == 12

    @pytest.mark.asyncio
    async def test_trading_data_endpoint(self, dashboard: WebDashboard, mock_trading_data: Dict[str, Any]) -> None:
        """Тест endpoint торговых данных."""
        with patch.object(dashboard, '_get_trading_data', return_value=mock_trading_data):
            response = await dashboard.get_trading_data()
            
            assert response["total_pnl"] == "1250.75"
            assert response["daily_pnl"] == "125.50"
            assert response["active_positions"] == 3
            assert response["win_rate"] == "65.8"
            assert response["total_trades"] == 150

    @pytest.mark.asyncio
    async def test_positions_endpoint(self, dashboard: WebDashboard, mock_position_data: List[Dict[str, Any]]) -> None:
        """Тест endpoint позиций."""
        with patch.object(dashboard, '_get_positions_data', return_value=mock_position_data):
            response = await dashboard.get_positions()
            
            assert len(response) == 2
            assert response[0]["symbol"] == "BTCUSDT"
            assert response[0]["side"] == "LONG"
            assert response[0]["pnl"] == "250.00"
            assert response[1]["symbol"] == "ETHUSDT"
            assert response[1]["side"] == "SHORT"

    @pytest.mark.asyncio
    async def test_analytics_endpoint(self, dashboard: WebDashboard, mock_analytics_data: Dict[str, Any]) -> None:
        """Тест endpoint аналитики."""
        with patch.object(dashboard, '_get_analytics_data', return_value=mock_analytics_data):
            response = await dashboard.get_analytics()
            
            assert "rsi" in response
            assert "macd" in response
            assert "ai_signals" in response
            assert response["rsi"]["BTCUSDT"] == 62.5
            assert len(response["ai_signals"]) == 2

    @pytest.mark.asyncio
    async def test_health_endpoint(self, dashboard: WebDashboard) -> None:
        """Тест endpoint проверки здоровья."""
        mock_health = {
            "api_status": "healthy",
            "database_status": "healthy",
            "exchange_connections": "healthy",
            "memory_usage": "normal",
            "last_update": "2023-12-01T12:00:00Z"
        }
        
        with patch.object(dashboard, '_check_health', return_value=mock_health):
            response = await dashboard.get_health()
            
            assert response["api_status"] == "healthy"
            assert response["database_status"] == "healthy"
            assert response["exchange_connections"] == "healthy"

    @pytest.mark.asyncio
    async def test_websocket_connection(self, dashboard: WebDashboard) -> None:
        """Тест WebSocket соединения."""
        mock_websocket = AsyncMock()
        
        # Симулируем подключение WebSocket клиента
        await dashboard.handle_websocket_connect(mock_websocket)
        
        assert mock_websocket in dashboard.websocket_clients
        assert len(dashboard.websocket_clients) == 1

    @pytest.mark.asyncio
    async def test_websocket_disconnection(self, dashboard: WebDashboard) -> None:
        """Тест отключения WebSocket."""
        mock_websocket = AsyncMock()
        
        # Подключаем и затем отключаем
        await dashboard.handle_websocket_connect(mock_websocket)
        await dashboard.handle_websocket_disconnect(mock_websocket)
        
        assert mock_websocket not in dashboard.websocket_clients
        assert len(dashboard.websocket_clients) == 0

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self, dashboard: WebDashboard) -> None:
        """Тест broadcast сообщений через WebSocket."""
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        # Подключаем несколько клиентов
        await dashboard.handle_websocket_connect(mock_websocket1)
        await dashboard.handle_websocket_connect(mock_websocket2)
        
        test_data = {"type": "update", "data": {"test": "value"}}
        
        # Отправляем broadcast
        await dashboard.broadcast_websocket_message(test_data)
        
        # Проверяем, что сообщение отправлено всем клиентам
        mock_websocket1.send_text.assert_called_once()
        mock_websocket2.send_text.assert_called_once()
        
        sent_data = json.loads(mock_websocket1.send_text.call_args[0][0])
        assert sent_data["type"] == "update"
        assert sent_data["data"]["test"] == "value"

    @pytest.mark.asyncio
    async def test_real_time_updates(self, dashboard: WebDashboard) -> None:
        """Тест real-time обновлений."""
        mock_websocket = AsyncMock()
        await dashboard.handle_websocket_connect(mock_websocket)
        
        # Симулируем обновление данных
        update_data = {
            "type": "trading_update",
            "timestamp": "2023-12-01T12:00:00Z",
            "data": {
                "total_pnl": "1300.00",
                "daily_pnl": "150.25"
            }
        }
        
        await dashboard.send_real_time_update(update_data)
        
        mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_message["type"] == "trading_update"

    def test_data_validation(self, dashboard: WebDashboard) -> None:
        """Тест валидации данных."""
        # Валидные данные
        valid_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "size": "0.5",
            "price": "45000.00"
        }
        
        assert dashboard._validate_position_data(valid_data) is True
        
        # Невалидные данные
        invalid_data = {
            "symbol": "",
            "side": "INVALID",
            "size": "-0.5",
            "price": "0"
        }
        
        with pytest.raises(ValidationError):
            dashboard._validate_position_data(invalid_data)

    def test_data_formatting(self, dashboard: WebDashboard) -> None:
        """Тест форматирования данных."""
        raw_position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "size": Decimal("0.5"),
            "entry_price": Decimal("45000.00"),
            "current_price": Decimal("45500.00")
        }
        
        formatted = dashboard._format_position_data(raw_position)
        
        assert formatted["symbol"] == "BTCUSDT"
        assert formatted["side"] == "LONG"
        assert formatted["size"] == "0.5"
        assert formatted["entry_price"] == "45000.00"
        assert formatted["current_price"] == "45500.00"
        assert "pnl" in formatted
        assert "pnl_percentage" in formatted

    def test_error_handling(self, dashboard: WebDashboard) -> None:
        """Тест обработки ошибок."""
        # Симулируем ошибку получения данных
        with patch.object(dashboard, '_get_trading_data', side_effect=Exception("Database error")):
            with pytest.raises(Exception, match="Database error"):
                dashboard.get_trading_data()

    def test_security_headers(self, dashboard: WebDashboard) -> None:
        """Тест security заголовков."""
        headers = dashboard._get_security_headers()
        
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_cors_configuration(self, dashboard: WebDashboard) -> None:
        """Тест настройки CORS."""
        cors_config = dashboard._get_cors_configuration()
        
        assert cors_config["allow_origins"] is not None
        assert cors_config["allow_methods"] is not None
        assert cors_config["allow_headers"] is not None
        assert "GET" in cors_config["allow_methods"]
        assert "POST" in cors_config["allow_methods"]

    @pytest.mark.asyncio
    async def test_rate_limiting(self, dashboard: WebDashboard) -> None:
        """Тест ограничения скорости запросов."""
        client_ip = "192.168.1.100"
        
        # Симулируем множественные запросы
        for i in range(100):
            is_allowed = await dashboard._check_rate_limit(client_ip)
            if i < 50:  # Первые 50 запросов должны пройти
                assert is_allowed is True
            else:  # Следующие должны быть заблокированы
                assert is_allowed is False

    def test_authentication(self, dashboard: WebDashboard) -> None:
        """Тест аутентификации."""
        # Валидный токен
        valid_token = "valid_jwt_token_here"
        assert dashboard._validate_auth_token(valid_token) is True
        
        # Невалидный токен
        invalid_token = "invalid_token"
        assert dashboard._validate_auth_token(invalid_token) is False
        
        # Отсутствующий токен
        assert dashboard._validate_auth_token(None) is False

    def test_session_management(self, dashboard: WebDashboard) -> None:
        """Тест управления сессиями."""
        session_id = dashboard._create_session("test_user")
        
        assert isinstance(session_id, str)
        assert len(session_id) > 10
        
        # Проверка сессии
        assert dashboard._validate_session(session_id) is True
        
        # Удаление сессии
        dashboard._destroy_session(session_id)
        assert dashboard._validate_session(session_id) is False

    @pytest.mark.asyncio
    async def test_caching(self, dashboard: WebDashboard) -> None:
        """Тест кэширования данных."""
        cache_key = "trading_data"
        test_data = {"total_pnl": "1000.00"}
        
        # Установка кэша
        await dashboard._set_cache(cache_key, test_data, ttl=60)
        
        # Получение из кэша
        cached_data = await dashboard._get_cache(cache_key)
        assert cached_data == test_data
        
        # Очистка кэша
        await dashboard._clear_cache(cache_key)
        cached_data = await dashboard._get_cache(cache_key)
        assert cached_data is None

    def test_logging(self, dashboard: WebDashboard) -> None:
        """Тест логирования."""
        with patch('logging.getLogger') as mock_logger:
            dashboard._log_request("GET", "/api/status", "192.168.1.1", 200)
            
            mock_logger.return_value.info.assert_called_once()
            call_args = mock_logger.return_value.info.call_args[0][0]
            assert "GET" in call_args
            assert "/api/status" in call_args
            assert "200" in call_args

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, dashboard: WebDashboard) -> None:
        """Тест мониторинга производительности."""
        # Симулируем выполнение запроса с мониторингом
        async def mock_handler():
            import asyncio
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        with dashboard._performance_monitor("test_endpoint"):
            result = await mock_handler()
        
        metrics = dashboard.get_performance_metrics()
        assert "test_endpoint" in metrics
        assert metrics["test_endpoint"]["count"] == 1
        assert metrics["test_endpoint"]["average_time"] > 0

    def test_static_file_serving(self, dashboard: WebDashboard) -> None:
        """Тест обслуживания статических файлов."""
        # Проверяем, что статические маршруты настроены
        static_routes = dashboard._get_static_routes()
        
        assert "/static" in static_routes
        assert "/assets" in static_routes
        assert static_routes["/static"]["directory"] is not None

    @pytest.mark.asyncio
    async def test_error_responses(self, dashboard: WebDashboard) -> None:
        """Тест ответов на ошибки."""
        # 404 ошибка
        response_404 = await dashboard._handle_404_error()
        assert response_404["status_code"] == 404
        assert response_404["message"] == "Resource not found"
        
        # 500 ошибка
        response_500 = await dashboard._handle_500_error("Internal error")
        assert response_500["status_code"] == 500
        assert "Internal error" in response_500["message"]

    def test_api_documentation(self, dashboard: WebDashboard) -> None:
        """Тест генерации API документации."""
        api_docs = dashboard._generate_api_documentation()
        
        assert "endpoints" in api_docs
        assert "version" in api_docs
        assert len(api_docs["endpoints"]) > 0
        
        # Проверяем наличие основных endpoints
        endpoint_paths = [ep["path"] for ep in api_docs["endpoints"]]
        assert "/api/status" in endpoint_paths
        assert "/api/trading" in endpoint_paths
        assert "/api/positions" in endpoint_paths

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, dashboard: WebDashboard) -> None:
        """Тест корректного завершения работы."""
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        # Подключаем клиентов
        await dashboard.handle_websocket_connect(mock_websocket1)
        await dashboard.handle_websocket_connect(mock_websocket2)
        
        # Инициируем graceful shutdown
        await dashboard.graceful_shutdown()
        
        # Проверяем, что все WebSocket соединения закрыты
        mock_websocket1.close.assert_called_once()
        mock_websocket2.close.assert_called_once()
        assert len(dashboard.websocket_clients) == 0

    def test_configuration_validation(self, dashboard: WebDashboard) -> None:
        """Тест валидации конфигурации."""
        # Валидная конфигурация
        valid_config = {
            "host": "localhost",
            "port": 8000,
            "debug": True,
            "cors_enabled": True
        }
        
        assert dashboard._validate_configuration(valid_config) is True
        
        # Невалидная конфигурация
        invalid_config = {
            "host": "",
            "port": -1,
            "debug": "not_boolean"
        }
        
        with pytest.raises(ValidationError):
            dashboard._validate_configuration(invalid_config)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, dashboard: WebDashboard) -> None:
        """Тест обработки одновременных запросов."""
        import asyncio
        
        async def make_request():
            return await dashboard.get_status()
        
        # Создаем множественные одновременные запросы
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Все запросы должны быть успешными
        assert len(results) == 10
        assert all("status" in result for result in results)

    def test_memory_usage(self, dashboard: WebDashboard) -> None:
        """Тест использования памяти."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Симулируем нагрузку
        for _ in range(1000):
            dashboard._format_position_data({
                "symbol": "BTCUSDT",
                "side": "long",
                "size": Decimal("0.5"),
                "entry_price": Decimal("45000.00"),
                "current_price": Decimal("45500.00")
            })
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Увеличение памяти должно быть разумным
        assert memory_increase < 50 * 1024 * 1024  # Менее 50MB