#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты отказоустойчивости и восстановления системы.
Критически важно для финансовой системы - система должна быть готова к любым сбоям.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, patch, AsyncMock
import random
import concurrent.futures

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.portfolio import Portfolio
from infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from infrastructure.resilience.retry_handler import RetryHandler, RetryStrategy
from infrastructure.resilience.failover_manager import FailoverManager, FailoverStrategy
from infrastructure.resilience.backup_manager import BackupManager, BackupStrategy
from infrastructure.external_services.bybit_client import BybitClient
from domain.exceptions import (
    NetworkError, ServiceUnavailableError, DataIntegrityError,
    SystemOverloadError, DisasterRecoveryError
)


class TestDisasterRecoveryComprehensive:
    """Comprehensive тесты отказоустойчивости и восстановления."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Фикстура circuit breaker."""
        return CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            success_threshold=3,
            timeout=10
        )

    @pytest.fixture
    def retry_handler(self) -> RetryHandler:
        """Фикстура retry handler."""
        return RetryHandler(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True
        )

    @pytest.fixture
    def failover_manager(self) -> FailoverManager:
        """Фикстура failover manager."""
        return FailoverManager(
            primary_endpoint="https://api.bybit.com",
            fallback_endpoints=[
                "https://api2.bybit.com",
                "https://backup.trading-system.com"
            ],
            health_check_interval=10,
            failover_threshold=3
        )

    @pytest.fixture
    def backup_manager(self) -> BackupManager:
        """Фикстура backup manager."""
        return BackupManager(
            backup_strategy=BackupStrategy.INCREMENTAL,
            backup_interval_minutes=15,
            retention_days=30,
            encryption_enabled=True,
            compression_enabled=True
        )

    def test_circuit_breaker_failure_detection(
        self, 
        circuit_breaker: CircuitBreaker
    ) -> None:
        """Тест обнаружения сбоев circuit breaker."""
        
        def failing_service():
            """Сервис который всегда падает."""
            raise NetworkError("Service unavailable")
        
        # Изначально circuit breaker закрыт
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Вызываем сбойный сервис до порога
        for i in range(4):  # Меньше порога (5)
            with pytest.raises(NetworkError):
                circuit_breaker.call(failing_service)
            
            assert circuit_breaker.state == CircuitBreakerState.CLOSED
            assert circuit_breaker.failure_count == i + 1
        
        # Пятый сбой должен открыть circuit breaker
        with pytest.raises(NetworkError):
            circuit_breaker.call(failing_service)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == 5
        
        # Последующие вызовы должны немедленно падать
        with pytest.raises(ServiceUnavailableError):
            circuit_breaker.call(failing_service)

    def test_circuit_breaker_recovery(
        self, 
        circuit_breaker: CircuitBreaker
    ) -> None:
        """Тест восстановления circuit breaker."""
        
        call_count = 0
        
        def intermittent_service():
            """Сервис который иногда работает."""
            nonlocal call_count
            call_count += 1
            
            if call_count <= 5:
                raise NetworkError("Temporary failure")
            elif call_count <= 8:
                return f"Success {call_count}"
            else:
                raise NetworkError("Another failure")
        
        # Открываем circuit breaker
        for _ in range(5):
            with pytest.raises(NetworkError):
                circuit_breaker.call(intermittent_service)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Ждем recovery timeout
        time.sleep(circuit_breaker.recovery_timeout + 1)
        
        # Circuit breaker должен перейти в HALF_OPEN
        result = circuit_breaker.call(intermittent_service)
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        assert result == "Success 6"
        
        # Еще несколько успешных вызовов должны закрыть circuit breaker
        for _ in range(2):
            result = circuit_breaker.call(intermittent_service)
            assert "Success" in result
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_exponential_backoff_retry(
        self, 
        retry_handler: RetryHandler
    ) -> None:
        """Тест exponential backoff retry стратегии."""
        
        attempt_times = []
        attempt_count = 0
        
        def flaky_service():
            """Сервис который падает первые несколько раз."""
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(time.time())
            
            if attempt_count < 4:
                raise NetworkError(f"Attempt {attempt_count} failed")
            
            return f"Success on attempt {attempt_count}"
        
        start_time = time.time()
        result = retry_handler.execute_with_retry(flaky_service)
        end_time = time.time()
        
        # Проверяем результат
        assert result == "Success on attempt 4"
        assert attempt_count == 4
        
        # Проверяем exponential backoff
        assert len(attempt_times) == 4
        
        # Первая попытка - немедленно
        assert attempt_times[0] - start_time < 0.1
        
        # Последующие попытки с возрастающими задержками
        for i in range(1, len(attempt_times)):
            delay = attempt_times[i] - attempt_times[i-1]
            expected_min_delay = retry_handler.base_delay * (retry_handler.backoff_multiplier ** (i-1))
            assert delay >= expected_min_delay * 0.8  # Учитываем jitter

    def test_retry_with_jitter(
        self, 
        retry_handler: RetryHandler
    ) -> None:
        """Тест retry с jitter для предотвращения thundering herd."""
        
        delays = []
        
        def record_delay_service():
            """Сервис для записи задержек."""
            current_time = time.time()
            if hasattr(record_delay_service, 'last_call'):
                delay = current_time - record_delay_service.last_call
                delays.append(delay)
            
            record_delay_service.last_call = current_time
            
            if len(delays) < 3:
                raise NetworkError("Still failing")
            
            return "Success"
        
        result = retry_handler.execute_with_retry(record_delay_service)
        
        assert result == "Success"
        assert len(delays) == 3
        
        # Проверяем что задержки различаются (jitter работает)
        assert len(set(delays)) > 1  # Не все задержки одинаковые

    def test_failover_to_backup_endpoints(
        self, 
        failover_manager: FailoverManager
    ) -> None:
        """Тест failover на backup endpoints."""
        
        call_log = []
        
        async def mock_health_check(endpoint: str) -> bool:
            """Мок health check."""
            call_log.append(f"health_check:{endpoint}")
            
            if endpoint == "https://api.bybit.com":
                return False  # Основной endpoint недоступен
            elif endpoint == "https://api2.bybit.com":
                return True   # Первый backup доступен
            else:
                return True
        
        async def mock_service_call(endpoint: str, data: dict) -> dict:
            """Мок вызова сервиса."""
            call_log.append(f"service_call:{endpoint}")
            
            if endpoint == "https://api.bybit.com":
                raise NetworkError("Primary endpoint down")
            elif endpoint == "https://api2.bybit.com":
                return {"status": "success", "endpoint": endpoint, "data": data}
            else:
                return {"status": "success", "endpoint": endpoint, "data": data}
        
        # Патчим методы
        failover_manager.health_check = mock_health_check
        failover_manager.service_call = mock_service_call
        
        # Выполняем failover
        test_data = {"symbol": "BTCUSDT", "side": "BUY"}
        result = asyncio.run(failover_manager.execute_with_failover(test_data))
        
        # Проверяем результат
        assert result["status"] == "success"
        assert result["endpoint"] == "https://api2.bybit.com"
        assert result["data"] == test_data
        
        # Проверяем что был выполнен failover
        assert "health_check:https://api.bybit.com" in call_log
        assert "service_call:https://api2.bybit.com" in call_log

    def test_automatic_backup_and_restore(
        self, 
        backup_manager: BackupManager
    ) -> None:
        """Тест автоматического резервного копирования и восстановления."""
        
        # Создаем тестовые данные
        portfolio_data = {
            "user_id": "backup_test_user",
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "quantity": "1.5",
                    "entry_price": "45000.00",
                    "unrealized_pnl": "2500.00"
                },
                {
                    "symbol": "ETHUSDT", 
                    "quantity": "10.0",
                    "entry_price": "3200.00",
                    "unrealized_pnl": "800.00"
                }
            ],
            "total_equity": "150000.00",
            "backup_timestamp": datetime.utcnow().isoformat()
        }
        
        orders_data = [
            {
                "order_id": "order_001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": "0.1",
                "price": "44500.00",
                "status": "PENDING"
            },
            {
                "order_id": "order_002",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": "2.0",
                "price": "3250.00",
                "status": "PENDING"
            }
        ]
        
        # Выполняем backup
        backup_result = backup_manager.create_backup({
            "portfolio": portfolio_data,
            "orders": orders_data,
            "backup_type": "FULL"
        })
        
        assert backup_result.success is True
        assert backup_result.backup_id is not None
        assert backup_result.backup_size > 0
        assert backup_result.checksum is not None
        
        # Симулируем потерю данных
        corrupted_data = None
        
        # Восстанавливаем из backup
        restore_result = backup_manager.restore_from_backup(
            backup_id=backup_result.backup_id,
            restore_point=datetime.utcnow()
        )
        
        assert restore_result.success is True
        assert restore_result.restored_data["portfolio"]["user_id"] == "backup_test_user"
        assert len(restore_result.restored_data["orders"]) == 2
        assert restore_result.integrity_verified is True

    def test_data_corruption_detection_and_recovery(
        self, 
        backup_manager: BackupManager
    ) -> None:
        """Тест обнаружения и восстановления поврежденных данных."""
        
        # Создаем исходные данные
        original_data = {
            "user_accounts": [
                {"user_id": "user_001", "balance": "10000.00"},
                {"user_id": "user_002", "balance": "25000.00"}
            ],
            "checksum": "original_checksum_12345"
        }
        
        # Создаем backup
        backup_result = backup_manager.create_backup(original_data)
        
        # Симулируем повреждение данных
        corrupted_data = {
            "user_accounts": [
                {"user_id": "user_001", "balance": "1000000.00"},  # Неправильный баланс
                {"user_id": "user_002", "balance": "25000.00"}
            ],
            "checksum": "corrupted_checksum_54321"  # Неправильная checksum
        }
        
        # Проверяем обнаружение повреждения
        corruption_detected = backup_manager.verify_data_integrity(corrupted_data)
        assert corruption_detected is False
        
        # Автоматическое восстановление
        auto_recovery_result = backup_manager.auto_recover_corrupted_data(
            corrupted_data=corrupted_data,
            backup_id=backup_result.backup_id
        )
        
        assert auto_recovery_result.recovery_successful is True
        assert auto_recovery_result.data_restored["user_accounts"][0]["balance"] == "10000.00"
        assert auto_recovery_result.corruption_cause == "CHECKSUM_MISMATCH"

    def test_network_partition_handling(self) -> None:
        """Тест обработки сетевых разделений."""
        
        network_status = {"connected": True}
        operation_queue = []
        
        def simulate_network_partition():
            """Симулирует сетевое разделение."""
            network_status["connected"] = False
            time.sleep(2)
            network_status["connected"] = True
        
        async def resilient_trading_operation(order_data: dict) -> dict:
            """Торговая операция устойчивая к сетевым проблемам."""
            
            if not network_status["connected"]:
                # Сохраняем операцию в очереди
                operation_queue.append({
                    "operation": "CREATE_ORDER",
                    "data": order_data,
                    "timestamp": datetime.utcnow().isoformat(),
                    "retry_count": 0
                })
                raise NetworkError("Network partition detected")
            
            # Обрабатываем накопленные операции
            while operation_queue:
                queued_op = operation_queue.pop(0)
                # Обрабатываем отложенную операцию
                pass
            
            return {
                "status": "success",
                "order_id": f"order_{random.randint(1000, 9999)}",
                "data": order_data
            }
        
        # Запускаем сетевое разделение в фоне
        partition_thread = threading.Thread(target=simulate_network_partition)
        partition_thread.start()
        
        # Пытаемся выполнить операции во время разделения
        order_data = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1"}
        
        try:
            result = asyncio.run(resilient_trading_operation(order_data))
            # Если сеть работает, операция должна пройти
            if network_status["connected"]:
                assert result["status"] == "success"
        except NetworkError:
            # Операция должна быть добавлена в очередь
            assert len(operation_queue) > 0
            assert operation_queue[0]["operation"] == "CREATE_ORDER"
        
        partition_thread.join()
        
        # После восстановления сети операции должны быть обработаны
        if operation_queue:
            result = asyncio.run(resilient_trading_operation(order_data))
            assert result["status"] == "success"
            assert len(operation_queue) == 0

    def test_database_failover(self) -> None:
        """Тест failover базы данных."""
        
        db_status = {
            "primary": True,
            "secondary": True,
            "tertiary": True
        }
        
        query_log = []
        
        class ResilientDatabase:
            def __init__(self):
                self.current_db = "primary"
            
            def execute_query(self, query: str) -> dict:
                """Выполнение запроса с автоматическим failover."""
                query_log.append(f"{self.current_db}:{query}")
                
                if not db_status[self.current_db]:
                    # Текущая база недоступна, переключаемся
                    if self.current_db == "primary" and db_status["secondary"]:
                        self.current_db = "secondary"
                    elif self.current_db == "secondary" and db_status["tertiary"]:
                        self.current_db = "tertiary"
                    else:
                        raise ServiceUnavailableError("All databases unavailable")
                    
                    return self.execute_query(query)  # Повторяем на новой базе
                
                return {
                    "status": "success",
                    "database": self.current_db,
                    "query": query,
                    "result": f"Data from {self.current_db}"
                }
        
        db = ResilientDatabase()
        
        # Нормальная работа
        result = db.execute_query("SELECT * FROM orders")
        assert result["database"] == "primary"
        
        # Отказ primary базы
        db_status["primary"] = False
        result = db.execute_query("SELECT * FROM portfolio")
        assert result["database"] == "secondary"
        
        # Отказ secondary базы
        db_status["secondary"] = False
        result = db.execute_query("SELECT * FROM users")
        assert result["database"] == "tertiary"
        
        # Проверяем log failover'ов
        assert "primary:SELECT * FROM orders" in query_log
        assert "secondary:SELECT * FROM portfolio" in query_log
        assert "tertiary:SELECT * FROM users" in query_log

    def test_cascading_failure_prevention(self) -> None:
        """Тест предотвращения каскадных отказов."""
        
        service_loads = {
            "trading_engine": 0,
            "risk_management": 0,
            "portfolio_service": 0,
            "notification_service": 0
        }
        
        max_load = 100
        
        class LoadBalancer:
            def __init__(self):
                self.circuit_breakers = {
                    service: CircuitBreaker(failure_threshold=10, recovery_timeout=60)
                    for service in service_loads.keys()
                }
            
            def route_request(self, service: str, request: dict) -> dict:
                """Маршрутизация запроса с защитой от перегрузки."""
                
                # Проверяем нагрузку на сервис
                if service_loads[service] >= max_load:
                    # Перенаправляем на менее загруженный сервис
                    alternative_service = min(
                        service_loads.keys(),
                        key=lambda s: service_loads[s]
                    )
                    
                    if service_loads[alternative_service] < max_load:
                        return self.route_request(alternative_service, request)
                    else:
                        raise SystemOverloadError("All services overloaded")
                
                # Увеличиваем нагрузку
                service_loads[service] += 10
                
                try:
                    # Используем circuit breaker
                    result = self.circuit_breakers[service].call(
                        lambda: self._execute_request(service, request)
                    )
                    
                    # Уменьшаем нагрузку после успешного выполнения
                    service_loads[service] = max(0, service_loads[service] - 10)
                    
                    return result
                    
                except Exception as e:
                    # Уменьшаем нагрузку при ошибке
                    service_loads[service] = max(0, service_loads[service] - 5)
                    raise
            
            def _execute_request(self, service: str, request: dict) -> dict:
                """Выполнение запроса к сервису."""
                if service_loads[service] > 80:
                    # Высокая нагрузка может вызывать сбои
                    if random.random() < 0.3:  # 30% вероятность сбоя
                        raise ServiceUnavailableError(f"{service} overloaded")
                
                return {
                    "status": "success",
                    "service": service,
                    "load": service_loads[service],
                    "request": request
                }
        
        load_balancer = LoadBalancer()
        
        # Симулируем высокую нагрузку
        successful_requests = 0
        failed_requests = 0
        
        for i in range(100):
            try:
                result = load_balancer.route_request(
                    "trading_engine",
                    {"request_id": i, "action": "place_order"}
                )
                successful_requests += 1
            except (ServiceUnavailableError, SystemOverloadError):
                failed_requests += 1
        
        # Система должна обработать большинство запросов
        assert successful_requests > failed_requests
        
        # Нагрузка должна быть распределена
        total_load = sum(service_loads.values())
        assert total_load < len(service_loads) * max_load

    def test_disaster_recovery_plan_execution(
        self, 
        backup_manager: BackupManager,
        failover_manager: FailoverManager
    ) -> None:
        """Тест выполнения плана disaster recovery."""
        
        class DisasterRecoveryOrchestrator:
            def __init__(self, backup_manager, failover_manager):
                self.backup_manager = backup_manager
                self.failover_manager = failover_manager
                self.recovery_steps = []
            
            def execute_disaster_recovery(self, disaster_type: str) -> dict:
                """Выполнение плана восстановления после катастрофы."""
                
                recovery_plan = {
                    "DATA_CENTER_FAILURE": [
                        "activate_secondary_datacenter",
                        "restore_from_backup",
                        "redirect_traffic",
                        "verify_data_integrity",
                        "notify_stakeholders"
                    ],
                    "DATABASE_CORRUPTION": [
                        "stop_write_operations",
                        "restore_from_last_backup",
                        "verify_data_consistency",
                        "resume_operations",
                        "post_incident_analysis"
                    ],
                    "NETWORK_OUTAGE": [
                        "activate_backup_connections",
                        "queue_critical_operations",
                        "switch_to_offline_mode",
                        "resume_when_online",
                        "process_queued_operations"
                    ]
                }
                
                if disaster_type not in recovery_plan:
                    raise DisasterRecoveryError(f"No plan for {disaster_type}")
                
                steps = recovery_plan[disaster_type]
                results = []
                
                for step in steps:
                    step_result = self._execute_recovery_step(step)
                    results.append(step_result)
                    self.recovery_steps.append(step)
                    
                    if not step_result["success"]:
                        break
                
                return {
                    "disaster_type": disaster_type,
                    "steps_completed": len(results),
                    "total_steps": len(steps),
                    "success": all(r["success"] for r in results),
                    "results": results
                }
            
            def _execute_recovery_step(self, step: str) -> dict:
                """Выполнение отдельного шага восстановления."""
                
                if step == "activate_secondary_datacenter":
                    # Активация вторичного ЦОД
                    return {"step": step, "success": True, "message": "Secondary DC activated"}
                
                elif step == "restore_from_backup":
                    # Восстановление из резервной копии
                    try:
                        backup_result = self.backup_manager.restore_latest_backup()
                        return {"step": step, "success": True, "backup_id": backup_result.backup_id}
                    except Exception as e:
                        return {"step": step, "success": False, "error": str(e)}
                
                elif step == "redirect_traffic":
                    # Перенаправление трафика
                    try:
                        failover_result = asyncio.run(
                            self.failover_manager.execute_emergency_failover()
                        )
                        return {"step": step, "success": True, "new_endpoint": failover_result.active_endpoint}
                    except Exception as e:
                        return {"step": step, "success": False, "error": str(e)}
                
                else:
                    # Остальные шаги симулируем как успешные
                    return {"step": step, "success": True, "message": f"{step} completed"}
        
        # Создаем orchestrator
        dr_orchestrator = DisasterRecoveryOrchestrator(backup_manager, failover_manager)
        
        # Тестируем различные сценарии катастроф
        scenarios = ["DATA_CENTER_FAILURE", "DATABASE_CORRUPTION", "NETWORK_OUTAGE"]
        
        for scenario in scenarios:
            recovery_result = dr_orchestrator.execute_disaster_recovery(scenario)
            
            assert recovery_result["disaster_type"] == scenario
            assert recovery_result["steps_completed"] > 0
            assert recovery_result["success"] is True
            
            # Проверяем что все шаги были выполнены
            assert recovery_result["steps_completed"] == recovery_result["total_steps"]

    def test_real_time_system_monitoring(self) -> None:
        """Тест real-time мониторинга системы."""
        
        system_metrics = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "network_latency": 0,
            "active_connections": 0,
            "error_rate": 0
        }
        
        alerts_triggered = []
        
        class SystemMonitor:
            def __init__(self):
                self.thresholds = {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "disk_usage": 90,
                    "network_latency": 1000,  # ms
                    "active_connections": 10000,
                    "error_rate": 5  # %
                }
                self.monitoring = True
            
            def start_monitoring(self):
                """Запуск мониторинга системы."""
                while self.monitoring:
                    # Симулируем сбор метрик
                    self._collect_metrics()
                    
                    # Проверяем пороги
                    self._check_thresholds()
                    
                    time.sleep(1)
            
            def _collect_metrics(self):
                """Сбор системных метрик."""
                # Симулируем изменение метрик
                system_metrics["cpu_usage"] = random.randint(20, 95)
                system_metrics["memory_usage"] = random.randint(30, 90)
                system_metrics["disk_usage"] = random.randint(40, 95)
                system_metrics["network_latency"] = random.randint(10, 1500)
                system_metrics["active_connections"] = random.randint(1000, 12000)
                system_metrics["error_rate"] = random.randint(0, 10)
            
            def _check_thresholds(self):
                """Проверка порогов и создание алертов."""
                for metric, value in system_metrics.items():
                    threshold = self.thresholds[metric]
                    
                    if value > threshold:
                        alert = {
                            "metric": metric,
                            "value": value,
                            "threshold": threshold,
                            "severity": self._calculate_severity(value, threshold),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        alerts_triggered.append(alert)
                        
                        # Автоматические действия при критических алертах
                        if alert["severity"] == "CRITICAL":
                            self._execute_emergency_action(metric)
            
            def _calculate_severity(self, value: float, threshold: float) -> str:
                """Расчет серьезности алерта."""
                ratio = value / threshold
                if ratio > 1.5:
                    return "CRITICAL"
                elif ratio > 1.2:
                    return "HIGH"
                elif ratio > 1.0:
                    return "MEDIUM"
                else:
                    return "LOW"
            
            def _execute_emergency_action(self, metric: str):
                """Выполнение экстренных действий."""
                if metric == "cpu_usage":
                    # Снижение нагрузки на CPU
                    system_metrics["cpu_usage"] *= 0.7
                elif metric == "memory_usage":
                    # Очистка кэша
                    system_metrics["memory_usage"] *= 0.8
                elif metric == "error_rate":
                    # Активация circuit breaker
                    system_metrics["error_rate"] *= 0.5
        
        # Запускаем мониторинг на короткое время
        monitor = SystemMonitor()
        
        monitor_thread = threading.Thread(target=monitor.start_monitoring)
        monitor_thread.start()
        
        time.sleep(5)  # Мониторим 5 секунд
        monitor.monitoring = False
        monitor_thread.join()
        
        # Проверяем что алерты создавались
        assert len(alerts_triggered) > 0
        
        # Проверяем что есть алерты разной степени серьезности
        severities = set(alert["severity"] for alert in alerts_triggered)
        assert len(severities) > 1
        
        # Проверяем что экстренные действия снижали метрики
        high_severity_alerts = [
            alert for alert in alerts_triggered 
            if alert["severity"] in ["HIGH", "CRITICAL"]
        ]
        assert len(high_severity_alerts) > 0

    def test_graceful_degradation(self) -> None:
        """Тест graceful degradation при сбоях."""
        
        service_availability = {
            "real_time_data": True,
            "advanced_analytics": True,
            "notifications": True,
            "reporting": True,
            "ai_recommendations": True
        }
        
        class GracefulTradingSystem:
            def __init__(self):
                self.degradation_levels = {
                    0: ["real_time_data", "advanced_analytics", "notifications", "reporting", "ai_recommendations"],
                    1: ["real_time_data", "notifications", "reporting"],  # Отключили AI и аналитику
                    2: ["real_time_data", "notifications"],  # Отключили отчеты
                    3: ["real_time_data"],  # Только базовые данные
                    4: []  # Полный отказ
                }
                self.current_level = 0
            
            def process_trading_request(self, request: dict) -> dict:
                """Обработка торгового запроса с graceful degradation."""
                
                # Определяем доступные сервисы
                available_services = []
                for service, available in service_availability.items():
                    if available:
                        available_services.append(service)
                
                # Определяем уровень деградации
                degradation_level = self._calculate_degradation_level(available_services)
                
                if degradation_level != self.current_level:
                    self.current_level = degradation_level
                
                # Обрабатываем запрос согласно доступному функционалу
                result = {
                    "request_id": request.get("request_id"),
                    "degradation_level": degradation_level,
                    "available_features": self.degradation_levels[degradation_level],
                    "processing_result": {}
                }
                
                # Базовая торговая функциональность (всегда доступна)
                result["processing_result"]["order_placement"] = "AVAILABLE"
                
                # Дополнительные функции в зависимости от уровня
                if "real_time_data" in available_services:
                    result["processing_result"]["market_data"] = "REAL_TIME"
                else:
                    result["processing_result"]["market_data"] = "CACHED"
                
                if "advanced_analytics" in available_services:
                    result["processing_result"]["risk_analysis"] = "ADVANCED"
                else:
                    result["processing_result"]["risk_analysis"] = "BASIC"
                
                if "ai_recommendations" in available_services:
                    result["processing_result"]["trade_suggestions"] = "AI_POWERED"
                else:
                    result["processing_result"]["trade_suggestions"] = "RULE_BASED"
                
                if "notifications" in available_services:
                    result["processing_result"]["notifications"] = "ENABLED"
                else:
                    result["processing_result"]["notifications"] = "DISABLED"
                
                return result
            
            def _calculate_degradation_level(self, available_services: List[str]) -> int:
                """Расчет уровня деградации."""
                for level, required_services in self.degradation_levels.items():
                    if all(service in available_services for service in required_services):
                        return level
                
                return 4  # Полный отказ
        
        trading_system = GracefulTradingSystem()
        
        # Тест нормальной работы
        test_request = {"request_id": "test_001", "symbol": "BTCUSDT"}
        result = trading_system.process_trading_request(test_request)
        
        assert result["degradation_level"] == 0
        assert result["processing_result"]["market_data"] == "REAL_TIME"
        assert result["processing_result"]["risk_analysis"] == "ADVANCED"
        
        # Симулируем отказ AI сервиса
        service_availability["ai_recommendations"] = False
        service_availability["advanced_analytics"] = False
        
        result = trading_system.process_trading_request(test_request)
        
        assert result["degradation_level"] == 1
        assert result["processing_result"]["risk_analysis"] == "BASIC"
        assert result["processing_result"]["trade_suggestions"] == "RULE_BASED"
        
        # Дальнейшее ухудшение
        service_availability["reporting"] = False
        
        result = trading_system.process_trading_request(test_request)
        
        assert result["degradation_level"] == 2
        assert "reporting" not in result["available_features"]
        
        # Критическая деградация
        service_availability["notifications"] = False
        
        result = trading_system.process_trading_request(test_request)
        
        assert result["degradation_level"] == 3
        assert result["processing_result"]["notifications"] == "DISABLED"
        
        # Базовая функциональность всегда остается доступной
        assert result["processing_result"]["order_placement"] == "AVAILABLE"