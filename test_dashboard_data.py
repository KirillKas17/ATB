#!/usr/bin/env python3
"""
Скрипт для проверки передачи данных в дашборд ATB.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

import aiohttp
import websockets

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardDataTester:
    """Тестер для проверки передачи данных в дашборд."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.test_results = []
        
    async def test_http_endpoints(self) -> Dict[str, bool]:
        """Тестирование HTTP endpoints."""
        logger.info("Тестирование HTTP endpoints...")
        
        endpoints = [
            ("/", "Главная страница"),
            ("/api/status", "Статус системы"),
            ("/api/trading", "Торговые данные"),
            ("/api/positions", "Позиции"),
            ("/api/analytics", "Аналитика"),
            ("/api/health", "Проверка здоровья")
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint, description in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    logger.info(f"Тестирование {description} ({url})")
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json() if endpoint != "/" else None
                            results[endpoint] = True
                            logger.info(f"✅ {description} - OK")
                            
                            if data:
                                logger.info(f"   Данные: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
                        else:
                            results[endpoint] = False
                            logger.error(f"❌ {description} - HTTP {response.status}")
                            
                except Exception as e:
                    results[endpoint] = False
                    logger.error(f"❌ {description} - Ошибка: {e}")
        
        return results
    
    async def test_websocket_connection(self) -> bool:
        """Тестирование WebSocket соединения."""
        logger.info("Тестирование WebSocket соединения...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info(f"✅ WebSocket подключен к {self.ws_url}")
                
                # Ожидание первого сообщения
                logger.info("Ожидание данных от WebSocket...")
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                
                data = json.loads(message)
                logger.info(f"✅ Получены данные WebSocket: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
                
                # Отправка тестового сообщения
                test_message = {
                    "type": "test",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Тестовое сообщение от клиента"
                }
                await websocket.send(json.dumps(test_message))
                logger.info("✅ Тестовое сообщение отправлено")
                
                return True
                
        except asyncio.TimeoutError:
            logger.error("❌ WebSocket: Таймаут ожидания данных")
            return False
        except Exception as e:
            logger.error(f"❌ WebSocket: Ошибка подключения - {e}")
            return False
    
    async def test_data_consistency(self) -> Dict[str, bool]:
        """Проверка консистентности данных."""
        logger.info("Проверка консистентности данных...")
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Получение данных из разных endpoints
            endpoints = ["/api/status", "/api/trading", "/api/positions", "/api/analytics"]
            data_sets = {}
            
            for endpoint in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data_sets[endpoint] = await response.json()
                        else:
                            logger.error(f"❌ Не удалось получить данные из {endpoint}")
                            return {endpoint: False}
                except Exception as e:
                    logger.error(f"❌ Ошибка получения данных из {endpoint}: {e}")
                    return {endpoint: False}
            
            # Проверка структуры данных
            required_fields = {
                "/api/status": ["status", "uptime", "cpu_usage", "memory_usage"],
                "/api/trading": ["total_pnl", "daily_pnl", "active_positions", "win_rate"],
                "/api/positions": [],  # Список позиций
                "/api/analytics": ["rsi", "macd", "bollinger_position", "ai_signals"]
            }
            
            for endpoint, required in required_fields.items():
                data = data_sets.get(endpoint, {})
                
                if endpoint == "/api/positions":
                    # Проверка что это список
                    if isinstance(data, list):
                        results[f"{endpoint}_structure"] = True
                        logger.info(f"✅ {endpoint} - структура данных корректна (список)")
                    else:
                        results[f"{endpoint}_structure"] = False
                        logger.error(f"❌ {endpoint} - неверная структура данных (ожидается список)")
                else:
                    # Проверка обязательных полей
                    missing_fields = [field for field in required if field not in data]
                    if not missing_fields:
                        results[f"{endpoint}_structure"] = True
                        logger.info(f"✅ {endpoint} - все обязательные поля присутствуют")
                    else:
                        results[f"{endpoint}_structure"] = False
                        logger.error(f"❌ {endpoint} - отсутствуют поля: {missing_fields}")
            
            # Проверка типов данных
            type_checks = {
                "/api/status": {
                    "status": str,
                    "cpu_usage": (int, float),
                    "memory_usage": (int, float)
                },
                "/api/trading": {
                    "total_pnl": (int, float),
                    "daily_pnl": (int, float),
                    "active_positions": int,
                    "win_rate": (int, float)
                }
            }
            
            for endpoint, type_requirements in type_checks.items():
                data = data_sets.get(endpoint, {})
                type_errors = []
                
                for field, expected_type in type_requirements.items():
                    if field in data:
                        if not isinstance(data[field], expected_type):
                            type_errors.append(f"{field} (ожидается {expected_type}, получено {type(data[field])})")
                
                if not type_errors:
                    results[f"{endpoint}_types"] = True
                    logger.info(f"✅ {endpoint} - типы данных корректны")
                else:
                    results[f"{endpoint}_types"] = False
                    logger.error(f"❌ {endpoint} - ошибки типов: {type_errors}")
        
        return results
    
    async def test_performance(self) -> Dict[str, float]:
        """Тестирование производительности API."""
        logger.info("Тестирование производительности API...")
        
        performance_results = {}
        
        async with aiohttp.ClientSession() as session:
            endpoints = ["/api/status", "/api/trading", "/api/positions", "/api/analytics"]
            
            for endpoint in endpoints:
                url = f"{self.base_url}{endpoint}"
                times = []
                
                # 5 запросов для каждого endpoint
                for i in range(5):
                    start_time = time.time()
                    try:
                        async with session.get(url) as response:
                            await response.json()
                            end_time = time.time()
                            times.append(end_time - start_time)
                    except Exception as e:
                        logger.error(f"Ошибка при тестировании {endpoint}: {e}")
                
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    min_time = min(times)
                    
                    performance_results[endpoint] = {
                        "avg": avg_time,
                        "max": max_time,
                        "min": min_time
                    }
                    
                    logger.info(f"✅ {endpoint}: среднее время {avg_time:.3f}s (мин: {min_time:.3f}s, макс: {max_time:.3f}s)")
        
        return performance_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Запуск всех тестов."""
        logger.info("=" * 60)
        logger.info("           ТЕСТИРОВАНИЕ ДАШБОРДА ATB")
        logger.info("=" * 60)
        
        results = {}
        
        # Тест HTTP endpoints
        results["http_endpoints"] = await self.test_http_endpoints()
        
        # Тест WebSocket
        results["websocket"] = await self.test_websocket_connection()
        
        # Тест консистентности данных
        results["data_consistency"] = await self.test_data_consistency()
        
        # Тест производительности
        results["performance"] = await self.test_performance()
        
        # Подсчет результатов
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    total_tests += 1
                    if test_result:
                        passed_tests += 1
            elif isinstance(category_results, bool):
                total_tests += 1
                if category_results:
                    passed_tests += 1
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Вывод итогового отчета
        logger.info("=" * 60)
        logger.info("           ИТОГОВЫЙ ОТЧЕТ")
        logger.info("=" * 60)
        logger.info(f"Всего тестов: {total_tests}")
        logger.info(f"Пройдено: {passed_tests}")
        logger.info(f"Провалено: {total_tests - passed_tests}")
        logger.info(f"Процент успеха: {results['summary']['success_rate']:.1f}%")
        
        if results['summary']['success_rate'] >= 80:
            logger.info("✅ Дашборд работает корректно!")
        elif results['summary']['success_rate'] >= 60:
            logger.warning("⚠️ Дашборд работает с предупреждениями")
        else:
            logger.error("❌ Дашборд работает некорректно!")
        
        return results

async def main():
    """Главная функция."""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    tester = DashboardDataTester(base_url)
    results = await tester.run_all_tests()
    
    # Сохранение результатов в файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dashboard_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Результаты сохранены в файл: {filename}")

if __name__ == "__main__":
    asyncio.run(main()) 