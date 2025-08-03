#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для запуска расширенной системы обнаружения запутанности."""

import asyncio
import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from application.analysis.entanglement_monitor import EntanglementMonitor
from application.entanglement.stream_manager import StreamManager
from shared.config import reload_config as load_config


class ExtendedEntanglementRunner:
    """Запускатель расширенной системы обнаружения запутанности."""

    def __init__(self, config_path: str = "config/exchanges.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.monitor = None
        self.stream_manager = None
        self.is_running = False
        self.stats_file = Path("logs/entanglement_stats.json")
        
        # Настройка логирования
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации."""
        try:
            config = load_config()
            logger.info(f"Configuration loaded from {self.config_path}")
            return config.to_dict()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Получение конфигурации по умолчанию."""
        return {
            "monitoring": {
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                "detection_interval": 0.5,
                "max_lag_ms": 5.0,
                "correlation_threshold": 0.90
            },
            "new_exchanges": {
                "bingx": {"enabled": True, "api_key": None, "api_secret": None},
                "bitget": {"enabled": True, "api_key": None, "api_secret": None},
                "bybit": {"enabled": True, "api_key": None, "api_secret": None}
            }
        }

    def _setup_logging(self):
        """Настройка логирования."""
        log_config = self.config.get("logging", {})
        log_file = log_config.get("log_file", "logs/extended_entanglement.log")
        log_level = log_config.get("level", "INFO")
        
        # Создаем директорию для логов
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Настраиваем логирование
        logger.remove()  # Удаляем стандартный handler
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            log_file,
            level=log_level,
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "7 days"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )

    async def initialize_system(self):
        """Инициализация системы."""
        try:
            logger.info("Initializing extended entanglement detection system...")
            
            # Получаем настройки мониторинга
            monitoring_config = self.config.get("monitoring", {})
            symbols = monitoring_config.get("symbols", ["BTCUSDT", "ETHUSDT", "ADAUSDT"])
            detection_interval = monitoring_config.get("detection_interval", 0.5)
            max_lag_ms = monitoring_config.get("max_lag_ms", 5.0)
            correlation_threshold = monitoring_config.get("correlation_threshold", 0.90)
            
            # Создаем монитор
            self.monitor = EntanglementMonitor(
                log_file_path="logs/extended_entanglement_events.json",
                detection_interval=detection_interval,
                max_lag_ms=max_lag_ms,
                correlation_threshold=correlation_threshold,
                enable_new_exchanges=True
            )
            
            # Создаем StreamManager
            self.stream_manager = StreamManager(
                max_lag_ms=max_lag_ms,
                correlation_threshold=correlation_threshold,
                detection_interval=detection_interval
            )
            
            # Инициализируем новые биржи
            await self._initialize_new_exchanges(symbols)
            
            # Добавляем callback для обработки результатов
            self.stream_manager.add_entanglement_callback(self._handle_entanglement_result)
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    async def _initialize_new_exchanges(self, symbols: list):
        """Инициализация новых бирж."""
        try:
            if self.stream_manager is None:
                logger.error("StreamManager is not initialized")
                return
                
            new_exchanges_config = self.config.get("new_exchanges", {})
            api_keys = {}
            
            # Подготавливаем API ключи
            for exchange_name, exchange_config in new_exchanges_config.items():
                if exchange_config.get("enabled", False):
                    api_keys[exchange_name] = {
                        "api_key": exchange_config.get("api_key"),
                        "api_secret": exchange_config.get("api_secret")
                    }
            
            # Инициализируем биржи
            await self.stream_manager.initialize_exchanges(symbols, api_keys)
            
            logger.info(f"Initialized {len(api_keys)} new exchanges")
            
        except Exception as e:
            logger.error(f"Failed to initialize new exchanges: {e}")
            raise

    async def _handle_entanglement_result(self, result):
        """Обработка результата обнаружения запутанности."""
        try:
            # Логируем важные обнаружения
            if result.is_entangled and result.confidence > 0.8:
                logger.warning(
                    f"🔗 HIGH CONFIDENCE ENTANGLEMENT!\n"
                    f"   {result.exchange_pair[0]} ↔ {result.exchange_pair[1]} ({result.symbol})\n"
                    f"   Lag: {result.lag_ms:.2f}ms, Correlation: {result.correlation_score:.3f}, "
                    f"Confidence: {result.confidence:.3f}"
                )
            
            # Сохраняем статистику
            await self._save_stats()
            
        except Exception as e:
            logger.error(f"Error handling entanglement result: {e}")

    async def _save_stats(self):
        """Сохранение статистики."""
        try:
            if self.stream_manager:
                stats = self.stream_manager.get_entanglement_stats()
                
                # Добавляем timestamp
                stats["timestamp"] = asyncio.get_event_loop().time()
                
                # Сохраняем в файл
                with open(self.stats_file, "w") as f:
                    json.dump(stats, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    async def start_monitoring(self):
        """Запуск мониторинга."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
            
        self.is_running = True
        logger.info("Starting extended entanglement monitoring...")
        
        try:
            # Инициализируем систему
            await self.initialize_system()
            
            # Запускаем задачи мониторинга
            tasks = []
            
            # Задача StreamManager
            if self.stream_manager:
                tasks.append(asyncio.create_task(self.stream_manager.start_monitoring()))
            
            # Задача мониторинга статистики
            tasks.append(asyncio.create_task(self._monitor_statistics()))
            
            # Ждем завершения задач
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            self.is_running = False

    async def stop_monitoring(self):
        """Остановка мониторинга."""
        self.is_running = False
        
        if self.stream_manager:
            await self.stream_manager.stop_monitoring()
        
        logger.info("Extended entanglement monitoring stopped")

    async def _monitor_statistics(self):
        """Мониторинг статистики системы."""
        while self.is_running:
            try:
                if self.stream_manager:
                    status = self.stream_manager.get_status()
                    aggregator_stats = status.get("aggregator_stats", {})
                    entanglement_stats = status.get("entanglement_stats", {})
                    
                    # Логируем статистику каждые 60 секунд
                    logger.info(
                        f"📊 Stats: Updates={aggregator_stats.get('total_updates', 0)}, "
                        f"Active={aggregator_stats.get('active_sources', 0)}, "
                        f"Detections={entanglement_stats.get('entangled_detections', 0)}, "
                        f"Rate={entanglement_stats.get('detection_rate', 0):.2f}/sec"
                    )
                
                await asyncio.sleep(60)  # Обновляем каждую минуту
                
            except Exception as e:
                logger.error(f"Error monitoring statistics: {e}")
                await asyncio.sleep(5)

    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы."""
        status = {
            "is_running": self.is_running,
            "config_loaded": self.config is not None
        }
        
        if self.stream_manager:
            status.update(self.stream_manager.get_status())
        
        return status

    def print_status(self):
        """Вывод статуса системы."""
        status = self.get_system_status()
        
        print("\n" + "="*60)
        print("EXTENDED ENTANGLEMENT DETECTION SYSTEM STATUS")
        print("="*60)
        print(f"Running: {'Yes' if status['is_running'] else 'No'}")
        print(f"Config loaded: {'Yes' if status['config_loaded'] else 'No'}")
        
        if "aggregator_stats" in status:
            stats = status["aggregator_stats"]
            print(f"Active sources: {stats.get('active_sources', 0)}")
            print(f"Total updates: {stats.get('total_updates', 0)}")
            print(f"Buffer size: {stats.get('buffer_size', 0)}")
        
        if "entanglement_stats" in status:
            stats = status["entanglement_stats"]
            print(f"Total detections: {stats.get('total_detections', 0)}")
            print(f"Entangled detections: {stats.get('entangled_detections', 0)}")
            print(f"Detection rate: {stats.get('detection_rate', 0):.2f}/sec")
        
        print("="*60)


async def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Extended Entanglement Detection System")
    parser.add_argument("--config", default="config/exchanges.yaml", help="Path to config file")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    
    args = parser.parse_args()
    
    # Создаем запускатель
    runner = ExtendedEntanglementRunner(args.config)
    
    # Обработчик сигналов для graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(runner.stop_monitoring())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.status:
            # Показываем статус и выходим
            runner.print_status()
            return
        
        # Запускаем мониторинг
        await runner.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Останавливаем мониторинг
        await runner.stop_monitoring()
        
        # Показываем финальный статус
        runner.print_status()
        
        logger.info("Extended entanglement detection system stopped")


if __name__ == "__main__":
    # Создаем директории для логов
    Path("logs").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    
    # Запускаем систему
    asyncio.run(main()) 