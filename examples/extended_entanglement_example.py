# -*- coding: utf-8 -*-
"""Пример использования расширенной системы обнаружения запутанности с новыми биржами."""

import asyncio
import json
import time
from typing import Dict, List, Any

from loguru import logger

from application.analysis.entanglement_monitor import EntanglementMonitor
from application.entanglement.stream_manager import StreamManager
from domain.intelligence.entanglement_detector import EntanglementResult


class ExtendedEntanglementExample:
    """Пример расширенного мониторинга запутанности."""

    def __init__(self):
        self.monitor = EntanglementMonitor(
            log_file_path="logs/extended_entanglement_events.json",
            detection_interval=0.5,
            max_lag_ms=5.0,
            correlation_threshold=0.90,
            enable_new_exchanges=True
        )
        
        self.stream_manager = StreamManager(
            max_lag_ms=5.0,
            correlation_threshold=0.90,
            detection_interval=0.5
        )
        
        self.detection_history: List[Dict[str, Any]] = []
        self.is_running = False

    async def setup_exchanges(self) -> None:
        """Настройка подключений к биржам."""
        logger.info("Setting up exchange connections...")
        
        # Символы для мониторинга
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # API ключи (в реальном использовании загружаются из конфига)
        api_keys = {
            "bingx": {
                "api_key": None,  # Замените на реальный ключ
                "api_secret": None  # Замените на реальный секрет
            },
            "bitget": {
                "api_key": None,
                "api_secret": None
            },
            "bybit": {
                "api_key": None,
                "api_secret": None
            }
        }
        
        try:
            # Инициализируем новые биржи через StreamManager
            await self.stream_manager.initialize_exchanges(symbols, api_keys)
            
            # Добавляем callback для обработки результатов
            self.stream_manager.add_entanglement_callback(self._handle_entanglement_detection)
            
            logger.info("Exchange connections setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup exchanges: {e}")

    async def _handle_entanglement_detection(self, result: EntanglementResult):
        """Обработка обнаружения запутанности."""
        try:
            # Сохраняем в историю
            detection_record = {
                "timestamp": result.timestamp.value,
                "exchange_pair": result.exchange_pair,
                "symbol": result.symbol,
                "is_entangled": result.is_entangled,
                "correlation_score": result.correlation_score,
                "lag_ms": result.lag_ms,
                "confidence": result.confidence,
                "metadata": result.metadata
            }
            
            self.detection_history.append(detection_record)
            
            # Ограничиваем размер истории
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            # Логируем важные обнаружения
            if result.is_entangled and result.confidence > 0.8:
                logger.warning(
                    f"🔗 HIGH CONFIDENCE ENTANGLEMENT DETECTED!\n"
                    f"   Exchanges: {result.exchange_pair[0]} ↔ {result.exchange_pair[1]}\n"
                    f"   Symbol: {result.symbol}\n"
                    f"   Lag: {result.lag_ms:.2f}ms\n"
                    f"   Correlation: {result.correlation_score:.3f}\n"
                    f"   Confidence: {result.confidence:.3f}\n"
                    f"   Metadata: {result.metadata}"
                )
            
            # Анализируем паттерны
            await self._analyze_entanglement_patterns(result)
            
        except Exception as e:
            logger.error(f"Error handling entanglement detection: {e}")

    async def _analyze_entanglement_patterns(self, result: EntanglementResult):
        """Анализ паттернов запутанности."""
        try:
            # Анализируем частоту запутанности между биржами
            exchange_pair = tuple(sorted(result.exchange_pair))
            
            # Можно добавить более сложную логику анализа
            if result.is_entangled:
                logger.info(
                    f"Pattern Analysis: {exchange_pair[0]} ↔ {exchange_pair[1]} "
                    f"shows consistent entanglement for {result.symbol}"
                )
                
        except Exception as e:
            logger.error(f"Error analyzing entanglement patterns: {e}")

    async def start_monitoring(self):
        """Запуск мониторинга."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
            
        self.is_running = True
        logger.info("Starting extended entanglement monitoring...")
        
        try:
            # Настраиваем биржи
            await self.setup_exchanges()
            
            # Запускаем StreamManager
            stream_task = asyncio.create_task(self.stream_manager.start_monitoring())
            
            # Запускаем мониторинг статистики
            stats_task = asyncio.create_task(self._monitor_statistics())
            
            # Ждем завершения задач
            await asyncio.gather(stream_task, stats_task, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            self.is_running = False

    async def stop_monitoring(self):
        """Остановка мониторинга."""
        self.is_running = False
        await self.stream_manager.stop_monitoring()
        logger.info("Extended entanglement monitoring stopped")

    async def _monitor_statistics(self):
        """Мониторинг статистики системы."""
        while self.is_running:
            try:
                # Получаем статус системы
                status = self.stream_manager.get_status()
                aggregator_stats = status.get("aggregator_stats", {})
                source_status = status.get("source_status", {})
                entanglement_stats = status.get("entanglement_stats", {})
                
                # Логируем статистику каждые 30 секунд
                logger.info(
                    f"📊 System Status:\n"
                    f"   Total Updates: {aggregator_stats.get('total_updates', 0)}\n"
                    f"   Active Sources: {aggregator_stats.get('active_sources', 0)}\n"
                    f"   Buffer Size: {aggregator_stats.get('buffer_size', 0)}\n"
                    f"   Entanglement Detections: {entanglement_stats.get('entangled_detections', 0)}\n"
                    f"   Detection Rate: {entanglement_stats.get('detection_rate', 0):.2f}/sec"
                )
                
                # Проверяем статус источников
                for source_name, source_info in source_status.items():
                    if not source_info.get("is_active", False):
                        logger.warning(f"Source {source_name} is inactive")
                    elif source_info.get("error_count", 0) > 5:
                        logger.warning(f"Source {source_name} has high error count: {source_info['error_count']}")
                
                await asyncio.sleep(30)  # Обновляем каждые 30 секунд
                
            except Exception as e:
                logger.error(f"Error monitoring statistics: {e}")
                await asyncio.sleep(5)

    def get_detection_summary(self) -> Dict[str, Any]:
        """Получение сводки обнаружений."""
        if not self.detection_history:
            return {"message": "No detections recorded"}
        
        # Анализируем историю
        total_detections = len(self.detection_history)
        entangled_detections = len([d for d in self.detection_history if d["is_entangled"]])
        
        # Группируем по парам бирж
        exchange_pairs = {}
        for detection in self.detection_history:
            pair_key = tuple(sorted(detection["exchange_pair"]))
            if pair_key not in exchange_pairs:
                exchange_pairs[pair_key] = {
                    "total": 0,
                    "entangled": 0,
                    "avg_correlation": 0.0,
                    "avg_lag": 0.0
                }
            
            exchange_pairs[pair_key]["total"] += 1
            if detection["is_entangled"]:
                exchange_pairs[pair_key]["entangled"] += 1
            
            exchange_pairs[pair_key]["avg_correlation"] += detection["correlation_score"]
            exchange_pairs[pair_key]["avg_lag"] += detection["lag_ms"]
        
        # Вычисляем средние значения
        for pair_data in exchange_pairs.values():
            if pair_data["total"] > 0:
                pair_data["avg_correlation"] /= pair_data["total"]
                pair_data["avg_lag"] /= pair_data["total"]
        
        return {
            "total_detections": total_detections,
            "entangled_detections": entangled_detections,
            "entanglement_rate": entangled_detections / total_detections if total_detections > 0 else 0,
            "exchange_pairs": exchange_pairs,
            "recent_detections": self.detection_history[-10:]  # Последние 10 обнаружений
        }

    def save_detection_report(self, filename: str = "entanglement_report.json"):
        """Сохранение отчета об обнаружениях."""
        try:
            summary = self.get_detection_summary()
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detection report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save detection report: {e}")


async def main():
    """Основная функция примера."""
    logger.info("🚀 Starting Extended Entanglement Detection Example")
    
    example = ExtendedEntanglementExample()
    
    try:
        # Запускаем мониторинг
        await example.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Останавливаем мониторинг
        await example.stop_monitoring()
        
        # Сохраняем отчет
        example.save_detection_report()
        
        # Выводим сводку
        summary = example.get_detection_summary()
        logger.info(f"📋 Final Summary: {summary}")
        
        logger.info("✅ Extended Entanglement Detection Example completed")


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/extended_entanglement_example.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    
    # Запуск примера
    asyncio.run(main()) 