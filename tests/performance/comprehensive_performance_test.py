#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексный тест производительности торговой системы ATB.
Фокус на критически важных компонентах где скорость влияет на торговую эффективность.
"""

import asyncio
import cProfile
import io
import pstats
import time
import gc
import threading
import psutil
import pandas as pd
from shared.numpy_utils import np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock
import pytest
import statistics
import warnings

# Критически важные компоненты для тестирования
from infrastructure.core.feature_engineering import FeatureEngineer, FeatureConfig
from infrastructure.ml_services.ml_models import MLModelManager
from domain.intelligence.market_pattern_recognizer import MarketPatternRecognizer
from application.services.cache_service import CacheService

warnings.filterwarnings("ignore")


@dataclass
class PerformanceResult:
    """Результат теста производительности."""
    component_name: str
    operation: str
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    p95_time_ms: float
    throughput_ops_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    is_critical: bool
    performance_grade: str  # A, B, C, D, F
    recommendations: List[str]


class PerformanceProfiler:
    """Профайлер производительности для критических операций."""
    
    def __init__(self):
        self.results: List[PerformanceResult] = []
        self.critical_thresholds = {
            # Критические операции (влияют на торговые решения)
            'market_data_processing': {'max_ms': 100, 'avg_ms': 50},  # Обработка рыночных данных
            'pattern_recognition': {'max_ms': 200, 'avg_ms': 100},    # Распознавание паттернов
            'feature_generation': {'max_ms': 500, 'avg_ms': 200},     # Генерация признаков
            'ml_prediction': {'max_ms': 300, 'avg_ms': 150},          # ML предсказания
            'signal_analysis': {'max_ms': 150, 'avg_ms': 75},         # Анализ сигналов
            'cache_operations': {'max_ms': 10, 'avg_ms': 5},          # Кэш операции
            
            # Менее критические операции
            'model_training': {'max_ms': 30000, 'avg_ms': 15000},     # Обучение моделей
            'data_validation': {'max_ms': 1000, 'avg_ms': 500},       # Валидация данных
            'report_generation': {'max_ms': 2000, 'avg_ms': 1000},    # Генерация отчетов
        }
    
    def profile_function(self, func: Callable, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
        """Профилирование функции с множественными итерациями."""
        times = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_before = psutil.cpu_percent()
        
        # Прогрев
        for _ in range(5):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Основные измерения
        for i in range(iterations):
            gc.collect()
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # в миллисекундах
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                times.append(float('inf'))
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_after = psutil.cpu_percent()
        
        # Фильтруем бесконечные значения
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {
                'avg_time_ms': float('inf'),
                'max_time_ms': float('inf'),
                'min_time_ms': float('inf'),
                'p95_time_ms': float('inf'),
                'throughput_ops_sec': 0,
                'memory_usage_mb': memory_after - memory_before,
                'cpu_usage_percent': cpu_after - cpu_before
            }
        
        return {
            'avg_time_ms': statistics.mean(valid_times),
            'max_time_ms': max(valid_times),
            'min_time_ms': min(valid_times),
            'p95_time_ms': np.percentile(valid_times, 95),
            'throughput_ops_sec': 1000 / statistics.mean(valid_times) if statistics.mean(valid_times) > 0 else 0,
            'memory_usage_mb': memory_after - memory_before,
            'cpu_usage_percent': cpu_after - cpu_before
        }
    
    def grade_performance(self, operation: str, metrics: Dict[str, float]) -> str:
        """Оценка производительности операции."""
        if operation not in self.critical_thresholds:
            # Для неизвестных операций используем общие критерии
            if metrics['avg_time_ms'] < 50:
                return 'A'
            elif metrics['avg_time_ms'] < 200:
                return 'B'
            elif metrics['avg_time_ms'] < 1000:
                return 'C'
            elif metrics['avg_time_ms'] < 5000:
                return 'D'
            else:
                return 'F'
        
        thresholds = self.critical_thresholds[operation]
        avg_time = metrics['avg_time_ms']
        max_time = metrics['max_time_ms']
        
        # Проверка критических требований
        if avg_time <= thresholds['avg_ms'] * 0.5 and max_time <= thresholds['max_ms'] * 0.5:
            return 'A'  # Отличная производительность
        elif avg_time <= thresholds['avg_ms'] and max_time <= thresholds['max_ms']:
            return 'B'  # Хорошая производительность
        elif avg_time <= thresholds['avg_ms'] * 2 and max_time <= thresholds['max_ms'] * 2:
            return 'C'  # Удовлетворительная производительность
        elif avg_time <= thresholds['avg_ms'] * 5 and max_time <= thresholds['max_ms'] * 5:
            return 'D'  # Неудовлетворительная производительность
        else:
            return 'F'  # Критически плохая производительность
    
    def get_recommendations(self, operation: str, metrics: Dict[str, float], grade: str) -> List[str]:
        """Генерация рекомендаций по оптимизации."""
        recommendations = []
        
        if grade in ['D', 'F']:
            recommendations.append("🔴 КРИТИЧЕСКАЯ ПРОБЛЕМА: Требуется немедленная оптимизация")
        
        if metrics['avg_time_ms'] > 1000:
            recommendations.append("Рассмотрите асинхронную обработку или многопоточность")
        
        if metrics['memory_usage_mb'] > 100:
            recommendations.append("Оптимизируйте использование памяти - возможны утечки")
        
        if metrics['cpu_usage_percent'] > 80:
            recommendations.append("Высокая нагрузка на CPU - рассмотрите оптимизацию алгоритмов")
        
        if operation == 'feature_generation' and metrics['avg_time_ms'] > 200:
            recommendations.extend([
                "Используйте векторизованные операции NumPy/Pandas",
                "Кэшируйте промежуточные вычисления",
                "Рассмотрите использование Numba или Cython"
            ])
        
        if operation == 'pattern_recognition' and metrics['avg_time_ms'] > 100:
            recommendations.extend([
                "Предварительно вычислите индикаторы",
                "Используйте скользящие окна вместо полного пересчета",
                "Оптимизируйте алгоритмы распознавания паттернов"
            ])
        
        if operation == 'ml_prediction' and metrics['avg_time_ms'] > 150:
            recommendations.extend([
                "Используйте более простые модели для real-time предсказаний",
                "Предварительно загружайте модели в память",
                "Рассмотрите квантизацию моделей",
                "Используйте батчевую обработку"
            ])
        
        if operation == 'cache_operations' and metrics['avg_time_ms'] > 5:
            recommendations.extend([
                "Оптимизируйте структуру кэша",
                "Используйте более быстрые сериализаторы",
                "Рассмотрите in-memory кэш (Redis/Memcached)"
            ])
        
        return recommendations


class CriticalComponentsTester:
    """Тестирование критически важных компонентов."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Генерация тестовых рыночных данных."""
        np.random.seed(42)
        size = 10000  # 10k свечей
        
        dates = pd.date_range('2023-01-01', periods=size, freq='1min')
        base_price = 50000.0
        
        # Генерируем реалистичные OHLCV данные
        data = []
        for i in range(size):
            price_change = np.random.normal(0, 0.002) * base_price
            base_price = max(base_price + price_change, 1000.0)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.001)))
            low = base_price * (1 - abs(np.random.normal(0, 0.001)))
            open_price = base_price + np.random.normal(0, 0.0005) * base_price
            close = base_price
            volume = np.random.exponential(1000)
            
            data.append({
                'open': max(open_price, 1000.0),
                'high': max(high, max(open_price, close, 1000.0)),
                'low': min(low, min(open_price, close)),
                'close': max(close, 1000.0),
                'volume': max(volume, 1)
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_feature_engineering_performance(self) -> PerformanceResult:
        """Тест производительности генерации признаков."""
        config = FeatureConfig(
            use_technical_indicators=True,
            use_statistical_features=True,
            ema_periods=[5, 10, 20, 50],
            rsi_periods=[14, 21],
            rolling_windows=[5, 10, 20]
        )
        engineer = FeatureEngineer(config=config)
        
        def generate_features():
            return engineer.generate_features(self.test_data)
        
        metrics = self.profiler.profile_function(generate_features, iterations=10)
        grade = self.profiler.grade_performance('feature_generation', metrics)
        recommendations = self.profiler.get_recommendations('feature_generation', metrics, grade)
        
        return PerformanceResult(
            component_name="FeatureEngineer",
            operation="feature_generation",
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            p95_time_ms=metrics['p95_time_ms'],
            throughput_ops_sec=metrics['throughput_ops_sec'],
            memory_usage_mb=metrics['memory_usage_mb'],
            cpu_usage_percent=metrics['cpu_usage_percent'],
            is_critical=True,
            performance_grade=grade,
            recommendations=recommendations
        )
    
    def test_pattern_recognition_performance(self) -> PerformanceResult:
        """Тест производительности распознавания паттернов."""
        recognizer = MarketPatternRecognizer()
        
        # Создаем mock order book
        order_book = {
            'bids': [[49950, 10], [49940, 20], [49930, 15]],
            'asks': [[50050, 8], [50060, 18], [50070, 12]],
            'timestamp': time.time()
        }
        
        def detect_patterns():
            return recognizer.detect_whale_absorption('BTCUSDT', self.test_data, order_book)
        
        metrics = self.profiler.profile_function(detect_patterns, iterations=20)
        grade = self.profiler.grade_performance('pattern_recognition', metrics)
        recommendations = self.profiler.get_recommendations('pattern_recognition', metrics, grade)
        
        return PerformanceResult(
            component_name="MarketPatternRecognizer",
            operation="pattern_recognition",
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            p95_time_ms=metrics['p95_time_ms'],
            throughput_ops_sec=metrics['throughput_ops_sec'],
            memory_usage_mb=metrics['memory_usage_mb'],
            cpu_usage_percent=metrics['cpu_usage_percent'],
            is_critical=True,
            performance_grade=grade,
            recommendations=recommendations
        )
    
    def test_ml_prediction_performance(self) -> PerformanceResult:
        """Тест производительности ML предсказаний."""
        # Создаем простую обученную модель
        ml_manager = MLModelManager()
        
        # Подготавливаем данные для обучения
        features = self.test_data[['open', 'high', 'low', 'close', 'volume']].copy()
        features['returns'] = features['close'].pct_change()
        features = features.dropna()
        
        # Создаем целевую переменную (направление движения)
        target = (features['returns'].shift(-1) > 0).astype(int)
        target = target.dropna()
        features = features[:-1]
        
        # Быстрое обучение для теста
        ml_manager.train_models(features, target)
        
        # Тестовые данные для предсказания
        test_features = features.tail(100)
        
        def make_prediction():
            return ml_manager.predict(test_features, model_name="random_forest")
        
        metrics = self.profiler.profile_function(make_prediction, iterations=50)
        grade = self.profiler.grade_performance('ml_prediction', metrics)
        recommendations = self.profiler.get_recommendations('ml_prediction', metrics, grade)
        
        return PerformanceResult(
            component_name="MLModelManager",
            operation="ml_prediction",
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            p95_time_ms=metrics['p95_time_ms'],
            throughput_ops_sec=metrics['throughput_ops_sec'],
            memory_usage_mb=metrics['memory_usage_mb'],
            cpu_usage_percent=metrics['cpu_usage_percent'],
            is_critical=True,
            performance_grade=grade,
            recommendations=recommendations
        )
    
    def test_cache_operations_performance(self) -> PerformanceResult:
        """Тест производительности кэш операций."""
        cache_service = CacheService()
        
        # Тестовые данные
        test_key = "test_market_data_BTCUSDT"
        test_data = self.test_data.to_dict()
        
        def cache_operations():
            # Запись в кэш
            cache_service.set(test_key, test_data, ttl=300)
            # Чтение из кэша
            result = cache_service.get(test_key)
            return result
        
        metrics = self.profiler.profile_function(cache_operations, iterations=100)
        grade = self.profiler.grade_performance('cache_operations', metrics)
        recommendations = self.profiler.get_recommendations('cache_operations', metrics, grade)
        
        return PerformanceResult(
            component_name="CacheService",
            operation="cache_operations",
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            p95_time_ms=metrics['p95_time_ms'],
            throughput_ops_sec=metrics['throughput_ops_sec'],
            memory_usage_mb=metrics['memory_usage_mb'],
            cpu_usage_percent=metrics['cpu_usage_percent'],
            is_critical=True,
            performance_grade=grade,
            recommendations=recommendations
        )
    
    def test_data_processing_performance(self) -> PerformanceResult:
        """Тест производительности обработки рыночных данных."""
        
        def process_market_data():
            # Имитируем типичную обработку рыночных данных
            data = self.test_data.copy()
            
            # Расчет индикаторов
            data['sma_20'] = data['close'].rolling(20).mean()
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['rsi'] = 100 - (100 / (1 + data['close'].pct_change().rolling(14).apply(
                lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) != 0 else 0
            )))
            
            # Анализ объемов
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # Волатильность
            data['volatility'] = data['close'].pct_change().rolling(20).std()
            
            return data.dropna()
        
        metrics = self.profiler.profile_function(process_market_data, iterations=20)
        grade = self.profiler.grade_performance('market_data_processing', metrics)
        recommendations = self.profiler.get_recommendations('market_data_processing', metrics, grade)
        
        return PerformanceResult(
            component_name="MarketDataProcessor",
            operation="market_data_processing",
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            p95_time_ms=metrics['p95_time_ms'],
            throughput_ops_sec=metrics['throughput_ops_sec'],
            memory_usage_mb=metrics['memory_usage_mb'],
            cpu_usage_percent=metrics['cpu_usage_percent'],
            is_critical=True,
            performance_grade=grade,
            recommendations=recommendations
        )
    
    def run_all_tests(self) -> List[PerformanceResult]:
        """Запуск всех тестов производительности."""
        print("🚀 Запуск комплексного теста производительности...")
        print(f"📊 Размер тестовых данных: {len(self.test_data)} записей")
        print("=" * 80)
        
        tests = [
            ("Обработка рыночных данных", self.test_data_processing_performance),
            ("Генерация признаков", self.test_feature_engineering_performance),
            ("Распознавание паттернов", self.test_pattern_recognition_performance),
            ("ML предсказания", self.test_ml_prediction_performance),
            ("Кэш операции", self.test_cache_operations_performance),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"⏳ Тестирование: {test_name}")
            try:
                result = test_func()
                results.append(result)
                
                # Цветной вывод результата
                grade_colors = {
                    'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'
                }
                color = grade_colors.get(result.performance_grade, '❓')
                
                print(f"   {color} Оценка: {result.performance_grade}")
                print(f"   ⏱️  Среднее время: {result.avg_time_ms:.2f}ms")
                print(f"   📈 Пропускная способность: {result.throughput_ops_sec:.1f} ops/sec")
                
                if result.performance_grade in ['D', 'F']:
                    print(f"   ⚠️  ТРЕБУЕТ ОПТИМИЗАЦИИ!")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
            
            print("-" * 40)
        
        return results


def generate_performance_report(results: List[PerformanceResult]) -> str:
    """Генерация детального отчета по производительности."""
    report = []
    report.append("# 📊 ОТЧЕТ ПО ПРОИЗВОДИТЕЛЬНОСТИ ТОРГОВОЙ СИСТЕМЫ ATB")
    report.append("=" * 70)
    report.append("")
    
    # Общая статистика
    critical_issues = [r for r in results if r.performance_grade in ['D', 'F']]
    good_performance = [r for r in results if r.performance_grade in ['A', 'B']]
    
    report.append("## 📈 ОБЩАЯ СТАТИСТИКА")
    report.append(f"- Всего протестировано компонентов: {len(results)}")
    report.append(f"- Компоненты с хорошей производительностью (A-B): {len(good_performance)}")
    report.append(f"- Компоненты требующие оптимизации (D-F): {len(critical_issues)}")
    report.append("")
    
    if critical_issues:
        report.append("## 🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ ПРОИЗВОДИТЕЛЬНОСТИ")
        for result in critical_issues:
            report.append(f"### {result.component_name} - {result.operation}")
            report.append(f"- **Оценка**: {result.performance_grade}")
            report.append(f"- **Среднее время**: {result.avg_time_ms:.2f}ms")
            report.append(f"- **Максимальное время**: {result.max_time_ms:.2f}ms")
            report.append(f"- **Пропускная способность**: {result.throughput_ops_sec:.1f} ops/sec")
            report.append("")
            
            if result.recommendations:
                report.append("**Рекомендации по оптимизации:**")
                for rec in result.recommendations:
                    report.append(f"- {rec}")
                report.append("")
    
    # Детальные результаты
    report.append("## 📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    for result in sorted(results, key=lambda x: x.avg_time_ms, reverse=True):
        grade_emoji = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
        emoji = grade_emoji.get(result.performance_grade, '❓')
        
        report.append(f"### {emoji} {result.component_name} - {result.operation}")
        report.append(f"**Оценка производительности**: {result.performance_grade}")
        report.append("")
        
        report.append("**Метрики времени:**")
        report.append(f"- Среднее время: {result.avg_time_ms:.2f}ms")
        report.append(f"- Минимальное время: {result.min_time_ms:.2f}ms")
        report.append(f"- Максимальное время: {result.max_time_ms:.2f}ms")
        report.append(f"- 95-й перцентиль: {result.p95_time_ms:.2f}ms")
        report.append("")
        
        report.append("**Ресурсы:**")
        report.append(f"- Пропускная способность: {result.throughput_ops_sec:.1f} операций/сек")
        report.append(f"- Использование памяти: {result.memory_usage_mb:.2f}MB")
        report.append(f"- Нагрузка на CPU: {result.cpu_usage_percent:.1f}%")
        report.append("")
        
        if result.recommendations:
            report.append("**Рекомендации:**")
            for rec in result.recommendations:
                report.append(f"- {rec}")
            report.append("")
        
        report.append("-" * 50)
        report.append("")
    
    # Заключение
    report.append("## 🎯 ЗАКЛЮЧЕНИЕ И РЕКОМЕНДАЦИИ")
    
    if critical_issues:
        report.append("### ⚠️ Требуется немедленная оптимизация:")
        for result in critical_issues:
            report.append(f"- **{result.component_name}**: {result.avg_time_ms:.2f}ms (целевое время зависит от операции)")
    
    report.append("")
    report.append("### 🚀 Общие рекомендации по оптимизации:")
    report.append("1. **Векторизация операций**: Используйте NumPy/Pandas операции вместо циклов Python")
    report.append("2. **Кэширование**: Кэшируйте результаты тяжелых вычислений")
    report.append("3. **Асинхронность**: Используйте asyncio для I/O операций")
    report.append("4. **Профилирование**: Регулярно профилируйте код для выявления узких мест")
    report.append("5. **Мониторинг**: Настройте мониторинг производительности в production")
    
    return "\n".join(report)


# Основная функция для запуска тестов
def main():
    """Основная функция для запуска комплексного теста производительности."""
    print("🔬 Инициализация теста производительности...")
    
    tester = CriticalComponentsTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 80)
    print("📝 Генерация отчета...")
    
    report = generate_performance_report(results)
    
    # Сохранение отчета
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"performance_report_{timestamp}.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Отчет сохранен: {report_filename}")
    print("\n" + "=" * 80)
    print("✅ Тест производительности завершен!")
    
    # Вывод краткого резюме
    critical_count = len([r for r in results if r.performance_grade in ['D', 'F']])
    if critical_count > 0:
        print(f"⚠️  Обнаружено {critical_count} критических проблем производительности!")
        print("🔧 Рекомендуется немедленная оптимизация указанных компонентов.")
    else:
        print("🎉 Все компоненты показывают приемлемую производительность!")
    
    return results


if __name__ == "__main__":
    # Для запуска как pytest тест
    pytest.main([__file__, "-v"])
    
    # Для прямого запуска
    # main()


# Pytest тесты
class TestCriticalPerformance:
    """Pytest тесты производительности критических компонентов."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.tester = CriticalComponentsTester()
    
    def test_feature_engineering_performance_acceptable(self):
        """Тест: генерация признаков должна быть достаточно быстрой."""
        result = self.tester.test_feature_engineering_performance()
        assert result.performance_grade in ['A', 'B', 'C'], f"Производительность генерации признаков неприемлема: {result.performance_grade}"
        assert result.avg_time_ms < 1000, f"Слишком медленная генерация признаков: {result.avg_time_ms}ms"
    
    def test_pattern_recognition_performance_acceptable(self):
        """Тест: распознавание паттернов должно быть достаточно быстрым."""
        result = self.tester.test_pattern_recognition_performance()
        assert result.performance_grade in ['A', 'B', 'C'], f"Производительность распознавания паттернов неприемлема: {result.performance_grade}"
        assert result.avg_time_ms < 500, f"Слишком медленное распознавание паттернов: {result.avg_time_ms}ms"
    
    def test_cache_operations_performance_fast(self):
        """Тест: операции кэша должны быть очень быстрыми."""
        result = self.tester.test_cache_operations_performance()
        assert result.performance_grade in ['A', 'B'], f"Производительность кэша неприемлема: {result.performance_grade}"
        assert result.avg_time_ms < 20, f"Слишком медленные операции кэша: {result.avg_time_ms}ms"
    
    def test_market_data_processing_performance_acceptable(self):
        """Тест: обработка рыночных данных должна быть достаточно быстрой."""
        result = self.tester.test_data_processing_performance()
        assert result.performance_grade in ['A', 'B', 'C'], f"Производительность обработки данных неприемлема: {result.performance_grade}"
        assert result.avg_time_ms < 200, f"Слишком медленная обработка данных: {result.avg_time_ms}ms"
    
    def test_comprehensive_performance_report(self):
        """Тест: генерация комплексного отчета о производительности."""
        results = self.tester.run_all_tests()
        
        assert len(results) > 0, "Тесты производительности не вернули результатов"
        
        # Проверяем, что нет критических проблем
        critical_issues = [r for r in results if r.performance_grade == 'F']
        assert len(critical_issues) == 0, f"Обнаружены критические проблемы производительности: {[r.component_name for r in critical_issues]}"
        
        # Генерируем отчет
        report = generate_performance_report(results)
        assert len(report) > 1000, "Отчет слишком короткий"
        assert "ОТЧЕТ ПО ПРОИЗВОДИТЕЛЬНОСТИ" in report, "Отчет не содержит заголовок"