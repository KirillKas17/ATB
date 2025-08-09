#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенный тест производительности торговой системы ATB.
Анализ критически важных компонентов без внешних зависимостей.
"""

import time
import gc
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

# Добавляем пути для импорта
sys.path.append("/workspace")
sys.path.append("/workspace/domain")
sys.path.append("/workspace/infrastructure")
sys.path.append("/workspace/application")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ Pandas недоступен - некоторые тесты будут пропущены")
    PANDAS_AVAILABLE = False

try:
    from shared.numpy_utils import np

    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ NumPy недоступен - некоторые тесты будут пропущены")
    NUMPY_AVAILABLE = False


@dataclass
class PerformanceResult:
    """Результат теста производительности."""

    component_name: str
    operation: str
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    throughput_ops_sec: float
    performance_grade: str
    recommendations: List[str]
    test_passed: bool


class SimpleProfiler:
    """Простой профайлер производительности."""

    def __init__(self):
        self.critical_thresholds = {
            # Критические операции для торговой системы
            "market_data_processing": {"max_ms": 100, "avg_ms": 50},
            "pattern_recognition": {"max_ms": 200, "avg_ms": 100},
            "feature_generation": {"max_ms": 500, "avg_ms": 200},
            "signal_analysis": {"max_ms": 150, "avg_ms": 75},
            "cache_operations": {"max_ms": 10, "avg_ms": 5},
            "order_validation": {"max_ms": 50, "avg_ms": 25},
            "risk_calculation": {"max_ms": 100, "avg_ms": 50},
        }

    def profile_function(self, func, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
        """Профилирование функции."""
        times = []

        # Прогрев
        for _ in range(3):
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
                print(f"Ошибка в итерации {i}: {e}")
                times.append(float("inf"))

        # Фильтруем бесконечные значения
        valid_times = [t for t in times if t != float("inf")]

        if not valid_times:
            return {
                "avg_time_ms": float("inf"),
                "max_time_ms": float("inf"),
                "min_time_ms": float("inf"),
                "throughput_ops_sec": 0,
            }

        avg_time = statistics.mean(valid_times)
        return {
            "avg_time_ms": avg_time,
            "max_time_ms": max(valid_times),
            "min_time_ms": min(valid_times),
            "throughput_ops_sec": 1000 / avg_time if avg_time > 0 else 0,
        }

    def grade_performance(self, operation: str, metrics: Dict[str, float]) -> str:
        """Оценка производительности."""
        if operation not in self.critical_thresholds:
            # Общие критерии
            avg_time = metrics["avg_time_ms"]
            if avg_time < 50:
                return "A"
            elif avg_time < 200:
                return "B"
            elif avg_time < 1000:
                return "C"
            elif avg_time < 5000:
                return "D"
            else:
                return "F"

        thresholds = self.critical_thresholds[operation]
        avg_time = metrics["avg_time_ms"]
        max_time = metrics["max_time_ms"]

        if avg_time <= thresholds["avg_ms"] * 0.5 and max_time <= thresholds["max_ms"] * 0.5:
            return "A"
        elif avg_time <= thresholds["avg_ms"] and max_time <= thresholds["max_ms"]:
            return "B"
        elif avg_time <= thresholds["avg_ms"] * 2 and max_time <= thresholds["max_ms"] * 2:
            return "C"
        elif avg_time <= thresholds["avg_ms"] * 5 and max_time <= thresholds["max_ms"] * 5:
            return "D"
        else:
            return "F"

    def get_recommendations(self, operation: str, metrics: Dict[str, float], grade: str) -> List[str]:
        """Генерация рекомендаций."""
        recommendations = []

        if grade in ["D", "F"]:
            recommendations.append("🔴 КРИТИЧЕСКАЯ ПРОБЛЕМА: Требуется немедленная оптимизация")

        avg_time = metrics["avg_time_ms"]

        if avg_time > 1000:
            recommendations.append("Рассмотрите асинхронную обработку")

        if operation == "feature_generation" and avg_time > 200:
            recommendations.extend(
                [
                    "Используйте векторизованные операции",
                    "Кэшируйте промежуточные вычисления",
                    "Оптимизируйте алгоритмы расчета индикаторов",
                ]
            )

        if operation == "pattern_recognition" and avg_time > 100:
            recommendations.extend(
                [
                    "Предварительно вычислите индикаторы",
                    "Используйте скользящие окна",
                    "Оптимизируйте алгоритмы поиска паттернов",
                ]
            )

        if operation == "market_data_processing" and avg_time > 50:
            recommendations.extend(
                [
                    "Оптимизируйте операции с pandas DataFrame",
                    "Используйте более эффективные структуры данных",
                    "Рассмотрите batch-обработку",
                ]
            )

        return recommendations


class PerformanceTester:
    """Тестирование производительности критических компонентов."""

    def __init__(self):
        self.profiler = SimpleProfiler()
        self.results: List[PerformanceResult] = []

    def test_basic_data_operations(self) -> PerformanceResult:
        """Тест базовых операций с данными."""

        def basic_operations():
            # Имитируем работу с рыночными данными
            data = []
            for i in range(1000):
                data.append({"price": 50000 + i * 0.1, "volume": 100 + i, "timestamp": time.time() + i})

            # Базовая обработка
            prices = [d["price"] for d in data]
            volumes = [d["volume"] for d in data]

            # Простые вычисления
            avg_price = sum(prices) / len(prices)
            max_volume = max(volumes)

            return avg_price, max_volume

        metrics = self.profiler.profile_function(basic_operations, iterations=50)
        grade = self.profiler.grade_performance("market_data_processing", metrics)
        recommendations = self.profiler.get_recommendations("market_data_processing", metrics, grade)

        return PerformanceResult(
            component_name="BasicDataOperations",
            operation="market_data_processing",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B", "C"],
        )

    def test_technical_indicators(self) -> PerformanceResult:
        """Тест расчета технических индикаторов."""

        def calculate_indicators():
            # Генерируем тестовые данные
            prices = [50000 + i * 0.1 + (i % 10 - 5) for i in range(100)]

            # Простая скользящая средняя
            sma_period = 20
            sma = []
            for i in range(len(prices)):
                if i >= sma_period - 1:
                    avg = sum(prices[i - sma_period + 1 : i + 1]) / sma_period
                    sma.append(avg)
                else:
                    sma.append(prices[i])

            # RSI упрощенный
            rsi_values = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                if change > 0:
                    rsi_values.append(70 + change * 10)  # Упрощенный RSI
                else:
                    rsi_values.append(30 + change * 10)

            return sma, rsi_values

        metrics = self.profiler.profile_function(calculate_indicators, iterations=100)
        grade = self.profiler.grade_performance("feature_generation", metrics)
        recommendations = self.profiler.get_recommendations("feature_generation", metrics, grade)

        return PerformanceResult(
            component_name="TechnicalIndicators",
            operation="feature_generation",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B", "C"],
        )

    def test_pattern_detection(self) -> PerformanceResult:
        """Тест обнаружения паттернов."""

        def detect_patterns():
            # Простое обнаружение паттернов
            prices = [50000 + i * 0.1 + (i % 20 - 10) * 5 for i in range(50)]
            volumes = [1000 + i * 10 + (i % 15 - 7) * 50 for i in range(50)]

            patterns = []

            # Поиск аномалий объема
            for i in range(1, len(volumes)):
                if volumes[i] > volumes[i - 1] * 2:
                    patterns.append({"type": "volume_spike", "index": i})

            # Поиск резких движений цены
            for i in range(1, len(prices)):
                price_change = abs(prices[i] - prices[i - 1]) / prices[i - 1]
                if price_change > 0.05:  # 5% изменение
                    patterns.append({"type": "price_spike", "index": i})

            return patterns

        metrics = self.profiler.profile_function(detect_patterns, iterations=100)
        grade = self.profiler.grade_performance("pattern_recognition", metrics)
        recommendations = self.profiler.get_recommendations("pattern_recognition", metrics, grade)

        return PerformanceResult(
            component_name="PatternDetection",
            operation="pattern_recognition",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B", "C"],
        )

    def test_cache_simulation(self) -> PerformanceResult:
        """Тест кэш операций."""

        cache = {}

        def cache_operations():
            # Запись в кэш
            key = f"market_data_{time.time()}"
            data = {"price": 50000, "volume": 1000, "timestamp": time.time()}
            cache[key] = data

            # Чтение из кэша
            result = cache.get(key)

            # Очистка старых записей
            if len(cache) > 100:
                old_keys = list(cache.keys())[:10]
                for old_key in old_keys:
                    cache.pop(old_key, None)

            return result

        metrics = self.profiler.profile_function(cache_operations, iterations=200)
        grade = self.profiler.grade_performance("cache_operations", metrics)
        recommendations = self.profiler.get_recommendations("cache_operations", metrics, grade)

        return PerformanceResult(
            component_name="CacheSimulation",
            operation="cache_operations",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B"],
        )

    def test_order_validation(self) -> PerformanceResult:
        """Тест валидации ордеров."""

        def validate_order():
            # Имитация валидации ордера
            order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01, "price": 50000, "type": "LIMIT"}

            # Базовые проверки
            validations = []

            # Проверка символа
            if order["symbol"] in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
                validations.append("symbol_valid")

            # Проверка количества
            if 0.001 <= order["quantity"] <= 100:
                validations.append("quantity_valid")

            # Проверка цены
            if 1000 <= order["price"] <= 100000:
                validations.append("price_valid")

            # Проверка стороны
            if order["side"] in ["BUY", "SELL"]:
                validations.append("side_valid")

            return len(validations) == 4

        metrics = self.profiler.profile_function(validate_order, iterations=500)
        grade = self.profiler.grade_performance("order_validation", metrics)
        recommendations = self.profiler.get_recommendations("order_validation", metrics, grade)

        return PerformanceResult(
            component_name="OrderValidation",
            operation="order_validation",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B", "C"],
        )

    def test_risk_calculation(self) -> PerformanceResult:
        """Тест расчета рисков."""

        def calculate_risk():
            # Имитация расчета рисков
            portfolio = {
                "BTCUSDT": {"quantity": 0.5, "price": 50000},
                "ETHUSDT": {"quantity": 2.0, "price": 3000},
                "ADAUSDT": {"quantity": 1000, "price": 0.5},
            }

            total_value = 0
            risks = {}

            for symbol, position in portfolio.items():
                value = position["quantity"] * position["price"]
                total_value += value

                # Простой расчет риска
                volatility = 0.02  # 2% дневная волатильность
                var_95 = value * volatility * 1.65  # VaR 95%

                risks[symbol] = {"value": value, "var_95": var_95, "risk_percent": var_95 / value * 100}

            # Общий риск портфеля
            total_var = sum(r["var_95"] for r in risks.values())
            portfolio_risk = total_var / total_value * 100

            return portfolio_risk, risks

        metrics = self.profiler.profile_function(calculate_risk, iterations=200)
        grade = self.profiler.grade_performance("risk_calculation", metrics)
        recommendations = self.profiler.get_recommendations("risk_calculation", metrics, grade)

        return PerformanceResult(
            component_name="RiskCalculation",
            operation="risk_calculation",
            avg_time_ms=metrics["avg_time_ms"],
            max_time_ms=metrics["max_time_ms"],
            min_time_ms=metrics["min_time_ms"],
            throughput_ops_sec=metrics["throughput_ops_sec"],
            performance_grade=grade,
            recommendations=recommendations,
            test_passed=grade in ["A", "B", "C"],
        )

    def run_all_tests(self) -> List[PerformanceResult]:
        """Запуск всех тестов."""
        print("🚀 Запуск упрощенного теста производительности...")
        print("=" * 70)

        tests = [
            ("Базовые операции с данными", self.test_basic_data_operations),
            ("Технические индикаторы", self.test_technical_indicators),
            ("Обнаружение паттернов", self.test_pattern_detection),
            ("Кэш операции", self.test_cache_simulation),
            ("Валидация ордеров", self.test_order_validation),
            ("Расчет рисков", self.test_risk_calculation),
        ]

        results = []

        for test_name, test_func in tests:
            print(f"⏳ Тестирование: {test_name}")
            try:
                result = test_func()
                results.append(result)

                # Цветной вывод результата
                grade_colors = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}
                color = grade_colors.get(result.performance_grade, "❓")

                print(f"   {color} Оценка: {result.performance_grade}")
                print(f"   ⏱️  Среднее время: {result.avg_time_ms:.3f}ms")
                print(f"   📈 Пропускная способность: {result.throughput_ops_sec:.0f} ops/sec")

                if result.performance_grade in ["D", "F"]:
                    print(f"   ⚠️  ТРЕБУЕТ ОПТИМИЗАЦИИ!")

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

            print("-" * 40)

        return results


def generate_performance_report(results: List[PerformanceResult]) -> str:
    """Генерация отчета по производительности."""
    report = []
    report.append("# 📊 ОТЧЕТ ПО ПРОИЗВОДИТЕЛЬНОСТИ ТОРГОВОЙ СИСТЕМЫ ATB")
    report.append("=" * 70)
    report.append("")

    # Общая статистика
    critical_issues = [r for r in results if r.performance_grade in ["D", "F"]]
    good_performance = [r for r in results if r.performance_grade in ["A", "B"]]

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
            report.append(f"- **Среднее время**: {result.avg_time_ms:.3f}ms")
            report.append(f"- **Максимальное время**: {result.max_time_ms:.3f}ms")
            report.append(f"- **Пропускная способность**: {result.throughput_ops_sec:.0f} ops/sec")
            report.append("")

            if result.recommendations:
                report.append("**Рекомендации по оптимизации:**")
                for rec in result.recommendations:
                    report.append(f"- {rec}")
                report.append("")

    # Детальные результаты
    report.append("## 📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    for result in sorted(results, key=lambda x: x.avg_time_ms, reverse=True):
        grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}
        emoji = grade_emoji.get(result.performance_grade, "❓")

        report.append(f"### {emoji} {result.component_name} - {result.operation}")
        report.append(f"**Оценка производительности**: {result.performance_grade}")
        report.append("")

        report.append("**Метрики времени:**")
        report.append(f"- Среднее время: {result.avg_time_ms:.3f}ms")
        report.append(f"- Минимальное время: {result.min_time_ms:.3f}ms")
        report.append(f"- Максимальное время: {result.max_time_ms:.3f}ms")
        report.append(f"- Пропускная способность: {result.throughput_ops_sec:.0f} операций/сек")
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
            report.append(f"- **{result.component_name}**: {result.avg_time_ms:.3f}ms")

    report.append("")
    report.append("### 🚀 Общие рекомендации по оптимизации:")
    report.append("1. **Алгоритмическая оптимизация**: Пересмотрите сложные алгоритмы")
    report.append("2. **Кэширование**: Кэшируйте результаты тяжелых вычислений")
    report.append("3. **Структуры данных**: Используйте оптимальные структуры данных")
    report.append("4. **Профилирование**: Регулярно профилируйте код")
    report.append("5. **Мониторинг**: Настройте мониторинг производительности")

    # Специфичные рекомендации для торговой системы
    report.append("")
    report.append("### 💡 Специфичные рекомендации для торговой системы:")
    report.append("1. **Обработка рыночных данных**: Должна быть < 50ms для real-time торговли")
    report.append("2. **Валидация ордеров**: Должна быть < 25ms для быстрого исполнения")
    report.append("3. **Расчет рисков**: Должен быть < 50ms для оперативного контроля")
    report.append("4. **Распознавание паттернов**: Должно быть < 100ms для своевременных сигналов")
    report.append("5. **Кэш операции**: Должны быть < 5ms для минимальных задержек")

    return "\n".join(report)


def main():
    """Основная функция для запуска теста."""
    print("🔬 Инициализация упрощенного теста производительности...")

    tester = PerformanceTester()
    results = tester.run_all_tests()

    print("\n" + "=" * 70)
    print("📝 Генерация отчета...")

    report = generate_performance_report(results)

    # Сохранение отчета
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"performance_report_simplified_{timestamp}.md"

    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📄 Отчет сохранен: {report_filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения отчета: {e}")
        print("📄 Вывод отчета в консоль:")
        print(report)

    print("\n" + "=" * 70)
    print("✅ Упрощенный тест производительности завершен!")

    # Вывод краткого резюме
    critical_count = len([r for r in results if r.performance_grade in ["D", "F"]])
    failed_tests = len([r for r in results if not r.test_passed])

    if critical_count > 0:
        print(f"⚠️  Обнаружено {critical_count} критических проблем производительности!")
        print("🔧 Рекомендуется оптимизация указанных компонентов.")

    if failed_tests > 0:
        print(f"❌ {failed_tests} тестов не прошли проверку производительности!")
    else:
        print("🎉 Все тесты прошли проверку производительности!")

    return results


if __name__ == "__main__":
    main()
