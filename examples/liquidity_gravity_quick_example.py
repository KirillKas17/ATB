# -*- coding: utf-8 -*-
"""
Краткий пример использования системы гравитации ликвидности.

Этот пример демонстрирует базовое использование системы для анализа ликвидности
и корректировки агрессивности торговых агентов.
"""

import time
from datetime import datetime

from application.risk.liquidity_gravity_monitor import LiquidityGravityMonitor
from domain.market.liquidity_gravity import (LiquidityGravityModel,
                                             OrderBookSnapshot)
from domain.value_objects.timestamp import Timestamp


def create_sample_order_book(
    symbol: str = "BTC/USDT",
    spread_percentage: float = 0.1,
    volumes: tuple = (1.0, 1.0),
) -> OrderBookSnapshot:
    """Создание образца ордербука."""
    base_price = 50000.0
    spread = base_price * spread_percentage / 100

    bids = [
        (base_price - spread / 2, volumes[0]),
        (base_price - spread / 2 - 1, volumes[0] * 0.8),
    ]

    asks = [
        (base_price + spread / 2, volumes[1]),
        (base_price + spread / 2 + 1, volumes[1] * 0.8),
    ]

    return OrderBookSnapshot(
        bids=bids, asks=asks, timestamp=datetime.now(), symbol=symbol
    )


def main():
    """Основная функция примера."""
    print("=== Liquidity Gravity Field Model - Quick Example ===\n")

    # 1. Создание компонентов системы
    print("1. Инициализация системы...")
    gravity_model = LiquidityGravityModel()
    monitor = LiquidityGravityMonitor()

    print(f"   - Модель гравитации создана")
    print(f"   - Монитор гравитации создан\n")

    # 2. Анализ нормального ордербука
    print("2. Анализ нормального ордербука...")
    normal_order_book = create_sample_order_book(
        spread_percentage=0.1, volumes=(1.0, 1.0)
    )

    # Вычисление гравитации
    gravity = gravity_model.compute_liquidity_gravity(normal_order_book)
    print(f"   - Гравитация ликвидности: {gravity:.6f}")

    # Полный анализ
    gravity_result = gravity_model.analyze_liquidity_gravity(normal_order_book)
    print(f"   - Уровень риска: {gravity_result.risk_level}")
    print(
        f"   - Спред: {gravity_result.gravity_distribution['spread_percentage']:.2f}%"
    )
    print(f"   - Объем бидов: {gravity_result.gravity_distribution['bid_volume']:.2f}")
    print(f"   - Объем асков: {gravity_result.gravity_distribution['ask_volume']:.2f}\n")

    # 3. Анализ высокого риска
    print("3. Анализ ордербука с высоким риском...")
    high_risk_order_book = create_sample_order_book(
        spread_percentage=0.01, volumes=(10.0, 10.0)
    )

    # Вычисление гравитации
    gravity = gravity_model.compute_liquidity_gravity(high_risk_order_book)
    print(f"   - Гравитация ликвидности: {gravity:.6f}")

    # Полный анализ
    gravity_result = gravity_model.analyze_liquidity_gravity(high_risk_order_book)
    print(f"   - Уровень риска: {gravity_result.risk_level}")
    print(
        f"   - Спред: {gravity_result.gravity_distribution['spread_percentage']:.2f}%"
    )
    print(f"   - Объем бидов: {gravity_result.gravity_distribution['bid_volume']:.2f}")
    print(f"   - Объем асков: {gravity_result.gravity_distribution['ask_volume']:.2f}\n")

    # 4. Симуляция эволюции состояния агента
    print("4. Симуляция эволюции состояния агента...")

    # Создание серии ордербуков с разными уровнями риска
    order_books = [
        create_sample_order_book(
            spread_percentage=0.5, volumes=(0.1, 0.1)
        ),  # Низкий риск
        create_sample_order_book(
            spread_percentage=0.1, volumes=(1.0, 1.0)
        ),  # Средний риск
        create_sample_order_book(
            spread_percentage=0.01, volumes=(5.0, 5.0)
        ),  # Высокий риск
        create_sample_order_book(
            spread_percentage=0.001, volumes=(20.0, 20.0)
        ),  # Критический риск
    ]

    print("   Шаг | Спред % | Объем | Гравитация | Риск    ")
    print("   ----|---------|-------|------------|---------")

    for i, order_book in enumerate(order_books, 1):
        # Анализ
        gravity = gravity_model.compute_liquidity_gravity(order_book)
        gravity_result = gravity_model.analyze_liquidity_gravity(order_book)

        # Получение параметров для отображения
        spread_pct = gravity_result.gravity_distribution['spread_percentage']
        volume = gravity_result.gravity_distribution['bid_volume'] + gravity_result.gravity_distribution['ask_volume']

        print(
            f"   {i:2d}   | {spread_pct:6.2f}% | {volume:5.1f} | {gravity:10.6f} | {gravity_result.risk_level:7s}"
        )

    print()

    # 5. Статистика системы
    print("5. Статистика системы...")

    # Статистика модели
    model_stats = gravity_model.get_model_statistics()
    print(f"   - Гравитационная постоянная: {model_stats['gravitational_constant']}")
    print(f"   - Минимальный порог объема: {model_stats['min_volume_threshold']}")
    print(f"   - Максимальное расстояние цен: {model_stats['max_price_distance']}")

    # Статистика мониторинга
    monitor_stats = monitor.get_monitoring_statistics()
    print(f"   - Всего оценок: {monitor_stats['total_assessments']}")
    print(f"   - Высокий риск: {monitor_stats['high_risk_detections']}")
    print(f"   - Критический риск: {monitor_stats['critical_risk_detections']}")

    print("\n=== Пример завершен ===")


if __name__ == "__main__":
    main()
