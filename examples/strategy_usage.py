"""
Примеры использования стратегий Advanced Trading Bot.

Этот файл демонстрирует различные способы создания, настройки и использования
торговых стратегий в системе ATB.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, List, cast

from domain.entities.strategy import (
    Strategy, StrategyType, StrategyStatus, SignalType, SignalStrength
)
from domain.value_objects import Money
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


def create_trend_following_strategy() -> Strategy:
    """
    Создание трендследующей стратегии.
    
    Пример создания стратегии, которая следует за трендом.
    """
    config = {
        "name": "BTC Trend Follower",
        "description": "Трендследующая стратегия для BTC/USD",
        "strategy_type": StrategyType.TREND_FOLLOWING,
        "trading_pairs": ["BTC/USD"],
        "parameters": {
            "trend_period": 20,
            "trend_threshold": 0.02,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "position_size": 0.1,
            "confidence_threshold": 0.7,
            "max_signals": 2,
            "signal_cooldown": 1800
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    # Устанавливаем параметры
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_mean_reversion_strategy() -> Strategy:
    """
    Создание стратегии возврата к среднему.
    
    Пример создания стратегии, которая торгует на откатах от среднего значения.
    """
    config = {
        "name": "ETH Mean Reversion",
        "description": "Стратегия возврата к среднему для ETH/USD",
        "strategy_type": StrategyType.MEAN_REVERSION,
        "trading_pairs": ["ETH/USD"],
        "parameters": {
            "mean_reversion_threshold": 2.0,
            "lookback_period": 50,
            "stop_loss": 0.015,
            "take_profit": 0.03,
            "position_size": 0.08,
            "confidence_threshold": 0.65,
            "max_signals": 3,
            "signal_cooldown": 600
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_breakout_strategy() -> Strategy:
    """
    Создание стратегии пробоя.
    
    Пример создания стратегии, которая торгует на пробоях уровней.
    """
    config = {
        "name": "ADA Breakout Trader",
        "description": "Стратегия пробоя для ADA/USD",
        "strategy_type": StrategyType.BREAKOUT,
        "trading_pairs": ["ADA/USD"],
        "parameters": {
            "breakout_threshold": 1.5,
            "volume_multiplier": 2.0,
            "stop_loss": 0.025,
            "take_profit": 0.05,
            "position_size": 0.12,
            "confidence_threshold": 0.8,
            "max_signals": 2,
            "signal_cooldown": 900
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_scalping_strategy() -> Strategy:
    """
    Создание скальпинг стратегии.
    
    Пример создания стратегии для быстрых сделок с малым профитом.
    """
    config = {
        "name": "SOL Scalper",
        "description": "Скальпинг стратегия для SOL/USD",
        "strategy_type": StrategyType.SCALPING,
        "trading_pairs": ["SOL/USD"],
        "parameters": {
            "scalping_threshold": 0.1,
            "max_hold_time": 300,
            "stop_loss": 0.01,
            "take_profit": 0.02,
            "position_size": 0.05,
            "confidence_threshold": 0.55,
            "max_signals": 10,
            "signal_cooldown": 60
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_arbitrage_strategy() -> Strategy:
    """
    Создание арбитражной стратегии.
    
    Пример создания стратегии для торговли на разнице цен между биржами.
    """
    config = {
        "name": "Multi-Exchange Arbitrage",
        "description": "Арбитраж между Binance и Bybit",
        "strategy_type": StrategyType.ARBITRAGE,
        "trading_pairs": ["BTC/USD", "ETH/USD"],
        "parameters": {
            "arbitrage_threshold": 0.5,
            "max_slippage": 0.1,
            "stop_loss": 0.005,
            "take_profit": 0.01,
            "position_size": 0.15,
            "confidence_threshold": 0.9,
            "max_signals": 1,
            "signal_cooldown": 30
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_grid_strategy() -> Strategy:
    """
    Создание сеточной стратегии.
    
    Пример создания стратегии с размещением ордеров на разных уровнях.
    """
    config = {
        "name": "DOT Grid Bot",
        "description": "Сеточная стратегия для DOT/USD",
        "strategy_type": StrategyType.GRID,
        "trading_pairs": ["DOT/USD"],
        "parameters": {
            "grid_levels": 10,
            "grid_spacing": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "position_size": 0.1,
            "confidence_threshold": 0.6,
            "max_signals": 20,
            "signal_cooldown": 120
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_momentum_strategy() -> Strategy:
    """
    Создание стратегии импульса.
    
    Пример создания стратегии, которая торгует на импульсе движения цены.
    """
    config = {
        "name": "LINK Momentum Trader",
        "description": "Стратегия импульса для LINK/USD",
        "strategy_type": StrategyType.MOMENTUM,
        "trading_pairs": ["LINK/USD"],
        "parameters": {
            "momentum_period": 14,
            "momentum_threshold": 0.03,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "position_size": 0.09,
            "confidence_threshold": 0.75,
            "max_signals": 3,
            "signal_cooldown": 1200
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


def create_volatility_strategy() -> Strategy:
    """
    Создание волатильной стратегии.
    
    Пример создания стратегии, которая торгует на волатильности.
    """
    config = {
        "name": "AVAX Volatility Trader",
        "description": "Волатильная стратегия для AVAX/USD",
        "strategy_type": StrategyType.VOLATILITY,
        "trading_pairs": ["AVAX/USD"],
        "parameters": {
            "volatility_period": 20,
            "volatility_threshold": 0.05,
            "stop_loss": 0.025,
            "take_profit": 0.05,
            "position_size": 0.11,
            "confidence_threshold": 0.7,
            "max_signals": 4,
            "signal_cooldown": 900
        }
    }
    
    strategy = Strategy(
        name=str(config["name"]),
        description=str(config["description"]),
        strategy_type=StrategyType(config["strategy_type"]),
        trading_pairs=list(cast(List[str], config["trading_pairs"]))
    )
    
    strategy.parameters.update_parameters(dict(cast(Dict[str, Any], config["parameters"])))
    
    return strategy


async def demonstrate_strategy_usage():
    """
    Демонстрация использования стратегий.
    
    Показывает полный цикл работы со стратегиями:
    1. Создание стратегии
    2. Валидация конфигурации
    3. Генерация сигналов
    4. Проверка готовности к торговле
    5. Сохранение/загрузка состояния
    """
    print("=== ДЕМОНСТРАЦИЯ ИСПОЛЬЗОВАНИЯ СТРАТЕГИЙ ===\n")
    
    # Создаем различные стратегии
    strategies = [
        create_trend_following_strategy(),
        create_mean_reversion_strategy(),
        create_breakout_strategy(),
        create_scalping_strategy(),
        create_arbitrage_strategy(),
        create_grid_strategy(),
        create_momentum_strategy(),
        create_volatility_strategy()
    ]
    
    for strategy in strategies:
        print(f"📊 Стратегия: {strategy.name}")
        print(f"   Тип: {strategy.strategy_type.value}")
        print(f"   Пары: {', '.join(strategy.trading_pairs)}")
        
        # Валидация конфигурации
        config = {
            "name": strategy.name,
            "strategy_type": strategy.strategy_type.value,
            "trading_pairs": strategy.trading_pairs,
            "parameters": strategy.parameters.parameters
        }
        
        errors = strategy.validate_config(config)
        if errors:
            print(f"   ⚠️  Ошибки валидации: {errors}")
        else:
            print("   ✅ Конфигурация валидна")
        
        # Проверка готовности к торговле
        if strategy.is_ready_for_trading():
            print("   ✅ Готова к торговле")
        else:
            print("   ❌ Не готова к торговле")
        
        # Генерация сигналов
        try:
            signals = strategy.generate_signals(
                symbol=strategy.trading_pairs[0],
                amount=Decimal("100"),
                risk_level="medium"
            )
            print(f"   📈 Сгенерировано сигналов: {len(signals)}")
            
            if signals:
                latest_signal = signals[0]
                print(f"   🎯 Последний сигнал: {latest_signal.signal_type.value} "
                      f"(уверенность: {latest_signal.confidence})")
        
        except Exception as e:
            print(f"   ❌ Ошибка генерации сигналов: {e}")
        
        # Сохранение состояния
        try:
            success = strategy.save_state(f"state_{strategy.id}.pkl")
            if success:
                print("   💾 Состояние сохранено")
            else:
                print("   ❌ Ошибка сохранения состояния")
        except Exception as e:
            print(f"   ❌ Ошибка сохранения: {e}")
        
        print()


async def demonstrate_advanced_features():
    """
    Демонстрация продвинутых возможностей стратегий.
    """
    print("=== ПРОДВИНУТЫЕ ВОЗМОЖНОСТИ ===\n")
    
    # Создаем стратегию для демонстрации
    strategy = create_trend_following_strategy()
    
    print(f"🎯 Демонстрация стратегии: {strategy.name}")
    
    # 1. Обновление параметров во время работы
    print("\n1. Обновление параметров:")
    old_threshold = strategy.parameters.get_parameter("trend_strength")
    print(f"   Старый порог тренда: {old_threshold}")
    
    success = strategy.update_parameters({"trend_strength": 0.8})
    if success:
        new_threshold = strategy.parameters.get_parameter("trend_strength")
        print(f"   Новый порог тренда: {new_threshold}")
    
    # 2. Получение метрик производительности
    print("\n2. Метрики производительности:")
    metrics = strategy.get_performance_metrics()
    print(f"   Общее количество сделок: {metrics.get('total_trades', 0)}")
    print(f"   Прибыльные сделки: {metrics.get('winning_trades', 0)}")
    print(f"   Убыточные сделки: {metrics.get('losing_trades', 0)}")
    print(f"   Винрейт: {metrics.get('win_rate', '0%')}")
    
    # 3. Сброс состояния
    print("\n3. Сброс состояния:")
    strategy.reset()
    print("   ✅ Состояние сброшено")
    
    # 4. Проверка готовности
    print("\n4. Проверка готовности:")
    if strategy.is_ready_for_trading():
        print("   ✅ Стратегия готова к торговле")
    else:
        print("   ❌ Стратегия не готова к торговле")
    
    print()


def demonstrate_error_handling():
    """
    Демонстрация обработки ошибок.
    """
    print("=== ОБРАБОТКА ОШИБОК ===\n")
    
    # Создаем стратегию с некорректной конфигурацией
    print("1. Тест некорректной конфигурации:")
    bad_config = {
        "name": "",
        "strategy_type": "invalid_type",
        "trading_pairs": [],
        "parameters": {
            "stop_loss": -0.1,
            "take_profit": 15.0,
            "position_size": 2.0
        }
    }
    
    strategy = Strategy()
    errors = strategy.validate_config(bad_config)
    
    if errors:
        print("   Обнаружены ошибки валидации:")
        for error in errors:
            print(f"   ❌ {error}")
    else:
        print("   ✅ Конфигурация корректна")
    
    # Тест генерации сигналов для неподдерживаемой пары
    print("\n2. Тест генерации сигналов:")
    try:
        signals = strategy.generate_signals("UNSUPPORTED/PAIR")
        print(f"   Сгенерировано сигналов: {len(signals)}")
    except ValueError as e:
        print(f"   ❌ Ошибка: {e}")
    except Exception as e:
        print(f"   ❌ Неожиданная ошибка: {e}")
    
    print()


if __name__ == "__main__":
    """
    Запуск демонстрации.
    """
    print("🚀 ЗАПУСК ДЕМОНСТРАЦИИ СТРАТЕГИЙ ATB\n")
    
    # Запускаем демонстрации
    asyncio.run(demonstrate_strategy_usage())
    asyncio.run(demonstrate_advanced_features())
    demonstrate_error_handling()
    
    print("✅ Демонстрация завершена!")
    print("\n📚 Дополнительная информация:")
    print("   - Документация: docs/STRATEGY_GUIDE.md")
    print("   - API Reference: docs/API_REFERENCE.md")
    print("   - Примеры конфигураций: config/strategies/") 