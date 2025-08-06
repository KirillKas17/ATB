#!/usr/bin/env python3
"""
Примеры использования эволюционных агентов ATB
Демонстрирует различные сценарии и подходы к использованию
"""

import asyncio
from datetime import datetime

from shared.numpy_utils import np
import pandas as pd
# Импорт оригинальных агентов
from agents.agent_market_regime import MarketRegimeAgent
from agents.agent_portfolio import PortfolioAgent
from agents.agent_risk import RiskAgent
# Импорт эволюционных агентов
# Импорт эволюционной системы
from core.evolution_integration import (evolution_integration, start_evolution,
                                        stop_evolution)
from loguru import logger


def create_sample_market_data() -> Any:
    return None
    """Создание тестовых рыночных данных"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")

    # Симуляция цен BTC
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000000, 5000000, 100),
        }
    )

    return data


async def example_1_classic_only() -> None:
    """Пример 1: Использование только классических агентов"""
    logger.info("=== Пример 1: Только классические агенты ===")

    # Создание тестовых данных
    market_data = create_sample_market_data()

    # Использование классических агентов
    market_regime = MarketRegimeAgent()
    risk_agent = RiskAgent()
    portfolio_agent = PortfolioAgent()

    # Анализ рыночного режима
    regime, confidence = market_regime.detect_regime(market_data)
    logger.info(
        f"Классический анализ: режим={regime.name}, уверенность={confidence:.3f}"
    )

    # Анализ рисков
    risk_score = risk_agent.calculate_risk(market_data)
    logger.info(f"Классический анализ рисков: {risk_score:.3f}")

    # Управление портфелем
    portfolio_weights = portfolio_agent.calculate_weights(market_data)
    logger.info(f"Классические веса портфеля: {portfolio_weights}")


async def example_2_evolutionary_only() -> None:
    """Пример 2: Использование только эволюционных агентов"""
    logger.info("=== Пример 2: Только эволюционные агенты ===")

    # Запуск эволюционной системы
    await start_evolution()

    # Создание тестовых данных
    market_data = create_sample_market_data()

    # Использование эволюционных агентов
    evolvable_regime = evolution_integration.get_agent("market_regime")
    evolvable_risk = evolution_integration.get_agent("risk")
    evolvable_portfolio = evolution_integration.get_agent("portfolio")

    # Анализ рыночного режима с ML
    regime, confidence = await evolvable_regime.detect_regime(market_data)
    logger.info(
        f"Эволюционный анализ: режим={regime.name}, уверенность={confidence:.3f}"
    )

    # Анализ рисков с ML
    risk_score = await evolvable_risk.calculate_risk(market_data)
    logger.info(f"Эволюционный анализ рисков: {risk_score:.3f}")

    # Управление портфелем с ML
    portfolio_weights = await evolvable_portfolio.calculate_weights(market_data)
    logger.info(f"Эволюционные веса портфеля: {portfolio_weights}")

    # Остановка эволюционной системы
    await stop_evolution()


async def example_3_hybrid_approach() -> None:
    """Пример 3: Гибридный подход - сравнение результатов"""
    logger.info("=== Пример 3: Гибридный подход ===")

    # Запуск эволюционной системы
    await start_evolution()

    # Создание тестовых данных
    market_data = create_sample_market_data()

    # Классические агенты
    classic_regime = MarketRegimeAgent()
    RiskAgent()

    # Эволюционные агенты
    evolvable_regime = evolution_integration.get_agent("market_regime")
    evolution_integration.get_agent("risk")

    # Сравнение результатов
    classic_regime_result = classic_regime.detect_regime(market_data)
    evolvable_regime_result = await evolvable_regime.detect_regime(market_data)

    logger.info(
        f"Классический режим: {classic_regime_result[0].name}, уверенность={classic_regime_result[1]:.3f}"
    )
    logger.info(
        f"Эволюционный режим: {evolvable_regime_result[0].name}, уверенность={evolvable_regime_result[1]:.3f}"
    )

    # Анализ различий
    confidence_diff = abs(classic_regime_result[1] - evolvable_regime_result[1])
    logger.info(f"Разница в уверенности: {confidence_diff:.3f}")

    if confidence_diff > 0.3:
        logger.warning(
            "Большая разница в результатах - рекомендуется дополнительная проверка"
        )

    # Остановка эволюционной системы
    await stop_evolution()


async def example_4_evolution_monitoring() -> None:
    """Пример 4: Мониторинг эволюции агентов"""
    logger.info("=== Пример 4: Мониторинг эволюции ===")

    # Запуск эволюционной системы
    await start_evolution()

    # Создание тестовых данных для обучения
    market_data = create_sample_market_data()
    training_data = {
        "market_data": market_data,
        "features": [0.1] * 20,
        "targets": [0.5] * 5,
        "volatility": [0.1, 0.15, 0.12, 0.18, 0.11],
        "timestamp": datetime.now(),
    }

    # Получение агентов
    agents = evolution_integration.agents

    # Мониторинг до обучения
    logger.info("=== Статус до обучения ===")
    for name, agent in agents.items():
        stats = agent.get_evolution_stats()
        logger.info(
            f"{name}: performance={stats['performance']:.3f}, confidence={stats['confidence']:.3f}"
        )

    # Обучение агентов
    logger.info("=== Обучение агентов ===")
    for name, agent in agents.items():
        success = await agent.learn(training_data)
        logger.info(f"{name} обучение: {'успешно' if success else 'неудачно'}")

    # Мониторинг после обучения
    logger.info("=== Статус после обучения ===")
    for name, agent in agents.items():
        stats = agent.get_evolution_stats()
        logger.info(
            f"{name}: performance={stats['performance']:.3f}, confidence={stats['confidence']:.3f}"
        )

    # Получение общего статуса системы
    system_status = evolution_integration.get_system_status()
    logger.info(f"Статус эволюционной системы: {system_status}")

    # Остановка эволюционной системы
    await stop_evolution()


async def example_5_adaptive_trading() -> None:
    """Пример 5: Адаптивная торговля с эволюционными агентами"""
    logger.info("=== Пример 5: Адаптивная торговля ===")

    # Запуск эволюционной системы
    await start_evolution()

    # Создание симуляции торговых данных
    trading_sessions = []
    for i in range(10):
        session_data = {
            "market_data": create_sample_market_data(),
            "timestamp": datetime.now(),
            "session_id": i,
        }
        trading_sessions.append(session_data)

    # Симуляция адаптивной торговли
    for i, session in enumerate(trading_sessions):
        logger.info(f"=== Торговая сессия {i+1} ===")

        # Адаптация агентов к новым данным
        for name, agent in evolution_integration.agents.items():
            success = await agent.adapt(session)
            if success:
                logger.debug(f"{name} адаптировался к сессии {i+1}")

        # Анализ рынка
        market_regime = evolution_integration.get_agent("market_regime")
        regime, confidence = await market_regime.detect_regime(session["market_data"])

        # Принятие торговых решений на основе режима
        if regime.name == "TREND" and confidence > 0.7:
            logger.info(f"Сессия {i+1}: Сильный тренд - агрессивная торговля")
        elif regime.name == "SIDEWAYS" and confidence > 0.7:
            logger.info(f"Сессия {i+1}: Боковик - консервативная торговля")
        else:
            logger.info(f"Сессия {i+1}: Неопределенность - минимальная торговля")

        # Обучение на результатах сессии
        for name, agent in evolution_integration.agents.items():
            await agent.learn(session)

    # Финальная эволюция
    logger.info("=== Финальная эволюция ===")
    for name, agent in evolution_integration.agents.items():
        success = await agent.evolve({"timestamp": datetime.now()})
        logger.info(f"{name} эволюция: {'успешно' if success else 'неудачно'}")

    # Остановка эволюционной системы
    await stop_evolution()


async def example_6_performance_comparison() -> None:
    """Пример 6: Сравнение производительности классических и эволюционных агентов"""
    logger.info("=== Пример 6: Сравнение производительности ===")

    # Запуск эволюционной системы
    await start_evolution()

    # Создание тестовых данных
    test_data = create_sample_market_data()

    # Классические агенты
    classic_agents = {
        "market_regime": MarketRegimeAgent(),
        "risk": RiskAgent(),
        "portfolio": PortfolioAgent(),
    }

    # Эволюционные агенты
    evolvable_agents = {
        "market_regime": evolution_integration.get_agent("market_regime"),
        "risk": evolution_integration.get_agent("risk"),
        "portfolio": evolution_integration.get_agent("portfolio"),
    }

    # Сравнение производительности
    results = {}

    for agent_type in ["market_regime", "risk", "portfolio"]:
        logger.info(f"=== Сравнение {agent_type} ===")

        # Тестирование классического агента
        start_time = datetime.now()
        if agent_type == "market_regime":
            classic_result = classic_agents[agent_type].detect_regime(test_data)
        elif agent_type == "risk":
            classic_result = classic_agents[agent_type].calculate_risk(test_data)
        else:
            classic_result = classic_agents[agent_type].calculate_weights(test_data)
        classic_time = (datetime.now() - start_time).total_seconds()

        # Тестирование эволюционного агента
        start_time = datetime.now()
        if agent_type == "market_regime":
            evolvable_result = await evolvable_agents[agent_type].detect_regime(
                test_data
            )
        elif agent_type == "risk":
            evolvable_result = await evolvable_agents[agent_type].calculate_risk(
                test_data
            )
        else:
            evolvable_result = await evolvable_agents[agent_type].calculate_weights(
                test_data
            )
        evolvable_time = (datetime.now() - start_time).total_seconds()

        # Сохранение результатов
        results[agent_type] = {
            "classic": {"result": classic_result, "time": classic_time},
            "evolvable": {"result": evolvable_result, "time": evolvable_time},
        }

        logger.info(f"Классический: {classic_result}, время: {classic_time:.4f}с")
        logger.info(f"Эволюционный: {evolvable_result}, время: {evolvable_time:.4f}с")
        logger.info(f"Разница во времени: {evolvable_time - classic_time:.4f}с")

    # Остановка эволюционной системы
    await stop_evolution()

    return results


async def main() -> None:
    """Основная функция с примерами"""
    logger.info("Запуск примеров использования эволюционных агентов")

    try:
        # Пример 1: Только классические агенты
        await example_1_classic_only()

        # Пример 2: Только эволюционные агенты
        await example_2_evolutionary_only()

        # Пример 3: Гибридный подход
        await example_3_hybrid_approach()

        # Пример 4: Мониторинг эволюции
        await example_4_evolution_monitoring()

        # Пример 5: Адаптивная торговля
        await example_5_adaptive_trading()

        # Пример 6: Сравнение производительности
        await example_6_performance_comparison()

        logger.info("Все примеры выполнены успешно!")

    except Exception as e:
        logger.error(f"Ошибка в примерах: {e}")
        raise


if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Запуск примеров
    asyncio.run(main())
