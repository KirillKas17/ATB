#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Примеры использования DDD архитектуры Advanced Trading Bot
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

# Импорты сервисов приложения
from application.services.trading_service import TradingService
from domain.entities.market import OHLCV, Market, MarketData
from domain.entities.ml import Model, ModelType, Prediction, PredictionType
from domain.entities.portfolio_fixed import Balance, Portfolio, Position
from domain.entities.risk import RiskManager, RiskProfile, RiskType
from domain.entities.strategy import Signal, SignalType, Strategy, StrategyType
# Импорты доменных сущностей
from domain.entities.trading import OrderSide, OrderType
from domain.repositories.portfolio_repository import \
    InMemoryPortfolioRepository
# Импорты репозиториев
from domain.repositories.trading_repository import InMemoryTradingRepository
from domain.value_objects.currency import Currency
# Импорты общих компонентов
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
# Импорты инфраструктуры
from shared.exceptions import InsufficientFundsError
from domain.exceptions import TradingError

# Заглушка для RiskLimit, так как класс не существует
class RiskLimit:
    def __init__(self, risk_type, name, max_value, warning_threshold, critical_threshold):
        self.risk_type = risk_type
        self.name = name
        self.max_value = max_value
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold


async def example_1_basic_trading_workflow():
    """Пример 1: Базовый торговый workflow"""
    print("=== Пример 1: Базовый торговый workflow ===")

    # Инициализация репозиториев
    trading_repo = InMemoryTradingRepository()
    portfolio_repo = InMemoryPortfolioRepository()

    # Создание портфеля
    portfolio = Portfolio(
        account_id="test_account",
        total_equity=Money(Decimal("10000"), Currency.USD),
        free_margin=Money(Decimal("10000"), Currency.USD),
    )
    await portfolio_repo.save_portfolio(portfolio)

    # Создание менеджера рисков
    risk_profile = RiskProfile(
        name="Test Profile",
        max_position_size=Money(Decimal("1000"), Currency.USD),
        max_portfolio_size=Money(Decimal("10000"), Currency.USD),
        max_drawdown=Percentage(Decimal("20")),
        max_leverage=Decimal("3"),
    )
    risk_manager = RiskManager(risk_profile=risk_profile)

    # Создание торгового сервиса
    trading_service = TradingService(
        trading_repository=trading_repo,
        portfolio_repository=portfolio_repo,
        risk_manager=risk_manager,
    )

    # Создание ордера
    order = await trading_service.create_order(
        portfolio_id=portfolio.id,
        trading_pair="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Volume(Decimal("0.001")),
        price=Price(Decimal("50000")),
    )

    print(f"Создан ордер: {order.id}")
    print(f"Статус: {order.status.value}")
    print(f"Количество: {order.quantity.value}")
    print(f"Цена: {order.price.value if order.price else 'market'}")

    # Исполнение ордера
    trade = await trading_service.execute_order(
        order_id=order.id,
        execution_price=Price(Decimal("50000")),
        execution_quantity=Volume(Decimal("0.001")),
    )

    print(f"Исполнена сделка: {trade.id}")
    print(f"Количество: {trade.quantity.value}")
    print(f"Цена: {trade.price.value}")
    print(f"Общая стоимость: {trade.total_value.value}")

    # Получение сводки портфеля
    summary = await trading_service.get_portfolio_summary(portfolio.id)
    print(f"Общая стоимость портфеля: {summary['total_equity']}")
    print(f"Количество открытых позиций: {summary['open_positions_count']}")


async def example_2_risk_management():
    """Пример 2: Управление рисками"""
    print("\n=== Пример 2: Управление рисками ===")

    # Создание профиля риска
    risk_profile = RiskProfile(
        name="Conservative Profile",
        max_position_size=Money(Decimal("500"), Currency.USD),
        max_portfolio_size=Money(Decimal("5000"), Currency.USD),
        max_drawdown=Percentage(Decimal("10")),
        max_leverage=Decimal("2"),
        stop_loss_percentage=Percentage(Decimal("2")),
        take_profit_percentage=Percentage(Decimal("4")),
    )

    # Создание менеджера рисков
    risk_manager = RiskManager(risk_profile=risk_profile)

    # Добавление лимитов
    position_size_limit = RiskLimit(
        risk_type=RiskType.POSITION_SIZE,
        name="Position Size Limit",
        max_value=Decimal("500"),
        warning_threshold=Decimal("400"),
        critical_threshold=Decimal("450"),
    )
    risk_manager.add_limit(position_size_limit)

    # Проверка торговых параметров
    trade_params = {
        "position_size": 600,  # Превышает лимит
        "trading_pair": "BTCUSDT",
        "side": "buy",
    }

    result = risk_manager.should_allow_trade(trade_params)
    print(f"Разрешена ли сделка: {result['allowed']}")
    print(f"Причины: {result['reasons']}")
    print(f"Уровень риска: {result['risk_level']}")

    # Проверка всех лимитов
    violations = risk_manager.check_all_limits()
    print(f"Нарушения лимитов: {len(violations)}")
    for violation in violations:
        print(
            f"- {violation['name']}: {violation['current_value']} > {violation['max_value']}"
        )


async def example_3_strategy_workflow():
    """Пример 3: Workflow стратегии"""
    print("\n=== Пример 3: Workflow стратегии ===")

    # Создание стратегии
    strategy = Strategy(
        name="Trend Following Strategy",
        description="Следует за трендом",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=["BTCUSDT", "ETHUSDT"],
        parameters=StrategyParameters(
            parameters={
                "ma_short": 20,
                "ma_long": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
            }
        ),
    )

    # Создание сигнала
    signal = Signal(
        strategy_id=strategy.id,
        trading_pair="BTCUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("0.8"),
        price=Price(Decimal("50000")),
        quantity=Decimal("0.001"),
        stop_loss=Price(Decimal("49000")),
        take_profit=Price(Decimal("52000")),
    )

    strategy.add_signal(signal)

    print(f"Стратегия: {strategy.name}")
    print(f"Тип: {strategy.strategy_type.value}")
    print(f"Торговые пары: {strategy.trading_pairs}")
    print(f"Параметры: {strategy.parameters.parameters}")

    print(f"Сигнал: {signal.signal_type.value}")
    print(f"Уверенность: {signal.confidence}")
    print(f"Цена: {signal.price.value if signal.price else 'market'}")
    print(f"Можно ли действовать: {signal.is_actionable}")


async def example_4_market_data_analysis():
    """Пример 4: Анализ рыночных данных"""
    print("\n=== Пример 4: Анализ рыночных данных ===")

    # Создание рыночных данных
    market_data = MarketData(trading_pair="BTCUSDT", timeframe=Timeframe.MINUTE_1)

    # Добавление свечей
    ohlcv1 = OHLCV(
        timestamp=datetime.now(),
        open=Price(Decimal("50000")),
        high=Price(Decimal("50100")),
        low=Price(Decimal("49900")),
        close=Price(Decimal("50050")),
        volume=Volume(Decimal("100")),
    )

    ohlcv2 = OHLCV(
        timestamp=datetime.now(),
        open=Price(Decimal("50050")),
        high=Price(Decimal("50200")),
        low=Price(Decimal("50000")),
        close=Price(Decimal("50150")),
        volume=Volume(Decimal("150")),
    )

    market_data.add_ohlcv(ohlcv1)
    market_data.add_ohlcv(ohlcv2)

    # Создание рынка
    market = Market(
        trading_pair="BTCUSDT",
        exchange="bybit",
        base_currency=Currency.BTC,
        quote_currency=Currency.USDT,
        market_data=market_data,
    )

    # Обновление состояния рынка
    market.update_market_state(
        {
            "regime": "trending_up",
            "volatility": 0.02,
            "trend_strength": 0.7,
            "support_level": 50000,
            "resistance_level": 51000,
            "indicators": {
                "rsi": 65,
                "macd": 0.5,
                "bollinger_upper": 50500,
                "bollinger_lower": 49500,
            },
        }
    )

    print(f"Рыночные данные для: {market.trading_pair}")
    print(f"Количество свечей: {len(market.market_data.ohlcv)}")
    print(f"Режим рынка: {market.market_state.regime.value}")
    print(f"Волатильность: {market.market_state.volatility}")
    print(f"Сила тренда: {market.market_state.trend_strength}")
    print(f"Трендовый ли рынок: {market.market_state.is_trending()}")
    print(f"Волатильный ли рынок: {market.market_state.is_volatile()}")

    latest_candle = market.market_data.get_latest_ohlcv()
    if latest_candle:
        print(f"Последняя свеча:")
        print(f"- Открытие: {latest_candle.open.value}")
        print(f"- Максимум: {latest_candle.high.value}")
        print(f"- Минимум: {latest_candle.low.value}")
        print(f"- Закрытие: {latest_candle.close.value}")
        print(f"- Объем: {latest_candle.volume.value}")
        print(f"- Бычья: {latest_candle.is_bullish}")
        print(f"- Доджи: {latest_candle.is_doji}")


async def example_5_ml_integration():
    """Пример 5: Интеграция с ML"""
    print("\n=== Пример 5: Интеграция с ML ===")

    # Создание ML модели
    model = Model(
        name="Price Predictor v1",
        description="Предсказание цены BTC",
        model_type=ModelType.LSTM,
        trading_pair="BTCUSDT",
        prediction_type=PredictionType.PRICE,
        features=["price", "volume", "rsi", "macd"],
        target="next_price",
        accuracy=Decimal("0.75"),
        precision=Decimal("0.72"),
        recall=Decimal("0.78"),
        f1_score=Decimal("0.75"),
    )

    model.activate()
    model.mark_trained()

    print(f"Модель: {model.name}")
    print(f"Тип: {model.model_type.value}")
    print(f"Точность: {model.accuracy}")
    print(f"Готова для предсказаний: {model.is_ready_for_prediction()}")

    # Создание предсказания
    prediction = Prediction(
        model_id=model.id,
        trading_pair="BTCUSDT",
        prediction_type=PredictionType.PRICE,
        value=Price(Decimal("50500")),
        confidence=Decimal("0.8"),
        features={"current_price": 50000, "volume": 1000, "rsi": 65, "macd": 0.5},
    )

    print(f"Предсказание:")
    print(f"- Цена: {prediction.value.value}")
    print(f"- Уверенность: {prediction.confidence}")
    print(f"- Высокая уверенность: {prediction.is_high_confidence()}")

    # Создание ансамбля моделей
    ensemble = ModelEnsemble(
        name="Ensemble Predictor", description="Ансамбль моделей для предсказания"
    )

    ensemble.add_model(model, weight=Decimal("0.6"))

    # Предсказание ансамблем
    ensemble_prediction = ensemble.predict(
        {"price": 50000, "volume": 1000, "rsi": 65, "macd": 0.5}
    )

    if ensemble_prediction:
        print(f"Предсказание ансамбля:")
        print(f"- Цена: {ensemble_prediction.value}")
        print(f"- Уверенность: {ensemble_prediction.confidence}")


async def example_6_portfolio_management():
    """Пример 6: Управление портфелем"""
    print("\n=== Пример 6: Управление портфелем ===")

    # Создание портфеля
    portfolio = Portfolio(
        account_id="demo_account",
        total_equity=Money(Decimal("10000"), Currency.USD),
        free_margin=Money(Decimal("10000"), Currency.USD),
    )

    # Добавление балансов
    usd_balance = Balance(
        currency=Currency.USD,
        available=Money(Decimal("8000"), Currency.USD),
        total=Money(Decimal("10000"), Currency.USD),
        locked=Money(Decimal("2000"), Currency.USD),
    )

    btc_balance = Balance(
        currency=Currency.BTC,
        available=Money(Decimal("0.1"), Currency.BTC),
        total=Money(Decimal("0.1"), Currency.BTC),
        locked=Money(Decimal("0"), Currency.BTC),
    )

    portfolio.add_balance(Currency.USD, usd_balance)
    portfolio.add_balance(Currency.BTC, btc_balance)

    # Добавление позиций
    btc_position = Position(
        trading_pair="BTCUSDT",
        side="long",
        quantity=Volume(Decimal("0.05")),
        average_price=Money(Decimal("50000"), Currency.USD),
        current_price=Money(Decimal("51000"), Currency.USD),
        leverage=Decimal("2"),
    )

    eth_position = Position(
        trading_pair="ETHUSDT",
        side="short",
        quantity=Volume(Decimal("1.0")),
        average_price=Money(Decimal("3000"), Currency.USD),
        current_price=Money(Decimal("2900"), Currency.USD),
        leverage=Decimal("1.5"),
    )

    portfolio.add_position("BTCUSDT", btc_position)
    portfolio.add_position("ETHUSDT", eth_position)

    # Обновление метрик
    portfolio._recalculate_metrics()

    print(f"Портфель: {portfolio.account_id}")
    print(f"Общая стоимость: {portfolio.total_equity.value}")
    print(f"Использованная маржа: {portfolio.total_margin_used.value}")
    print(f"Свободная маржа: {portfolio.free_margin.value}")
    print(f"Уровень маржи: {portfolio.margin_level.value}%")

    print(f"Балансы:")
    for currency, balance in portfolio.balances.items():
        print(f"- {currency.currency_code}: {balance.available.value} / {balance.total.value}")

    print(f"Позиции:")
    for trading_pair, position in portfolio.positions.items():
        print(
            f"- {trading_pair}: {position.side} {position.quantity.value} @ {position.average_price.value}"
        )
        print(f"  Текущая цена: {position.current_price.value}")
        print(f"  PnL: {position.total_pnl.value}")
        print(f"  Открыта: {position.is_open}")


async def example_7_error_handling():
    """Пример 7: Обработка ошибок"""
    print("\n=== Пример 7: Обработка ошибок ===")

    # Инициализация компонентов
    trading_repo = InMemoryTradingRepository()
    portfolio_repo = InMemoryPortfolioRepository()

    portfolio = Portfolio(
        account_id="test_account",
        total_equity=Money(Decimal("1000"), Currency.USD),
        free_margin=Money(Decimal("1000"), Currency.USD),
    )
    await portfolio_repo.save_portfolio(portfolio)

    risk_profile = RiskProfile(
        max_position_size=Money(Decimal("500"), Currency.USD),
        max_portfolio_size=Money(Decimal("1000"), Currency.USD),
    )
    risk_manager = RiskManager(risk_profile=risk_profile)

    trading_service = TradingService(
        trading_repository=trading_repo,
        portfolio_repository=portfolio_repo,
        risk_manager=risk_manager,
    )

    # Попытка создать ордер с недостаточными средствами
    try:
        order = await trading_service.create_order(
            portfolio_id=portfolio.id,
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.1")),  # 5000 USD при цене 50000
            price=Price(Decimal("50000")),
        )
    except InsufficientFundsError as e:
        print(f"Ошибка недостатка средств: {e}")
    except TradingError as e:
        print(f"Ошибка торговли: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

    # Попытка отменить несуществующий ордер
    try:
        await trading_service.cancel_order(uuid4())
    except TradingError as e:
        print(f"Ошибка при отмене ордера: {e}")


async def main():
    """Главная функция с примерами"""
    print("Примеры использования DDD архитектуры Advanced Trading Bot")
    print("=" * 60)

    await example_1_basic_trading_workflow()
    await example_2_risk_management()
    await example_3_strategy_workflow()
    await example_4_market_data_analysis()
    await example_5_ml_integration()
    await example_6_portfolio_management()
    await example_7_error_handling()

    print("\n" + "=" * 60)
    print("Все примеры выполнены успешно!")
    print("DDD архитектура обеспечивает:")
    print("- Четкое разделение ответственности")
    print("- Типобезопасность")
    print("- Легкое тестирование")
    print("- Масштабируемость")
    print("- Поддерживаемость")


if __name__ == "__main__":
    asyncio.run(main())
