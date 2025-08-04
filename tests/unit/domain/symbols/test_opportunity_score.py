"""Тесты для калькулятора opportunity score."""
from domain.symbols.opportunity_score import OpportunityScoreCalculator
from domain.type_definitions import MarketDataFrame, OrderBookData
def test_opportunity_score_calculator_init() -> None:
    calc = OpportunityScoreCalculator()
    assert calc is not None
    assert calc.config is not None
def test_opportunity_score_calculator_basic() -> None:
    calc = OpportunityScoreCalculator()
    # Создаем минимальные данные
    symbol = "BTCUSDT"
    # Создаем DataFrame с OHLCV данными
    df = pd.DataFrame({
        'open': [50000, 50100, 50200, 50300, 50400],
        'high': [50100, 50200, 50300, 50400, 50500],
        'low': [49900, 50000, 50100, 50200, 50300],
        'close': [50100, 50200, 50300, 50400, 50500],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    market_data = cast(MarketDataFrame, df)
    # Создаем данные стакана заявок
    order_book = cast(OrderBookData, {
        'bids': [[50000, 1.0], [49999, 2.0]],
        'asks': [[50001, 1.0], [50002, 2.0]]
    })
    result = calc.calculate_opportunity_score(symbol, market_data, order_book)
    assert result['symbol'] == symbol
    assert 'total_score' in result
    assert 'confidence' in result
    assert 'market_phase' in result
def test_opportunity_score_config_validation() -> None:
    # Проверяем, что калькулятор создается с валидной конфигурацией
    calc = OpportunityScoreCalculator()
    assert calc.config is not None
    # Проверяем наличие основных параметров конфигурации
    assert hasattr(calc.config, 'alpha1_liquidity_score') 
