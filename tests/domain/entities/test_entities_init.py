"""
Тесты для проверки импортов в entities/__init__.py.
"""

import pytest
from domain.entities import (
    Market,
    TradingPair,
    DomainOrder,
    DomainPosition,
    Trade,
    Portfolio,
    Account,
    MLModel,
    MLPrediction,
    Signal,
    RiskMetrics,
    Strategy,
    Pattern
)


class TestEntitiesImports:
    """Тесты импортов сущностей домена."""

    def test_all_entities_imported(self) -> None:
        """Тест, что все сущности импортированы корректно."""
        # Проверяем, что все классы импортированы и являются классами
        assert isinstance(Market, type)
        assert isinstance(TradingPair, type)
        assert isinstance(DomainOrder, type)
        assert isinstance(DomainPosition, type)
        assert isinstance(Trade, type)
        assert isinstance(Portfolio, type)
        assert isinstance(Account, type)
        assert isinstance(MLModel, type)
        assert isinstance(MLPrediction, type)
        assert isinstance(Signal, type)
        assert isinstance(RiskMetrics, type)
        assert isinstance(Strategy, type)
        assert isinstance(Pattern, type)

    def test_entities_instantiation(self) -> None:
        """Тест создания экземпляров сущностей."""
        # Проверяем, что можно создать экземпляры с минимальными параметрами
        try:
            market = Market()
            assert isinstance(market, Market)
        except Exception as e:
            pytest.fail(f"Не удалось создать Market: {e}")

        # TradingPair - это тип, а не класс
        assert TradingPair == str

        # Проверяем, что классы существуют и имеют правильные имена
        assert DomainOrder.__name__ == "Order"
        assert DomainPosition.__name__ == "Position"
        assert Trade.__name__ == "Trade"
        assert Portfolio.__name__ == "Portfolio"
        assert Account.__name__ == "Account"
        assert MLModel.__name__ == "Model"
        assert MLPrediction.__name__ == "Prediction"
        assert Signal.__name__ == "Signal"
        assert RiskMetrics.__name__ == "RiskMetrics"
        assert Strategy.__name__ == "Strategy"
        assert Pattern.__name__ == "Pattern"



    def test_entities_module_paths(self) -> None:
        """Тест путей модулей сущностей."""
        assert Market.__module__ == "domain.entities.market"
        # TradingPair - это тип, а не класс
        assert TradingPair == str
        assert DomainOrder.__module__ == "domain.entities.order"
        assert DomainPosition.__module__ == "domain.entities.position"
        assert Trade.__module__ == "domain.entities.trade"
        assert Portfolio.__module__ == "domain.entities.portfolio"
        assert Account.__module__ == "domain.entities.account"
        assert MLModel.__module__ == "domain.entities.ml"
        assert MLPrediction.__module__ == "domain.entities.ml"
        assert Signal.__module__ == "domain.entities.signal"
        assert RiskMetrics.__module__ == "domain.entities.risk"
        assert Strategy.__module__ == "domain.entities.strategy"
        assert Pattern.__module__ == "domain.entities.pattern" 