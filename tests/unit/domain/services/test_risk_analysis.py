"""
Unit тесты для domain.services.risk_analysis

Покрывает:
- Создание и валидацию RiskAnalysis объектов
- Расчет рисков
- Анализ портфеля
- Сериализация/десериализация
"""
import pytest
from decimal import Decimal
from datetime import datetime
from domain.services.risk_analysis import RiskAnalysis
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp


class TestRiskAnalysis:
    """Тесты для RiskAnalysis service"""

    def test_risk_analysis_creation(self):
        """Тест создания RiskAnalysis"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        assert risk_analysis.portfolio_value.amount == Decimal("100000.00")
        assert risk_analysis.total_exposure.amount == Decimal("50000.00")
        assert risk_analysis.max_drawdown == Decimal("0.15")
        assert risk_analysis.sharpe_ratio == Decimal("1.25")
        assert risk_analysis.var_95 == Decimal("0.05")

    def test_risk_analysis_calculate_exposure_ratio(self):
        """Тест расчета коэффициента экспозиции"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        exposure_ratio = risk_analysis.calculate_exposure_ratio()
        assert exposure_ratio == Decimal("0.50")

    def test_risk_analysis_calculate_risk_score(self):
        """Тест расчета оценки риска"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        risk_score = risk_analysis.calculate_risk_score()
        assert isinstance(risk_score, Decimal)
        assert risk_score >= Decimal("0")
        assert risk_score <= Decimal("1")

    def test_risk_analysis_is_high_risk(self):
        """Тест определения высокого риска"""
        # Высокий риск
        high_risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("90000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.30"),
            sharpe_ratio=Decimal("0.5"),
            var_95=Decimal("0.15"),
            timestamp=Timestamp(datetime.now())
        )
        assert high_risk_analysis.is_high_risk() is True
        
        # Низкий риск
        low_risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("20000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.05"),
            sharpe_ratio=Decimal("2.0"),
            var_95=Decimal("0.02"),
            timestamp=Timestamp(datetime.now())
        )
        assert low_risk_analysis.is_high_risk() is False

    def test_risk_analysis_get_risk_level(self):
        """Тест получения уровня риска"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        risk_level = risk_analysis.get_risk_level()
        assert risk_level in ["LOW", "MEDIUM", "HIGH"]

    def test_risk_analysis_to_dict(self):
        """Тест сериализации в словарь"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        data = risk_analysis.to_dict()
        assert data["portfolio_value"]["amount"] == "100000.00"
        assert data["total_exposure"]["amount"] == "50000.00"
        assert data["max_drawdown"] == "0.15"
        assert data["sharpe_ratio"] == "1.25"
        assert data["var_95"] == "0.05"

    def test_risk_analysis_from_dict(self):
        """Тест десериализации из словаря"""
        data = {
            "portfolio_value": {"amount": "100000.00", "currency": "USD"},
            "total_exposure": {"amount": "50000.00", "currency": "USD"},
            "max_drawdown": "0.15",
            "sharpe_ratio": "1.25",
            "var_95": "0.05",
            "timestamp": datetime.now().isoformat()
        }
        
        risk_analysis = RiskAnalysis.from_dict(data)
        assert risk_analysis.portfolio_value.amount == Decimal("100000.00")
        assert risk_analysis.total_exposure.amount == Decimal("50000.00")
        assert risk_analysis.max_drawdown == Decimal("0.15")
        assert risk_analysis.sharpe_ratio == Decimal("1.25")
        assert risk_analysis.var_95 == Decimal("0.05")

    def test_risk_analysis_validation(self):
        """Тест валидации данных"""
        # Невалидный max_drawdown
        with pytest.raises(ValueError):
            RiskAnalysis(
                portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
                total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
                max_drawdown=Decimal("1.5"),  # > 1.0
                sharpe_ratio=Decimal("1.25"),
                var_95=Decimal("0.05"),
                timestamp=Timestamp(datetime.now())
            )
        
        # Невалидный var_95
        with pytest.raises(ValueError):
            RiskAnalysis(
                portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
                total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
                max_drawdown=Decimal("0.15"),
                sharpe_ratio=Decimal("1.25"),
                var_95=Decimal("1.5"),  # > 1.0
                timestamp=Timestamp(datetime.now())
            )

    def test_risk_analysis_equality(self):
        """Тест равенства объектов"""
        risk_analysis1 = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        risk_analysis2 = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        assert risk_analysis1 == risk_analysis2
        assert hash(risk_analysis1) == hash(risk_analysis2)

    def test_risk_analysis_repr(self):
        """Тест строкового представления"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        repr_str = repr(risk_analysis)
        assert "RiskAnalysis" in repr_str
        assert "100000.00" in repr_str
        assert "0.15" in repr_str

    def test_risk_analysis_str(self):
        """Тест строкового представления для пользователя"""
        risk_analysis = RiskAnalysis(
            portfolio_value=Money(amount=Decimal("100000.00"), currency=Currency.USD),
            total_exposure=Money(amount=Decimal("50000.00"), currency=Currency.USD),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            var_95=Decimal("0.05"),
            timestamp=Timestamp(datetime.now())
        )
        
        str_repr = str(risk_analysis)
        assert "100000.00" in str_repr
        assert "0.15" in str_repr
        assert "1.25" in str_repr 