"""
Модуль стресс-тестирования портфеля.

Содержит промышленные функции для стресс-тестирования,
применения сценариев и анализа устойчивости портфеля.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.type_definitions.risk_types import PortfolioRisk, StressTestResult
from domain.value_objects import Currency, Money

__all__ = [
    'apply_stress_scenario', 'perform_stress_test', 'generate_default_scenarios',
    'calc_scenario_impact', 'validate_scenario'
]

def generate_default_scenarios() -> List[Dict[str, Any]]:
    """Генерация стандартных сценариев стресс-тестирования."""
    return [
        {
            'name': 'Market Crash',
            'market_shock': -0.20,
            'volatility_increase': 2.0,
            'correlation_breakdown': True,
            'probability': Decimal('0.01'),
            'description': 'Глобальный крах рынка с ростом волатильности'
        },
        {
            'name': 'Interest Rate Hike',
            'rate_increase': 0.02,
            'duration_impact': -0.05,
            'probability': Decimal('0.05'),
            'description': 'Повышение процентных ставок'
        },
        {
            'name': 'Liquidity Crisis',
            'bid_ask_spread_increase': 3.0,
            'volume_decrease': 0.5,
            'probability': Decimal('0.02'),
            'description': 'Кризис ликвидности'
        },
        {
            'name': 'Currency Crisis',
            'currency_devaluation': -0.15,
            'probability': Decimal('0.03'),
            'description': 'Девальвация валюты'
        },
        {
            'name': 'Sector Rotation',
            'tech_sector_shock': -0.10,
            'defensive_sector_gain': 0.05,
            'probability': Decimal('0.08'),
            'description': 'Ротация секторов'
        }
    ]

def validate_scenario(scenario: Dict[str, Any]) -> bool:
    """Валидация сценария стресс-тестирования."""
    required_fields = ['name', 'probability']
    
    for field in required_fields:
        if field not in scenario:
            return False
    
    if not isinstance(scenario['name'], str) or not scenario['name']:
        return False
    
    if not isinstance(scenario['probability'], Decimal) or scenario['probability'] < 0:
        return False
    
    return True

def calc_scenario_impact(
    portfolio_risk: PortfolioRisk,
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """Расчёт воздействия сценария на портфель."""
    impact = {
        'portfolio_value_change': Money(Decimal('0'), Currency.USD),
        'var_change': Money(Decimal('0'), Currency.USD),
        'max_drawdown_change': Decimal('0'),
        'affected_positions': [],
        'correlation_breakdown': False,
        'liquidity_impact': Decimal('0'),
        'recovery_time_days': None
    }
    
    # Применяем рыночный шок
    if 'market_shock' in scenario:
        market_shock = scenario['market_shock']
        portfolio_value = float(portfolio_risk.total_value.value)
        value_change = portfolio_value * market_shock
        impact['portfolio_value_change'] = Money(Decimal(str(value_change)), Currency.USD)
    
    # Применяем изменение волатильности
    if 'volatility_increase' in scenario:
        volatility_increase = scenario['volatility_increase']
        current_var = float(portfolio_risk.risk_metrics.var_95.value)
        var_change = current_var * (volatility_increase - 1)
        impact['var_change'] = Money(Decimal(str(var_change)), Currency.USD)
    
    # Применяем изменение корреляций
    if scenario.get('correlation_breakdown', False):
        impact['correlation_breakdown'] = True
        # Упрощённо: увеличиваем риск на 20%
        risk_increase = float(portfolio_risk.total_risk) * 0.2
        impact['max_drawdown_change'] = -Decimal(str(risk_increase))
    
    # Применяем кризис ликвидности
    if 'bid_ask_spread_increase' in scenario:
        spread_increase = scenario['bid_ask_spread_increase']
        liquidity_impact = Decimal(str(spread_increase * 0.01))  # 1% от спреда
        impact['liquidity_impact'] = liquidity_impact
    
    # Определяем затронутые позиции
    for position in portfolio_risk.position_risks:
        if hasattr(position, 'contribution_to_portfolio_risk') and float(position.contribution_to_portfolio_risk.value) > 0.05:  # >5% вклада в риск
            if hasattr(position, 'symbol'):
                impact['affected_positions'].append(position.symbol)
    
    # Рассчитываем время восстановления
    if float(impact['portfolio_value_change'].value) < 0:
        # Упрощённо: 1 день на каждый 1% потерь
        recovery_days = int(abs(float(impact['portfolio_value_change'].value)) * 100)
        impact['recovery_time_days'] = max(1, min(recovery_days, 365))  # От 1 до 365 дней
    
    return impact

def apply_stress_scenario(
    portfolio_risk: PortfolioRisk, 
    scenario: Dict[str, Any]
) -> StressTestResult:
    """Применение стресс-сценария к портфелю."""
    if not validate_scenario(scenario):
        raise ValueError(f"Invalid stress test scenario: {scenario}")
    
    scenario_name = scenario.get('name', 'Unknown Scenario')
    scenario_probability = scenario.get('probability', Decimal('0.01'))
    
    # Рассчитываем воздействие
    impact = calc_scenario_impact(portfolio_risk, scenario)
    
    return StressTestResult(
        scenario_name=scenario_name,
        portfolio_value_change=impact['portfolio_value_change'],
        var_change=impact['var_change'],
        max_drawdown_change=impact['max_drawdown_change'],
        affected_positions=impact['affected_positions'],
        correlation_breakdown=impact['correlation_breakdown'],
        liquidity_impact=impact['liquidity_impact'],
        recovery_time_days=impact['recovery_time_days'],
        test_timestamp=datetime.now(),
        scenario_probability=scenario_probability
    )

def perform_stress_test(
    portfolio_risk: PortfolioRisk,
    scenarios: Optional[List[Dict[str, Any]]] = None
) -> List[StressTestResult]:
    """Выполнение полного стресс-тестирования."""
    if scenarios is None:
        scenarios = generate_default_scenarios()
    
    results = []
    
    for scenario in scenarios:
        try:
            result = apply_stress_scenario(portfolio_risk, scenario)
            results.append(result)
        except Exception as e:
            # Логируем ошибку и продолжаем с другими сценариями
            print(f"Error applying scenario {scenario.get('name', 'Unknown')}: {e}")
            continue
    
    return results 