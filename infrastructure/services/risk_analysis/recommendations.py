"""
Модуль генерации рекомендаций по рискам.

Содержит промышленные функции для генерации рекомендаций,
предложений по ребалансировке и предупреждений о рисках.
"""

from decimal import Decimal
from typing import Any, Dict, List

from domain.types.risk_types import PortfolioRisk, RiskLevel

__all__ = [
    'generate_risk_recommendations', 'generate_rebalancing_suggestions',
    'generate_risk_alerts', 'assess_risk_level', 'generate_portfolio_insights'
]

def assess_risk_level(portfolio_risk: PortfolioRisk) -> RiskLevel:
    """Оценка уровня риска портфеля."""
    total_risk = float(portfolio_risk.total_risk)
    max_drawdown = float(portfolio_risk.risk_metrics.max_drawdown)
    sharpe_ratio = float(portfolio_risk.risk_metrics.sharpe_ratio)
    
    # Критерии оценки риска
    if total_risk > 0.25 or max_drawdown < -0.20 or sharpe_ratio < 0.3:
        return RiskLevel.VERY_HIGH
    elif total_risk > 0.15 or max_drawdown < -0.15 or sharpe_ratio < 0.5:
        return RiskLevel.HIGH
    elif total_risk > 0.10 or max_drawdown < -0.10 or sharpe_ratio < 0.8:
        return RiskLevel.MEDIUM
    elif total_risk > 0.05 or max_drawdown < -0.05 or sharpe_ratio < 1.2:
        return RiskLevel.LOW
    else:
        return RiskLevel.VERY_LOW

def generate_risk_recommendations(portfolio_risk: PortfolioRisk) -> List[str]:
    """Генерация рекомендаций по рискам."""
    recommendations = []
    risk_level = assess_risk_level(portfolio_risk)
    
    # Общие рекомендации по уровню риска
    if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        recommendations.append("Consider reducing portfolio risk through diversification")
        recommendations.append("Review position sizes and consider reducing exposure")
    
    # Рекомендации по концентрации
    if portfolio_risk.concentration_risk > Decimal('0.3'):
        recommendations.append("Portfolio is highly concentrated - consider rebalancing")
        recommendations.append("Diversify across different asset classes and sectors")
    
    # Рекомендации по диверсификации
    if portfolio_risk.diversification_ratio < Decimal('1.2'):
        recommendations.append("Low diversification - consider adding uncorrelated assets")
        recommendations.append("Review correlation matrix for optimization opportunities")
    
    # Рекомендации по ликвидности
    if portfolio_risk.liquidity_risk > Decimal('0.1'):
        recommendations.append("High liquidity risk - consider more liquid instruments")
        recommendations.append("Review bid-ask spreads and trading volumes")
    
    # Рекомендации по валютному риску
    if portfolio_risk.currency_risk > Decimal('0.05'):
        recommendations.append("Significant currency risk - consider hedging strategies")
        recommendations.append("Review currency exposure and consider diversification")
    
    # Рекомендации по секторальному риску
    if portfolio_risk.sector_risk > Decimal('0.2'):
        recommendations.append("High sector concentration - diversify across sectors")
        recommendations.append("Consider defensive sectors for risk reduction")
    
    # Рекомендации по метрикам
    if portfolio_risk.risk_metrics.sharpe_ratio < Decimal('0.5'):
        recommendations.append("Low risk-adjusted returns - review strategy effectiveness")
        recommendations.append("Consider alternative investment strategies")
    
    if portfolio_risk.risk_metrics.max_drawdown < Decimal('-0.15'):
        recommendations.append("High maximum drawdown - implement stop-loss strategies")
        recommendations.append("Consider risk management overlay")
    
    return recommendations

def generate_rebalancing_suggestions(portfolio_risk: PortfolioRisk) -> List[str]:
    """Генерация предложений по ребалансировке."""
    suggestions = []
    
    # Анализ позиций с высоким вкладом в риск
    high_risk_positions = [
        pos for pos in portfolio_risk.position_risks 
        if pos.contribution_to_portfolio_risk > Decimal('0.1')
    ]
    
    for position in high_risk_positions:
        suggestions.append(
            f"Review position in {position.symbol} - high risk contribution "
            f"({position.contribution_to_portfolio_risk:.1%})"
        )
    
    # Анализ позиций с низкой корреляцией
    low_correlation_positions = [
        pos for pos in portfolio_risk.position_risks 
        if pos.correlation_with_portfolio < Decimal('0.3')
    ]
    
    if low_correlation_positions:
        suggestions.append(
            f"Consider increasing positions in low-correlation assets: "
            f"{', '.join([pos.symbol for pos in low_correlation_positions[:3]])}"
        )
    
    # Анализ позиций с высоким unrealized PnL
    high_pnl_positions = [
        pos for pos in portfolio_risk.position_risks 
        if pos.unrealized_pnl.value > 0 and pos.unrealized_pnl.value > pos.market_value.value * Decimal('0.1')
    ]
    
    for position in high_pnl_positions:
        suggestions.append(
            f"Consider taking profits on {position.symbol} - "
            f"unrealized PnL: {position.unrealized_pnl.value:.2f}"
        )
    
    # Анализ позиций с высокими убытками
    high_loss_positions = [
        pos for pos in portfolio_risk.position_risks 
        if pos.unrealized_pnl.value < 0 and abs(pos.unrealized_pnl.value) > pos.market_value.value * Decimal('0.05')
    ]
    
    for position in high_loss_positions:
        suggestions.append(
            f"Review losing position in {position.symbol} - "
            f"unrealized loss: {position.unrealized_pnl.value:.2f}"
        )
    
    return suggestions

def generate_risk_alerts(portfolio_risk: PortfolioRisk) -> List[str]:
    """Генерация предупреждений о рисках."""
    alerts = []
    
    # Критические алерты
    if portfolio_risk.risk_metrics.max_drawdown < Decimal('-0.20'):
        alerts.append("🚨 CRITICAL: Maximum drawdown exceeds 20% - immediate action required")
    
    if portfolio_risk.risk_metrics.sharpe_ratio < Decimal('0.3'):
        alerts.append("🚨 CRITICAL: Sharpe ratio below 0.3 - poor risk-adjusted performance")
    
    if portfolio_risk.concentration_risk > Decimal('0.5'):
        alerts.append("🚨 CRITICAL: Portfolio concentration risk exceeds 50%")
    
    # Высокие алерты
    if portfolio_risk.risk_metrics.max_drawdown < Decimal('-0.15'):
        alerts.append("⚠️ HIGH: Maximum drawdown exceeds 15% - review risk management")
    
    if portfolio_risk.risk_metrics.sharpe_ratio < Decimal('0.5'):
        alerts.append("⚠️ HIGH: Sharpe ratio below 0.5 - consider strategy adjustment")
    
    if portfolio_risk.total_risk > Decimal('0.25'):
        alerts.append("⚠️ HIGH: Portfolio risk exceeds 25% - consider risk reduction")
    
    # Средние алерты
    if portfolio_risk.risk_metrics.max_drawdown < Decimal('-0.10'):
        alerts.append("⚠️ MEDIUM: Maximum drawdown exceeds 10% - monitor closely")
    
    if portfolio_risk.liquidity_risk > Decimal('0.15'):
        alerts.append("⚠️ MEDIUM: High liquidity risk - may impact trading")
    
    if portfolio_risk.currency_risk > Decimal('0.08'):
        alerts.append("⚠️ MEDIUM: Significant currency risk exposure")
    
    # Информационные алерты
    if portfolio_risk.diversification_ratio < Decimal('1.1'):
        alerts.append("ℹ️ INFO: Low diversification ratio - optimization opportunity")
    
    if len(portfolio_risk.position_risks) < 5:
        alerts.append("ℹ️ INFO: Low number of positions - consider diversification")
    
    return alerts

def generate_portfolio_insights(portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
    """Генерация аналитических инсайтов по портфелю."""
    insights: Dict[str, Any] = {
        'risk_level': assess_risk_level(portfolio_risk).value,
        'key_metrics': {
            'total_risk': float(portfolio_risk.total_risk),
            'sharpe_ratio': float(portfolio_risk.risk_metrics.sharpe_ratio),
            'max_drawdown': float(portfolio_risk.risk_metrics.max_drawdown),
            'diversification_ratio': float(portfolio_risk.diversification_ratio),
            'concentration_risk': float(portfolio_risk.concentration_risk)
        },
        'top_risk_contributors': [],
        'diversification_opportunities': [],
        'performance_insights': []
    }
    
    # Топ-3 позиции по вкладу в риск
    sorted_positions = sorted(
        portfolio_risk.position_risks,
        key=lambda x: x.contribution_to_portfolio_risk,
        reverse=True
    )
    
    insights['top_risk_contributors'] = [
        {
            'symbol': pos.symbol,
            'contribution': float(pos.contribution_to_portfolio_risk),
            'market_value': float(pos.market_value.value)
        }
        for pos in sorted_positions[:3]
    ]
    
    # Возможности диверсификации
    low_correlation_positions = [
        pos for pos in portfolio_risk.position_risks 
        if pos.correlation_with_portfolio < Decimal('0.3')
    ]
    
    insights['diversification_opportunities'] = [
        {
            'symbol': pos.symbol,
            'correlation': float(pos.correlation_with_portfolio),
            'current_weight': float(pos.market_value.value / portfolio_risk.total_value.value)
        }
        for pos in low_correlation_positions[:5]
    ]
    
    # Инсайты по производительности
    performance_insights = insights['performance_insights']
    if isinstance(performance_insights, list):
        if portfolio_risk.risk_metrics.sharpe_ratio > Decimal('1.5'):
            performance_insights.append("Excellent risk-adjusted returns")
        elif portfolio_risk.risk_metrics.sharpe_ratio > Decimal('1.0'):
            performance_insights.append("Good risk-adjusted returns")
        else:
            performance_insights.append("Below-average risk-adjusted returns")
        
        if portfolio_risk.diversification_ratio > Decimal('1.5'):
            performance_insights.append("Well-diversified portfolio")
        else:
            performance_insights.append("Limited diversification benefits")
    
    return insights 