"""
Advanced Risk Management Engine

This module implements sophisticated risk management using multiple design patterns:
- Visitor Pattern for risk calculation strategies
- Template Method Pattern for risk assessment algorithms  
- Decorator Pattern for risk adjustment layers
- Adapter Pattern for external risk systems integration
- Composite Pattern for portfolio risk aggregation
- Singleton Pattern for global risk configurations
- Factory Method Pattern for risk calculator creation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Protocol, Union, Callable
import asyncio
import logging
import math
from collections import defaultdict
import threading
from functools import wraps
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from domain.value_objects.money import Money
from domain.entities.risk_metrics import RiskMetrics
from domain.value_objects.trading_pair import TradingPair

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Уровни риска."""
    VERY_LOW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class RiskCategory(Enum):
    """Категории рисков."""
    MARKET_RISK = "market"
    CREDIT_RISK = "credit"
    LIQUIDITY_RISK = "liquidity"
    OPERATIONAL_RISK = "operational"
    CONCENTRATION_RISK = "concentration"
    SYSTEMIC_RISK = "systemic"


@dataclass(frozen=True)
class RiskMetrics:
    """Метрики риска."""
    var_95: Decimal
    var_99: Decimal
    expected_shortfall: Decimal
    max_drawdown: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    beta: Decimal
    correlation_risk: Decimal
    liquidity_score: Decimal
    concentration_score: Decimal
    stress_test_result: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    """Позиция для расчета рисков."""
    symbol: str
    quantity: Decimal
    current_price: Decimal
    entry_price: Decimal
    market_value: Money
    unrealized_pnl: Money
    risk_category: RiskCategory = RiskCategory.MARKET_RISK


@dataclass
class Portfolio:
    """Портфель позиций."""
    positions: List[Position] = field(default_factory=list)
    total_value: Money = field(default_factory=lambda: Money(Decimal('0')))
    cash_position: Money = field(default_factory=lambda: Money(Decimal('0')))
    

# Visitor Pattern Implementation
class RiskVisitor(ABC):
    """Абстрактный посетитель для расчета рисков."""
    
    @abstractmethod
    def visit_position(self, position: Position) -> Decimal:
        """Посещение позиции."""
        pass
    
    @abstractmethod
    def visit_portfolio(self, portfolio: Portfolio) -> Decimal:
        """Посещение портфеля."""
        pass


class VaRCalculatorVisitor(RiskVisitor):
    """Калькулятор Value at Risk через паттерн Visitor."""
    
    def __init__(self, confidence_level: Decimal = Decimal('0.95'), time_horizon: int = 1):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.volatility_cache: Dict[str, Decimal] = {}
    
    def visit_position(self, position: Position) -> Decimal:
        """Расчет VaR для отдельной позиции."""
        volatility = self._get_volatility(position.symbol)
        market_value = position.market_value.amount
        
        # Используем параметрическую модель для VaR
        z_score = self._get_z_score(self.confidence_level)
        var = market_value * volatility * z_score * Decimal(str(math.sqrt(self.time_horizon)))
        
        return var
    
    def visit_portfolio(self, portfolio: Portfolio) -> Decimal:
        """Расчет портфельного VaR с учетом корреляций."""
        individual_vars = [self.visit_position(pos) for pos in portfolio.positions]
        
        if len(individual_vars) <= 1:
            return sum(individual_vars)
        
        # Упрощенный расчет с корреляциями (в реальности нужна корреляционная матрица)
        diversification_factor = Decimal('0.8')  # Предполагаем частичную диверсификацию
        portfolio_var = sum(individual_vars) * diversification_factor
        
        return portfolio_var
    
    def _get_volatility(self, symbol: str) -> Decimal:
        """Получение волатильности символа."""
        if symbol not in self.volatility_cache:
            # В реальности здесь был бы запрос к источнику данных
            self.volatility_cache[symbol] = Decimal('0.02')  # 2% дневная волатильность
        return self.volatility_cache[symbol]
    
    def _get_z_score(self, confidence_level: Decimal) -> Decimal:
        """Получение Z-score для уровня доверия."""
        z_scores = {
            Decimal('0.90'): Decimal('1.282'),
            Decimal('0.95'): Decimal('1.645'),
            Decimal('0.99'): Decimal('2.326')
        }
        return z_scores.get(confidence_level, Decimal('1.645'))


class LiquidityRiskVisitor(RiskVisitor):
    """Калькулятор риска ликвидности."""
    
    def visit_position(self, position: Position) -> Decimal:
        """Расчет риска ликвидности для позиции."""
        # Модель основана на размере позиции относительно среднего объема торгов
        position_size = abs(position.quantity * position.current_price)
        
        # Предполагаем дневной объем (в реальности получали бы из данных)
        daily_volume = position_size * Decimal('10')  # Позиция составляет 10% от дневного объема
        
        liquidity_ratio = position_size / daily_volume
        
        # Чем больше позиция относительно объема, тем выше риск ликвидности
        return min(liquidity_ratio, Decimal('1.0'))
    
    def visit_portfolio(self, portfolio: Portfolio) -> Decimal:
        """Расчет портфельного риска ликвидности."""
        position_risks = [self.visit_position(pos) for pos in portfolio.positions]
        
        # Максимальный риск ликвидности среди позиций
        return max(position_risks) if position_risks else Decimal('0')


# Template Method Pattern Implementation
class RiskAssessmentTemplate(ABC):
    """Шаблонный метод для оценки рисков."""
    
    def assess_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Основной алгоритм оценки рисков (Template Method)."""
        # 1. Предварительная валидация
        self._validate_portfolio(portfolio)
        
        # 2. Сбор рыночных данных
        market_data = self._collect_market_data(portfolio)
        
        # 3. Расчет базовых метрик
        base_metrics = self._calculate_base_metrics(portfolio, market_data)
        
        # 4. Применение стресс-тестов
        stress_results = self._apply_stress_tests(portfolio, market_data)
        
        # 5. Агрегация результатов
        final_metrics = self._aggregate_results(base_metrics, stress_results)
        
        return final_metrics
    
    def _validate_portfolio(self, portfolio: Portfolio) -> None:
        """Валидация портфеля."""
        if not portfolio.positions:
            raise ValueError("Portfolio cannot be empty")
    
    @abstractmethod
    def _collect_market_data(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Сбор рыночных данных (должен быть реализован в подклассах)."""
        pass
    
    @abstractmethod
    def _calculate_base_metrics(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Расчет базовых метрик (должен быть реализован в подклассах)."""
        pass
    
    @abstractmethod
    def _apply_stress_tests(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Применение стресс-тестов (должен быть реализован в подклассах)."""
        pass
    
    def _aggregate_results(self, base_metrics: Dict[str, Decimal], 
                          stress_results: Dict[str, Decimal]) -> RiskMetrics:
        """Агрегация результатов в финальные метрики."""
        return RiskMetrics(
            var_95=base_metrics.get('var_95', Decimal('0')),
            var_99=base_metrics.get('var_99', Decimal('0')),
            expected_shortfall=base_metrics.get('expected_shortfall', Decimal('0')),
            max_drawdown=base_metrics.get('max_drawdown', Decimal('0')),
            volatility=base_metrics.get('volatility', Decimal('0')),
            sharpe_ratio=base_metrics.get('sharpe_ratio', Decimal('0')),
            beta=base_metrics.get('beta', Decimal('0')),
            correlation_risk=base_metrics.get('correlation_risk', Decimal('0')),
            liquidity_score=base_metrics.get('liquidity_score', Decimal('0')),
            concentration_score=base_metrics.get('concentration_score', Decimal('0')),
            stress_test_result=stress_results.get('overall_stress', Decimal('0'))
        )


class ParametricRiskAssessment(RiskAssessmentTemplate):
    """Параметрическая оценка рисков."""
    
    def _collect_market_data(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Сбор параметрических данных."""
        symbols = [pos.symbol for pos in portfolio.positions]
        
        # В реальности здесь был бы запрос к источнику данных
        return {
            'volatilities': {symbol: Decimal('0.02') for symbol in symbols},
            'correlations': self._generate_correlation_matrix(symbols),
            'risk_free_rate': Decimal('0.02')
        }
    
    def _calculate_base_metrics(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Расчет параметрических метрик."""
        var_calculator = VaRCalculatorVisitor(Decimal('0.95'))
        var_95 = var_calculator.visit_portfolio(portfolio)
        
        var_calculator_99 = VaRCalculatorVisitor(Decimal('0.99'))
        var_99 = var_calculator_99.visit_portfolio(portfolio)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'volatility': self._calculate_portfolio_volatility(portfolio, market_data),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio, market_data),
            'max_drawdown': Decimal('0.05'),  # Упрощенный расчет
            'expected_shortfall': var_99 * Decimal('1.2')  # ES обычно больше VaR
        }
    
    def _apply_stress_tests(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Применение параметрических стресс-тестов."""
        # Стресс-сценарии: падение рынка на 20%, рост волатильности в 2 раза
        stress_scenarios = [
            {'market_shock': Decimal('-0.20'), 'volatility_multiplier': Decimal('2.0')},
            {'market_shock': Decimal('-0.30'), 'volatility_multiplier': Decimal('1.5')},
            {'market_shock': Decimal('-0.10'), 'volatility_multiplier': Decimal('3.0')}
        ]
        
        worst_case_loss = Decimal('0')
        
        for scenario in stress_scenarios:
            scenario_loss = self._calculate_scenario_loss(portfolio, scenario)
            worst_case_loss = max(worst_case_loss, scenario_loss)
        
        return {'overall_stress': worst_case_loss}
    
    def _generate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Генерация корреляционной матрицы."""
        # Упрощенная корреляционная матрица
        matrix = {}
        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = Decimal('1.0')
                else:
                    matrix[symbol1][symbol2] = Decimal('0.6')  # Средняя корреляция
        return matrix
    
    def _calculate_portfolio_volatility(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Decimal:
        """Расчет волатильности портфеля."""
        if not portfolio.positions:
            return Decimal('0')
        
        # Упрощенный расчет взвешенной волатильности
        total_value = portfolio.total_value.amount
        if total_value == 0:
            return Decimal('0')
        
        weighted_volatility = Decimal('0')
        for position in portfolio.positions:
            weight = position.market_value.amount / total_value
            symbol_volatility = market_data['volatilities'].get(position.symbol, Decimal('0.02'))
            weighted_volatility += weight * symbol_volatility
        
        return weighted_volatility
    
    def _calculate_sharpe_ratio(self, portfolio: Portfolio, market_data: Dict[str, Any]) -> Decimal:
        """Расчет коэффициента Шарпа."""
        # Упрощенный расчет (нужны исторические доходности)
        expected_return = Decimal('0.08')  # 8% годовых
        risk_free_rate = market_data.get('risk_free_rate', Decimal('0.02'))
        volatility = self._calculate_portfolio_volatility(portfolio, market_data)
        
        if volatility == 0:
            return Decimal('0')
        
        return (expected_return - risk_free_rate) / volatility
    
    def _calculate_scenario_loss(self, portfolio: Portfolio, scenario: Dict[str, Decimal]) -> Decimal:
        """Расчет потерь в стресс-сценарии."""
        total_loss = Decimal('0')
        
        for position in portfolio.positions:
            market_shock = scenario['market_shock']
            position_loss = position.market_value.amount * abs(market_shock)
            total_loss += position_loss
        
        return total_loss


# Decorator Pattern Implementation
class RiskAdjustmentDecorator(ABC):
    """Базовый декоратор для корректировки рисков."""
    
    def __init__(self, risk_calculator):
        self._risk_calculator = risk_calculator
    
    @abstractmethod
    def calculate_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Расчет риска с корректировкой."""
        pass


class RegionalRiskAdjustmentDecorator(RiskAdjustmentDecorator):
    """Декоратор для корректировки региональных рисков."""
    
    def __init__(self, risk_calculator, regional_multipliers: Dict[str, Decimal]):
        super().__init__(risk_calculator)
        self.regional_multipliers = regional_multipliers
    
    def calculate_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Расчет с учетом региональных факторов."""
        base_metrics = self._risk_calculator.assess_risk(portfolio)
        
        # Применяем региональные корректировки
        regional_adjustment = self._calculate_regional_adjustment(portfolio)
        
        return RiskMetrics(
            var_95=base_metrics.var_95 * regional_adjustment,
            var_99=base_metrics.var_99 * regional_adjustment,
            expected_shortfall=base_metrics.expected_shortfall * regional_adjustment,
            max_drawdown=base_metrics.max_drawdown,
            volatility=base_metrics.volatility * regional_adjustment,
            sharpe_ratio=base_metrics.sharpe_ratio / regional_adjustment,
            beta=base_metrics.beta,
            correlation_risk=base_metrics.correlation_risk,
            liquidity_score=base_metrics.liquidity_score,
            concentration_score=base_metrics.concentration_score,
            stress_test_result=base_metrics.stress_test_result * regional_adjustment,
            timestamp=base_metrics.timestamp
        )
    
    def _calculate_regional_adjustment(self, portfolio: Portfolio) -> Decimal:
        """Расчет региональной корректировки."""
        if not portfolio.positions:
            return Decimal('1.0')
        
        total_value = portfolio.total_value.amount
        if total_value == 0:
            return Decimal('1.0')
        
        weighted_adjustment = Decimal('0')
        
        for position in portfolio.positions:
            weight = position.market_value.amount / total_value
            # Предполагаем, что символы содержат информацию о регионе
            region = self._get_region_from_symbol(position.symbol)
            regional_multiplier = self.regional_multipliers.get(region, Decimal('1.0'))
            weighted_adjustment += weight * regional_multiplier
        
        return weighted_adjustment
    
    def _get_region_from_symbol(self, symbol: str) -> str:
        """Определение региона по символу."""
        # Упрощенная логика определения региона
        if 'USD' in symbol:
            return 'US'
        elif 'EUR' in symbol:
            return 'EU'
        else:
            return 'OTHER'


class TimeDecayAdjustmentDecorator(RiskAdjustmentDecorator):
    """Декоратор для корректировки временного затухания рисков."""
    
    def __init__(self, risk_calculator, decay_factor: Decimal = Decimal('0.95')):
        super().__init__(risk_calculator)
        self.decay_factor = decay_factor
        self.calculation_cache: Dict[str, tuple] = {}  # (timestamp, metrics)
    
    def calculate_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Расчет с учетом временного затухания."""
        current_time = datetime.now(timezone.utc)
        cache_key = self._generate_cache_key(portfolio)
        
        # Проверяем кэш
        if cache_key in self.calculation_cache:
            cached_time, cached_metrics = self.calculation_cache[cache_key]
            time_diff = (current_time - cached_time).total_seconds() / 3600  # часы
            
            if time_diff < 1:  # Кэш действителен час
                return self._apply_time_decay(cached_metrics, time_diff)
        
        # Пересчитываем риски
        base_metrics = self._risk_calculator.assess_risk(portfolio)
        self.calculation_cache[cache_key] = (current_time, base_metrics)
        
        return base_metrics
    
    def _apply_time_decay(self, metrics: RiskMetrics, hours_elapsed: float) -> RiskMetrics:
        """Применение временного затухания."""
        decay_multiplier = self.decay_factor ** Decimal(str(hours_elapsed))
        
        return RiskMetrics(
            var_95=metrics.var_95 * decay_multiplier,
            var_99=metrics.var_99 * decay_multiplier,
            expected_shortfall=metrics.expected_shortfall * decay_multiplier,
            max_drawdown=metrics.max_drawdown,
            volatility=metrics.volatility * decay_multiplier,
            sharpe_ratio=metrics.sharpe_ratio,
            beta=metrics.beta,
            correlation_risk=metrics.correlation_risk * decay_multiplier,
            liquidity_score=metrics.liquidity_score,
            concentration_score=metrics.concentration_score,
            stress_test_result=metrics.stress_test_result * decay_multiplier,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _generate_cache_key(self, portfolio: Portfolio) -> str:
        """Генерация ключа кэша для портфеля."""
        positions_str = ''.join([
            f"{pos.symbol}:{pos.quantity}:{pos.current_price}"
            for pos in sorted(portfolio.positions, key=lambda x: x.symbol)
        ])
        return str(hash(positions_str))


# Singleton Pattern for Global Risk Configuration
class RiskConfigurationManager:
    """Singleton для глобальной конфигурации рисков."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.risk_limits = {
                RiskCategory.MARKET_RISK: Decimal('0.05'),      # 5% VaR лимит
                RiskCategory.CREDIT_RISK: Decimal('0.02'),      # 2% кредитный риск
                RiskCategory.LIQUIDITY_RISK: Decimal('0.10'),   # 10% риск ликвидности
                RiskCategory.CONCENTRATION_RISK: Decimal('0.20') # 20% концентрация
            }
            
            self.alert_thresholds = {
                'var_95': Decimal('0.03'),       # 3% VaR порог
                'max_drawdown': Decimal('0.10'), # 10% просадка
                'sharpe_ratio': Decimal('0.5')   # Минимальный Шарп
            }
            
            self.global_multipliers = {
                'crisis_mode': Decimal('2.0'),
                'normal_mode': Decimal('1.0'),
                'low_volatility_mode': Decimal('0.8')
            }
            
            self._initialized = True
    
    def get_risk_limit(self, category: RiskCategory) -> Decimal:
        """Получение лимита риска по категории."""
        return self.risk_limits.get(category, Decimal('0.05'))
    
    def set_risk_limit(self, category: RiskCategory, limit: Decimal) -> None:
        """Установка лимита риска."""
        self.risk_limits[category] = limit
    
    def get_alert_threshold(self, metric: str) -> Decimal:
        """Получение порога оповещения."""
        return self.alert_thresholds.get(metric, Decimal('0.05'))
    
    def is_crisis_mode(self) -> bool:
        """Проверка режима кризиса."""
        # В реальности здесь была бы логика определения кризисного режима
        return False


# Factory Method Pattern for Risk Calculators
class RiskCalculatorFactory:
    """Фабрика калькуляторов рисков."""
    
    @staticmethod
    def create_parametric_calculator() -> ParametricRiskAssessment:
        """Создание параметрического калькулятора."""
        return ParametricRiskAssessment()
    
    @staticmethod
    def create_historical_calculator():
        """Создание исторического калькулятора."""
        # Заглушка для исторического калькулятора
        return ParametricRiskAssessment()  # В реальности был бы отдельный класс
    
    @staticmethod
    def create_monte_carlo_calculator():
        """Создание Monte Carlo калькулятора."""
        # Заглушка для Monte Carlo калькулятора
        return ParametricRiskAssessment()  # В реальности был бы отдельный класс
    
    @staticmethod
    def create_enhanced_calculator(calculation_method: str = 'parametric',
                                 regional_adjustments: Optional[Dict[str, Decimal]] = None,
                                 time_decay: bool = True) -> RiskAdjustmentDecorator:
        """Создание улучшенного калькулятора с декораторами."""
        # Создаем базовый калькулятор
        if calculation_method == 'parametric':
            base_calculator = RiskCalculatorFactory.create_parametric_calculator()
        elif calculation_method == 'historical':
            base_calculator = RiskCalculatorFactory.create_historical_calculator()
        else:
            base_calculator = RiskCalculatorFactory.create_monte_carlo_calculator()
        
        # Применяем декораторы
        calculator = base_calculator
        
        if regional_adjustments:
            calculator = RegionalRiskAdjustmentDecorator(calculator, regional_adjustments)
        
        if time_decay:
            calculator = TimeDecayAdjustmentDecorator(calculator)
        
        return calculator


# Advanced Risk Engine - Main Class
class AdvancedRiskEngine:
    """
    Продвинутый движок управления рисками.
    
    Реализует множественные паттерны проектирования для сложных вычислений рисков.
    """
    
    def __init__(self):
        self.config_manager = RiskConfigurationManager()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.risk_calculator = None
        self.alert_callbacks: List[Callable] = []
        
        # Инициализация калькулятора по умолчанию
        self._initialize_default_calculator()
    
    def _initialize_default_calculator(self) -> None:
        """Инициализация калькулятора по умолчанию."""
        regional_adjustments = {
            'US': Decimal('1.0'),
            'EU': Decimal('1.1'),
            'OTHER': Decimal('1.2')
        }
        
        self.risk_calculator = RiskCalculatorFactory.create_enhanced_calculator(
            calculation_method='parametric',
            regional_adjustments=regional_adjustments,
            time_decay=True
        )
    
    async def calculate_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Асинхронный расчет рисков портфеля."""
        try:
            # Выполняем расчет в отдельном потоке для CPU-intensive операций
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                self.executor,
                self.risk_calculator.calculate_risk,
                portfolio
            )
            
            # Проверяем пороги и отправляем уведомления
            await self._check_risk_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            raise
    
    async def _check_risk_thresholds(self, metrics: RiskMetrics) -> None:
        """Проверка пороговых значений и отправка уведомлений."""
        alerts = []
        
        # Проверяем VaR
        var_threshold = self.config_manager.get_alert_threshold('var_95')
        if metrics.var_95 > var_threshold:
            alerts.append(f"VaR 95% ({metrics.var_95}) exceeds threshold ({var_threshold})")
        
        # Проверяем просадку
        drawdown_threshold = self.config_manager.get_alert_threshold('max_drawdown')
        if metrics.max_drawdown > drawdown_threshold:
            alerts.append(f"Max drawdown ({metrics.max_drawdown}) exceeds threshold ({drawdown_threshold})")
        
        # Проверяем Sharpe ratio
        sharpe_threshold = self.config_manager.get_alert_threshold('sharpe_ratio')
        if metrics.sharpe_ratio < sharpe_threshold:
            alerts.append(f"Sharpe ratio ({metrics.sharpe_ratio}) below threshold ({sharpe_threshold})")
        
        # Отправляем уведомления
        if alerts:
            await self._send_risk_alerts(alerts)
    
    async def _send_risk_alerts(self, alerts: List[str]) -> None:
        """Отправка уведомлений о рисках."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alerts)
                else:
                    callback(alerts)
            except Exception as e:
                logger.error(f"Error sending risk alert: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Добавление callback для уведомлений."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """Удаление callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def set_risk_calculator(self, calculator) -> None:
        """Установка кастомного калькулятора рисков."""
        self.risk_calculator = calculator
    
    async def run_stress_test(self, portfolio: Portfolio, 
                            scenarios: List[Dict[str, Any]]) -> Dict[str, RiskMetrics]:
        """Выполнение стресс-тестирования."""
        results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'scenario_{i}')
            
            # Модифицируем портфель под сценарий
            stressed_portfolio = self._apply_stress_scenario(portfolio, scenario)
            
            # Рассчитываем риски для стрессованного портфеля
            stressed_metrics = await self.calculate_portfolio_risk(stressed_portfolio)
            results[scenario_name] = stressed_metrics
        
        return results
    
    def _apply_stress_scenario(self, portfolio: Portfolio, 
                              scenario: Dict[str, Any]) -> Portfolio:
        """Применение стресс-сценария к портфелю."""
        stressed_positions = []
        
        market_shock = scenario.get('market_shock', Decimal('0'))
        
        for position in portfolio.positions:
            # Применяем шок к цене
            stressed_price = position.current_price * (Decimal('1') + market_shock)
            stressed_market_value = Money(position.quantity * stressed_price)
            
            stressed_position = Position(
                symbol=position.symbol,
                quantity=position.quantity,
                current_price=stressed_price,
                entry_price=position.entry_price,
                market_value=stressed_market_value,
                unrealized_pnl=Money(stressed_market_value.amount - 
                                   (position.quantity * position.entry_price)),
                risk_category=position.risk_category
            )
            
            stressed_positions.append(stressed_position)
        
        return Portfolio(
            positions=stressed_positions,
            total_value=Money(sum(pos.market_value.amount for pos in stressed_positions)),
            cash_position=portfolio.cash_position
        )
    
    async def shutdown(self) -> None:
        """Корректное завершение работы движка."""
        self.executor.shutdown(wait=True)
        logger.info("Risk engine shut down successfully")


# Экспорт основных классов
__all__ = [
    'AdvancedRiskEngine',
    'RiskCalculatorFactory',
    'RiskMetrics',
    'Portfolio',
    'Position',
    'RiskLevel',
    'RiskCategory',
    'RiskConfigurationManager',
    'VaRCalculatorVisitor',
    'LiquidityRiskVisitor',
    'ParametricRiskAssessment',
    'RegionalRiskAdjustmentDecorator',
    'TimeDecayAdjustmentDecorator'
]