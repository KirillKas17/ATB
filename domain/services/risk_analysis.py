"""
Продвинутая система анализа рисков с полной реализацией всех методов.
Включает математические модели оценки рисков, стресс-тестирование и оптимизацию портфеля.
"""
import asyncio
from shared.numpy_utils import np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import math
import statistics
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Уровни риска."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PositionRisk:
    """Риск позиции."""
    position_id: str
    symbol: str
    risk_level: RiskLevel
    var_1d: float  # Value at Risk (1 день)
    var_5d: float  # Value at Risk (5 дней)
    max_drawdown: float  # Максимальная просадка
    volatility: float  # Волатильность
    correlation_risk: float  # Корреляционный риск
    liquidity_risk: float  # Риск ликвидности
    concentration_risk: float  # Риск концентрации
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Валидация данных о риске."""
        if self.var_1d < 0 or self.var_5d < 0:
            raise ValueError("VaR values must be non-negative")
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        if not (0 <= self.correlation_risk <= 1):
            raise ValueError("Correlation risk must be between 0 and 1")
        if not (0 <= self.liquidity_risk <= 1):
            raise ValueError("Liquidity risk must be between 0 and 1")
        if not (0 <= self.concentration_risk <= 1):
            raise ValueError("Concentration risk must be between 0 and 1")


class RiskMetricType(Enum):
    """Типы метрик риска."""
    VALUE_AT_RISK = "var"
    EXPECTED_SHORTFALL = "es"
    MAX_DRAWDOWN = "max_dd"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    STRESS_TEST = "stress_test"


class RiskType(Enum):
    """Типы рисков."""
    MARKET = "market"
    CREDIT = "credit"
    OPERATIONAL = "operational"
    LIQUIDITY = "liquidity"
    CURRENCY = "currency"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"

@dataclass
class RiskLimits:
    """Лимиты рисков."""
    max_loss: float = 0.2
    max_exposure: float = 1.0
    max_concentration: float = 0.3
    max_leverage: float = 3.0

@dataclass
class RiskMetrics:
    """Метрики риска."""
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    cvar_99: float = 0.0  # Conditional VaR 99%
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'beta': self.beta,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'risk_level': self.risk_level.value
        }

@dataclass
class RiskLimit:
    """Лимит риска."""
    risk_type: RiskType
    limit_value: float
    current_value: float
    threshold_warning: float  # При достижении этого значения - предупреждение
    threshold_critical: float  # При достижении этого значения - критическое состояние
    
    @property
    def utilization_ratio(self) -> float:
        """Коэффициент использования лимита."""
        if self.limit_value == 0:
            return 0.0
        return min(1.0, abs(self.current_value) / abs(self.limit_value))
    
    @property
    def is_warning(self) -> bool:
        """Достигнуто предупреждающее значение."""
        return self.utilization_ratio >= self.threshold_warning
    
    @property
    def is_critical(self) -> bool:
        """Достигнуто критическое значение."""
        return self.utilization_ratio >= self.threshold_critical
    
    @property
    def is_breached(self) -> bool:
        """Лимит превышен."""
        return self.utilization_ratio >= 1.0

@dataclass
class PortfolioRisk:
    """Риск портфеля."""
    portfolio_value: float
    daily_var: float
    monthly_var: float
    annual_var: float
    expected_shortfall: float
    volatility: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    
    def get_total_risk_score(self) -> float:
        """Общий балл риска от 0 до 1."""
        # Нормализация и взвешивание различных компонентов риска
        var_component = min(1.0, abs(self.daily_var) / self.portfolio_value)
        vol_component = min(1.0, self.volatility)
        corr_component = min(1.0, self.correlation_risk)
        conc_component = min(1.0, self.concentration_risk)
        liq_component = min(1.0, self.liquidity_risk)
        
        # Взвешенная сумма компонентов
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]
        components = [var_component, vol_component, corr_component, conc_component, liq_component]
        
        return sum(w * c for w, c in zip(weights, components))

@dataclass
class StressTestResult:
    """Результат стресс-теста."""
    scenario_name: str
    base_portfolio_value: float
    stressed_portfolio_value: float
    loss_amount: float
    loss_percentage: float
    var_breach: bool
    recovery_time_days: Optional[int] = None
    
    @property
    def severity_level(self) -> RiskLevel:
        """Уровень серьёзности потерь."""
        loss_pct = abs(self.loss_percentage)
        if loss_pct < 0.05:
            return RiskLevel.VERY_LOW
        elif loss_pct < 0.10:
            return RiskLevel.LOW
        elif loss_pct < 0.20:
            return RiskLevel.MODERATE
        elif loss_pct < 0.35:
            return RiskLevel.HIGH
        elif loss_pct < 0.50:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME

class RiskAnalysisService(ABC):
    """Абстрактный базовый класс для сервиса анализа рисков."""
    
    @abstractmethod
    async def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRisk:
        """Расчёт риска портфеля."""
        pass
    
    @abstractmethod
    async def calculate_position_risk(self, position_data: Dict[str, Any]) -> RiskMetrics:
        """Расчёт риска позиции."""
        pass
    
    @abstractmethod
    async def validate_risk_limits(self, portfolio_data: Dict[str, Any], limits: List[RiskLimit]) -> List[RiskLimit]:
        """Валидация лимитов риска."""
        pass
    
    @abstractmethod
    async def stress_test(self, portfolio_data: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> List[StressTestResult]:
        """Стресс-тестирование портфеля."""
        pass
    
    @abstractmethod
    async def optimize_portfolio(self, assets_data: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, float]:
        """Оптимизация портфеля по критериям риска."""
        pass

class AdvancedRiskAnalysisService(RiskAnalysisService):
    """Продвинутый сервис анализа рисков с полной реализацией."""
    
    def __init__(self, confidence_level: float = 0.95, risk_free_rate: float = 0.02):
        """
        Инициализация сервиса анализа рисков.
        
        Args:
            confidence_level: Уровень доверия для VaR расчётов
            risk_free_rate: Безрисковая ставка для расчёта Sharpe ratio
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.estimation_window = 252  # Дней для оценки параметров
        
        # Кэш для ковариационных матриц
        self._covariance_cache: Dict[str, Tuple[datetime, np.ndarray]] = {}
        self._cache_ttl = timedelta(hours=1)
    
    async def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRisk:
        """
        Расчёт комплексного риска портфеля.
        
        Args:
            portfolio_data: Данные портфеля с весами позиций и историческими данными
            
        Returns:
            PortfolioRisk: Объект с метриками риска портфеля
        """
        try:
            # Извлечение данных
            weights = np.array(list(portfolio_data.get('weights', {}).values()))
            returns_data = portfolio_data.get('historical_returns', {})
            portfolio_value = portfolio_data.get('total_value', 1000000)
            
            if len(weights) == 0 or not returns_data:
                logger.warning("Недостаточно данных для расчёта риска портфеля")
                return self._create_default_portfolio_risk(portfolio_value)
            
            # Подготовка матрицы доходностей
            returns_df = pd.DataFrame(returns_data)
            if returns_df.empty:
                return self._create_default_portfolio_risk(portfolio_value)
            
            # Нормализация весов
            weights = weights / np.sum(weights) if np.sum(weights) != 0 else weights
            
            # Расчёт доходности портфеля
            portfolio_returns = returns_df.dot(weights)
            
            # Расчёт основных метрик
            daily_volatility = portfolio_returns.std()
            annual_volatility = daily_volatility * np.sqrt(252)
            
            # VaR расчёты
            daily_var_95 = self._calculate_var(portfolio_returns, 0.95) * portfolio_value
            daily_var_99 = self._calculate_var(portfolio_returns, 0.99) * portfolio_value
            monthly_var = daily_var_95 * np.sqrt(21)
            annual_var = daily_var_95 * np.sqrt(252)
            
            # Expected Shortfall (CVaR)
            expected_shortfall = self._calculate_expected_shortfall(portfolio_returns, 0.95) * portfolio_value
            
            # Корреляционный риск
            correlation_risk = await self._calculate_correlation_risk(returns_df, weights)
            
            # Риск концентрации
            concentration_risk = self._calculate_concentration_risk(weights)
            
            # Риск ликвидности (упрощённая модель)
            liquidity_risk = await self._calculate_liquidity_risk(portfolio_data)
            
            return PortfolioRisk(
                portfolio_value=portfolio_value,
                daily_var=daily_var_95,
                monthly_var=monthly_var,
                annual_var=annual_var,
                expected_shortfall=expected_shortfall,
                volatility=annual_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчёта риска портфеля: {e}")
            return self._create_default_portfolio_risk(portfolio_data.get('total_value', 1000000))
    
    async def calculate_position_risk(self, position_data: Dict[str, Any]) -> RiskMetrics:
        """
        Расчёт риска отдельной позиции.
        
        Args:
            position_data: Данные позиции включая историю цен
            
        Returns:
            RiskMetrics: Метрики риска позиции
        """
        try:
            # Извлечение данных
            price_history = position_data.get('price_history', [])
            position_size = position_data.get('size', 0)
            market_data = position_data.get('market_returns', [])
            
            if not price_history:
                logger.warning("Нет исторических данных для расчёта риска позиции")
                return RiskMetrics()
            
            # Преобразование в доходности
            prices = np.array(price_history)
            returns = np.diff(np.log(prices))
            
            if len(returns) < 30:  # Минимум данных для надёжной оценки
                return RiskMetrics(risk_level=RiskLevel.HIGH)
            
            # Основные статистики
            mean_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
            
            # VaR расчёты
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            cvar_95 = self._calculate_expected_shortfall(returns, 0.95)
            cvar_99 = self._calculate_expected_shortfall(returns, 0.99)
            
            # Sharpe ratio
            sharpe_ratio = (mean_return * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return * 252 - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Beta (если есть рыночные данные)
            beta = 0.0
            tracking_error = 0.0
            information_ratio = 0.0
            
            if market_data and len(market_data) == len(returns):
                market_returns = np.array(market_data)
                beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 0
                excess_returns = returns - market_returns
                tracking_error = np.std(excess_returns) * np.sqrt(252)
                information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
            
            # Определение уровня риска
            risk_level = self._determine_risk_level(volatility, var_95, max_drawdown)
            
            return RiskMetrics(
                var_95=var_95 * position_size,
                var_99=var_99 * position_size,
                cvar_95=cvar_95 * position_size,
                cvar_99=cvar_99 * position_size,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчёта риска позиции: {e}")
            return RiskMetrics(risk_level=RiskLevel.HIGH)
    
    async def validate_risk_limits(self, portfolio_data: Dict[str, Any], limits: List[RiskLimit]) -> List[RiskLimit]:
        """
        Валидация лимитов риска портфеля.
        
        Args:
            portfolio_data: Данные портфеля
            limits: Список лимитов для проверки
            
        Returns:
            List[RiskLimit]: Обновлённые лимиты с текущими значениями
        """
        try:
            # Расчёт текущих рисков портфеля
            portfolio_risk = await self.calculate_portfolio_risk(portfolio_data)
            portfolio_value = portfolio_data.get('total_value', 1000000)
            weights = np.array(list(portfolio_data.get('weights', {}).values()))
            
            updated_limits = []
            
            for limit in limits:
                updated_limit = RiskLimit(
                    risk_type=limit.risk_type,
                    limit_value=limit.limit_value,
                    current_value=limit.current_value,  # Будет обновлено ниже
                    threshold_warning=limit.threshold_warning,
                    threshold_critical=limit.threshold_critical
                )
                
                # Расчёт текущего значения в зависимости от типа риска
                if limit.risk_type == RiskType.MARKET:
                    updated_limit.current_value = abs(portfolio_risk.daily_var) / portfolio_value
                
                elif limit.risk_type == RiskType.VOLATILITY:
                    updated_limit.current_value = portfolio_risk.volatility
                
                elif limit.risk_type == RiskType.CONCENTRATION:
                    updated_limit.current_value = portfolio_risk.concentration_risk
                
                elif limit.risk_type == RiskType.LIQUIDITY:
                    updated_limit.current_value = portfolio_risk.liquidity_risk
                
                elif limit.risk_type == RiskType.CORRELATION:
                    updated_limit.current_value = portfolio_risk.correlation_risk
                
                else:
                    # Для других типов риска используем общий балл
                    updated_limit.current_value = portfolio_risk.get_total_risk_score()
                
                updated_limits.append(updated_limit)
                
                # Логирование нарушений
                if updated_limit.is_breached:
                    logger.critical(f"Превышен лимит риска {limit.risk_type.value}: "
                                  f"{updated_limit.current_value:.4f} > {limit.limit_value:.4f}")
                elif updated_limit.is_critical:
                    logger.error(f"Критический уровень риска {limit.risk_type.value}: "
                               f"{updated_limit.current_value:.4f} (лимит: {limit.limit_value:.4f})")
                elif updated_limit.is_warning:
                    logger.warning(f"Предупреждающий уровень риска {limit.risk_type.value}: "
                                 f"{updated_limit.current_value:.4f} (лимит: {limit.limit_value:.4f})")
            
            return updated_limits
            
        except Exception as e:
            logger.error(f"Ошибка валидации лимитов риска: {e}")
            return limits
    
    async def stress_test(self, portfolio_data: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> List[StressTestResult]:
        """
        Стресс-тестирование портфеля по различным сценариям.
        
        Args:
            portfolio_data: Данные портфеля
            scenarios: Список сценариев для тестирования
            
        Returns:
            List[StressTestResult]: Результаты стресс-тестов
        """
        try:
            results = []
            base_portfolio_value = portfolio_data.get('total_value', 1000000)
            weights = list(portfolio_data.get('weights', {}).values())
            asset_prices = portfolio_data.get('current_prices', {})
            
            if len(weights) == 0 or not asset_prices:
                logger.warning("Недостаточно данных для стресс-тестирования")
                return []
            
            # Стандартные сценарии если не предоставлены
            if not scenarios:
                scenarios = self._create_default_stress_scenarios()
            
            # Расчёт базового VaR для сравнения
            portfolio_risk = await self.calculate_portfolio_risk(portfolio_data)
            base_var = abs(portfolio_risk.daily_var)
            
            for scenario in scenarios:
                try:
                    scenario_name = scenario.get('name', 'Unnamed Scenario')
                    shock_type = scenario.get('type', 'market_crash')
                    shock_magnitude = scenario.get('magnitude', -0.20)
                    correlation_adjustment = scenario.get('correlation_increase', 0.0)
                    
                    # Применение шока к ценам активов
                    stressed_prices = self._apply_stress_shock(
                        asset_prices, shock_type, shock_magnitude, correlation_adjustment
                    )
                    
                    # Расчёт стоимости портфеля после шока
                    stressed_value = self._calculate_stressed_portfolio_value(
                        weights, asset_prices, stressed_prices
                    )
                    
                    loss_amount = base_portfolio_value - stressed_value
                    loss_percentage = loss_amount / base_portfolio_value if base_portfolio_value > 0 else 0
                    
                    # Проверка превышения VaR
                    var_breach = abs(loss_amount) > base_var
                    
                    # Оценка времени восстановления (упрощённая модель)
                    recovery_time = self._estimate_recovery_time(loss_percentage, scenario)
                    
                    result = StressTestResult(
                        scenario_name=scenario_name,
                        base_portfolio_value=base_portfolio_value,
                        stressed_portfolio_value=stressed_value,
                        loss_amount=loss_amount,
                        loss_percentage=loss_percentage,
                        var_breach=var_breach,
                        recovery_time_days=recovery_time
                    )
                    
                    results.append(result)
                    
                    logger.info(f"Стресс-тест '{scenario_name}': потери {loss_percentage:.2%}, "
                              f"время восстановления {recovery_time} дней")
                    
                except Exception as e:
                    logger.error(f"Ошибка в стресс-тесте '{scenario.get('name', 'Unknown')}': {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка стресс-тестирования: {e}")
            return []
    
    async def optimize_portfolio(self, assets_data: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Оптимизация портфеля по критериям минимизации риска.
        
        Args:
            assets_data: Данные по активам включая доходности
            constraints: Ограничения оптимизации
            
        Returns:
            Dict[str, float]: Оптимальные веса активов
        """
        try:
            # Извлечение данных
            returns_data = assets_data.get('historical_returns', {})
            min_weights = constraints.get('min_weights', {})
            max_weights = constraints.get('max_weights', {})
            target_return = constraints.get('target_return', None)
            max_risk = constraints.get('max_risk', None)
            
            if not returns_data:
                logger.warning("Нет данных о доходностях для оптимизации")
                return {}
            
            # Подготовка данных
            returns_df = pd.DataFrame(returns_data)
            asset_names = returns_df.columns.tolist()
            n_assets = len(asset_names)
            
            if n_assets == 0:
                return {}
            
            # Расчёт ковариационной матрицы (с регуляризацией Ledoit-Wolf)
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df.fillna(0)).covariance_
            
            # Средние доходности
            mean_returns = returns_df.mean().values
            
            # Целевая функция: минимизация риска (дисперсии портфеля)
            def objective(weights: np.ndarray) -> float:
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return float(portfolio_variance)  # Explicit cast to float
            
            # Ограничения
            constraints_list = []
            
            # Сумма весов = 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Целевая доходность (если задана)
            if target_return is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda w: np.dot(w, mean_returns) * 252 - target_return
                })
            
            # Максимальный риск (если задан)
            if max_risk is not None:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_risk**2 - np.dot(w, np.dot(cov_matrix, w)) * 252
                })
            
            # Границы весов
            bounds = []
            for i, asset in enumerate(asset_names):
                min_weight = min_weights.get(asset, 0.0)
                max_weight = max_weights.get(asset, 1.0)
                bounds.append((min_weight, max_weight))
            
            # Начальное приближение (равные веса)
            initial_guess = np.ones(n_assets) / n_assets
            
            # Оптимизация
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = {
                    asset: float(weight) 
                    for asset, weight in zip(asset_names, result.x)
                    if abs(weight) > 1e-6  # Исключаем очень маленькие веса
                }
                
                # Валидация результата
                total_weight = sum(optimal_weights.values())
                if abs(total_weight - 1.0) > 1e-3:
                    logger.warning(f"Сумма весов не равна 1: {total_weight}")
                    # Нормализация
                    optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}
                
                logger.info(f"Оптимизация портфеля завершена успешно. Риск: {np.sqrt(result.fun * 252):.4f}")
                return optimal_weights
            
            else:
                logger.error(f"Оптимизация не сошлась: {result.message}")
                # Возврат равновесного портфеля
                return {asset: 1.0/n_assets for asset in asset_names}
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации портфеля: {e}")
            return {}
    
    def detect_regime_change(self, returns_data: np.ndarray, window_size: int = 50) -> Dict[str, Any]:
        """
        Обнаружение изменения рыночного режима.
        
        Args:
            returns_data: Временной ряд доходностей
            window_size: Размер окна для анализа
            
        Returns:
            Dict[str, Any]: Информация об изменении режима
        """
        try:
            if len(returns_data) < 2 * window_size:
                return {'regime_change': False, 'confidence': 0.0}
            
            # Разделение на два окна
            recent_window = returns_data[-window_size:]
            previous_window = returns_data[-2*window_size:-window_size]
            
            # Статистический тест на изменение дисперсии (тест Левена)
            stat, p_value = stats.levene(recent_window, previous_window)
            
            # Тест на изменение среднего (t-тест)
            t_stat, t_p_value = stats.ttest_ind(recent_window, previous_window)
            
            # Объединённая оценка изменения режима
            variance_change = p_value < 0.05
            mean_change = t_p_value < 0.05
            
            regime_change = variance_change or mean_change
            confidence = 1.0 - min(p_value, t_p_value)
            
            return {
                'regime_change': regime_change,
                'confidence': confidence,
                'variance_change': variance_change,
                'mean_change': mean_change,
                'test_statistics': {
                    'levene_stat': stat,
                    'levene_p_value': p_value,
                    't_stat': t_stat,
                    't_p_value': t_p_value
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения изменения режима: {e}")
            return {'regime_change': False, 'confidence': 0.0}
    
    def forecast_risk_metrics(
        self, 
        returns_data: np.ndarray, 
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:  # Changed from Dict[str, float] to allow mixed types
        """
        Прогнозирование метрик риска на основе исторических данных.
        
        Args:
            returns_data: Исторические доходности
            forecast_horizon: Горизонт прогнозирования в днях
            
        Returns:
            Dict[str, Any]: Прогнозные метрики риска
        """
        try:
            if len(returns_data) < 30:
                logger.warning("Недостаточно данных для прогнозирования риска")
                return {
                    'forecasted_volatility': 0.0, 
                    'confidence_interval_lower': 0.0,
                    'confidence_interval_upper': 0.0,
                    'forecast_horizon_days': forecast_horizon,
                    'model': 'EWMA'
                }
            
            # EWMA модель для прогнозирования волатильности
            lambda_param = 0.94  # Параметр затухания для EWMA
            
            # Расчёт EWMA волатильности
            squared_returns = returns_data ** 2
            ewma_variance = np.zeros(len(squared_returns))
            ewma_variance[0] = squared_returns[0]
            
            for i in range(1, len(squared_returns)):
                ewma_variance[i] = lambda_param * ewma_variance[i-1] + (1 - lambda_param) * squared_returns[i]
            
            # Прогноз волатильности
            current_variance = ewma_variance[-1]
            forecasted_variance = current_variance  # Упрощённый прогноз (случайное блуждание для волатильности)
            forecasted_volatility = np.sqrt(forecasted_variance * forecast_horizon)
            
            # Доверительный интервал (упрощённый)
            volatility_std = np.std(np.sqrt(ewma_variance))
            confidence_lower = max(0, forecasted_volatility - 1.96 * volatility_std)
            confidence_upper = forecasted_volatility + 1.96 * volatility_std
            
            return {
                'forecasted_volatility': float(forecasted_volatility),
                'confidence_interval_lower': float(confidence_lower),
                'confidence_interval_upper': float(confidence_upper),
                'forecast_horizon_days': int(forecast_horizon),
                'model': str('EWMA')
            }
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования риска: {e}")
            return {
                'forecasted_volatility': 0.0, 
                'confidence_interval_lower': 0.0,
                'confidence_interval_upper': 0.0,
                'forecast_horizon_days': int(forecast_horizon),
                'model': 'EWMA'
            }
    
    # Вспомогательные методы
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Расчёт Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Расчёт Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    async def _calculate_correlation_risk(self, returns_df: pd.DataFrame, weights: np.ndarray) -> float:
        """Расчёт корреляционного риска."""
        try:
            if returns_df.empty or len(weights) != len(returns_df.columns):
                return 0.0
            
            # Расчёт корреляционной матрицы
            correlation_matrix = returns_df.corr()
            
            # Средняя корреляция взвешенная по портфелю
            weighted_correlation = 0.0
            total_weight_pairs = 0.0
            
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    correlation = correlation_matrix.iloc[i, j]
                    if not np.isnan(correlation):
                        weight_product = weights[i] * weights[j]
                        weighted_correlation += abs(correlation) * weight_product
                        total_weight_pairs += weight_product
            
            if total_weight_pairs > 0:
                weighted_correlation /= total_weight_pairs
            
            return min(1.0, weighted_correlation)
            
        except Exception as e:
            logger.error(f"Ошибка расчёта корреляционного риска: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Расчёт риска концентрации по индексу Херфиндаля."""
        if len(weights) == 0:
            return 0.0
        
        # Нормализация весов
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        # Индекс Херфиндаля-Хиршмана
        hhi = np.sum(normalized_weights ** 2)
        
        # Преобразование в риск концентрации (0 = равномерное распределение, 1 = полная концентрация)
        n = len(weights)
        min_hhi = 1.0 / n  # Минимальный HHI при равномерном распределении
        concentration_risk = (hhi - min_hhi) / (1.0 - min_hhi) if n > 1 else 0.0
        
        return min(1.0, max(0.0, concentration_risk))
    
    async def _calculate_liquidity_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Расчёт риска ликвидности (упрощённая модель)."""
        try:
            liquidity_scores = portfolio_data.get('liquidity_scores', {})
            weights = portfolio_data.get('weights', {})
            
            if not liquidity_scores or not weights:
                return 0.5  # Средний уровень при отсутствии данных
            
            # Взвешенный балл ликвидности
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for asset, weight in weights.items():
                if asset in liquidity_scores:
                    liquidity_score = liquidity_scores[asset]  # 0 = неликвидный, 1 = высоколиквидный
                    total_weighted_score += (1.0 - liquidity_score) * weight  # Инвертируем для риска
                    total_weight += weight
            
            if total_weight > 0:
                return total_weighted_score / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Ошибка расчёта риска ликвидности: {e}")
            return 0.5
    
    def _determine_risk_level(self, volatility: float, var_95: float, max_drawdown: float) -> RiskLevel:
        """Определение общего уровня риска."""
        # Нормализация метрик
        vol_score = min(1.0, volatility / 0.5)  # Волатильность > 50% = высокий риск
        var_score = min(1.0, abs(var_95) / 0.1)  # VaR > 10% = высокий риск
        dd_score = min(1.0, abs(max_drawdown) / 0.3)  # Drawdown > 30% = высокий риск
        
        # Общий балл риска
        risk_score = (vol_score + var_score + dd_score) / 3
        
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        elif risk_score < 0.95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def _create_default_portfolio_risk(self, portfolio_value: float) -> PortfolioRisk:
        """Создание объекта PortfolioRisk с значениями по умолчанию."""
        return PortfolioRisk(
            portfolio_value=portfolio_value,
            daily_var=portfolio_value * 0.02,  # 2% дневной VaR
            monthly_var=portfolio_value * 0.08,  # 8% месячный VaR
            annual_var=portfolio_value * 0.20,  # 20% годовой VaR
            expected_shortfall=portfolio_value * 0.025,
            volatility=0.15,  # 15% годовая волатильность
            correlation_risk=0.3,
            concentration_risk=0.2,
            liquidity_risk=0.1
        )
    
    def _create_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Создание стандартных сценариев стресс-тестирования."""
        return [
            {
                'name': 'Market Crash -20%',
                'type': 'market_crash',
                'magnitude': -0.20,
                'correlation_increase': 0.3
            },
            {
                'name': 'Market Crash -35%',
                'type': 'market_crash',
                'magnitude': -0.35,
                'correlation_increase': 0.5
            },
            {
                'name': 'Volatility Spike',
                'type': 'volatility_spike',
                'magnitude': 2.0,
                'correlation_increase': 0.2
            },
            {
                'name': 'Interest Rate Shock +3%',
                'type': 'interest_rate_shock',
                'magnitude': 0.03,
                'correlation_increase': 0.1
            },
            {
                'name': 'Liquidity Crisis',
                'type': 'liquidity_crisis',
                'magnitude': -0.15,
                'correlation_increase': 0.7
            }
        ]
    
    def _apply_stress_shock(
        self, 
        asset_prices: Dict[str, float], 
        shock_type: str, 
        magnitude: float,
        correlation_adjustment: float
    ) -> Dict[str, float]:
        """Применение стресс-шока к ценам активов."""
        stressed_prices = {}
        
        for asset, price in asset_prices.items():
            if shock_type == 'market_crash':
                # Все активы падают с учётом корреляций
                base_shock = magnitude
                correlation_factor = 1.0 + correlation_adjustment * np.random.uniform(-0.5, 0.5)
                shock = base_shock * correlation_factor
                stressed_prices[asset] = price * (1 + shock)
            
            elif shock_type == 'volatility_spike':
                # Рост волатильности моделируется случайными скачками
                volatility_shock = np.random.normal(0, magnitude * 0.05)  # 5% дополнительная волатильность
                stressed_prices[asset] = price * (1 + volatility_shock)
            
            elif shock_type == 'interest_rate_shock':
                # Влияние изменения процентных ставок (больше на облигации, меньше на акции)
                duration_proxy = np.random.uniform(2, 10)  # Упрощённая дюрация
                price_change = -duration_proxy * magnitude * 0.01  # Конвертация в изменение цены
                stressed_prices[asset] = price * (1 + price_change)
            
            elif shock_type == 'liquidity_crisis':
                # Снижение ликвидности ведёт к увеличению спредов и падению цен
                liquidity_impact = magnitude * np.random.uniform(0.7, 1.3)
                stressed_prices[asset] = price * (1 + liquidity_impact)
            
            else:
                # По умолчанию применяем базовый шок
                stressed_prices[asset] = price * (1 + magnitude)
        
        return stressed_prices
    
    def _calculate_stressed_portfolio_value(
        self, 
        weights: List[float], 
        base_prices: Dict[str, float], 
        stressed_prices: Dict[str, float]
    ) -> float:
        """Расчёт стоимости портфеля после стресс-теста."""
        assets = list(base_prices.keys())
        base_value = sum(base_prices[asset] * weight for asset, weight in zip(assets, weights))
        stressed_value = sum(stressed_prices.get(asset, base_prices[asset]) * weight 
                           for asset, weight in zip(assets, weights))
        return float(stressed_value)  # Explicit cast to float
    
    def _estimate_recovery_time(self, loss_percentage: float, scenario: Dict[str, Any]) -> Optional[int]:
        """Оценка времени восстановления после потерь."""
        loss_magnitude = abs(loss_percentage)
        
        # Базовое время восстановления в зависимости от размера потерь
        if loss_magnitude < 0.05:
            base_recovery = 30  # 1 месяц
        elif loss_magnitude < 0.10:
            base_recovery = 90  # 3 месяца
        elif loss_magnitude < 0.20:
            base_recovery = 180  # 6 месяцев
        elif loss_magnitude < 0.35:
            base_recovery = 365  # 1 год
        elif loss_magnitude < 0.50:
            base_recovery = 730  # 2 года
        else:
            base_recovery = 1095  # 3 года
        
        # Корректировка в зависимости от типа шока
        shock_type = scenario.get('type', 'market_crash')
        if shock_type == 'liquidity_crisis':
            base_recovery = int(base_recovery * 1.5)  # Дольше восстановление при проблемах ликвидности
        elif shock_type == 'volatility_spike':
            base_recovery = int(base_recovery * 0.7)  # Быстрее при волатильности
        
        return base_recovery

class DefaultRiskAnalysisService(AdvancedRiskAnalysisService):
    """Дефолтная реализация сервиса анализа рисков."""
    
    def __init__(self) -> None:
        super().__init__()
        logger.info("Initialized DefaultRiskAnalysisService")
    
    async def initialize_async(self) -> None:
        """Асинхронная инициализация сервиса."""
        # Здесь можно добавить асинхронную инициализацию при необходимости
        pass

# Создание глобального экземпляра сервиса
risk_analysis_service = AdvancedRiskAnalysisService()
