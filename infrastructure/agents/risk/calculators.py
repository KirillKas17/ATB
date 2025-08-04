# -*- coding: utf-8 -*-
"""Калькуляторы рисков для risk agent."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats

from .types import RiskConfig, RiskLevel, RiskLimits, RiskMetrics


class DefaultRiskCalculator:
    """Базовый калькулятор рисков."""

    def __init__(self) -> None:
        """Инициализация калькулятора рисков."""
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}

    async def initialize(self) -> None:
        """Инициализация калькулятора рисков."""
        logger.info("DefaultRiskCalculator initialized")

    async def calculate_risk_metrics(
        self, market_data: pd.DataFrame, positions: Dict[str, Any]
    ) -> RiskMetrics:
        """
        Расчет метрик риска.
        Args:
            market_data: Рыночные данные
            positions: Словарь позиций
        Returns:
            Метрики риска
        """
        try:
            if market_data.empty:
                return RiskMetrics()
            # Расчет доходностей
            returns = market_data["close"].pct_change().dropna()
            if len(returns) < 20:
                return RiskMetrics()
            # Расчет основных метрик риска
            var_95 = await self.calculate_var(returns, 0.95)
            var_99 = await self.calculate_var(returns, 0.99)
            max_dd = await self.calculate_max_drawdown(market_data["close"])
            kelly = await self.calculate_kelly_criterion(returns)
            volatility = await self.calculate_volatility(returns)
            # Расчет дополнительных метрик
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_dd)
            # Расчет портфельных метрик
            exposure_level = self._calculate_exposure_level(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            correlation_risk = self._calculate_correlation_risk(positions)
            liquidity_risk = self._calculate_liquidity_risk(market_data)
            # Расчет итогового скора
            confidence_score = self._calculate_confidence_score(
                var_95, volatility, max_dd, len(returns)
            )
            
            # Создаем детализированные метрики
            details = {
                "var_95": var_95,
                "var_99": var_99,
                "max_drawdown": max_dd,
                "kelly_criterion": kelly,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "exposure_level": exposure_level,
                "concentration_risk": concentration_risk,
                "correlation_risk": correlation_risk,
                "liquidity_risk": liquidity_risk,
                "confidence_score": confidence_score,
                "timestamp": datetime.now(),
            }
            
            return RiskMetrics(value=confidence_score, details=details)
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics()

    async def calculate_var(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """
        Расчет Value at Risk.
        Args:
            returns: Временной ряд доходностей
            confidence: Уровень доверия
        Returns:
            VaR
        """
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            # Расчет VaR
            mean_return_result = returns.mean()
            std_return_result = returns.std()
            
            # Проверяем типы и приводим к float
            if hasattr(mean_return_result, '__float__'):
                mean_return = float(mean_return_result)
            else:
                mean_return = 0.0
            if hasattr(std_return_result, '__float__'):
                std_return = float(std_return_result)
            else:
                std_return = 0.0
                
            z_score = stats.norm.ppf(confidence)
            # VaR
            var = abs(mean_return - z_score * std_return)
            if hasattr(var, '__float__'):
                var_float = float(var)
            else:
                var_float = float(str(var))
            return min(1.0, max(0.0, var_float))
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    async def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Расчет максимальной просадки.
        Args:
            equity_curve: Кривая доходности
        Returns:
            Максимальная просадка
        """
        try:
            if equity_curve.empty or len(equity_curve) < 2:
                return 0.0
            # Расчет кумулятивных доходностей
            pct_change = equity_curve.pct_change()
            # Преобразуем в numpy массив для унификации
            pct_array = pct_change.to_numpy() if hasattr(pct_change, 'to_numpy') else np.asarray(pct_change)
            cumulative_returns = np.cumprod(1 + pct_array)
            # Расчет просадки
            if hasattr(cumulative_returns, 'expanding'):
                rolling_max = cumulative_returns.expanding().max()
            else:
                rolling_max = cumulative_returns.apply(lambda x: max(cumulative_returns[:x.name+1]))
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            # Максимальная просадка
            drawdown_min_result = drawdown.min()
            if hasattr(drawdown_min_result, '__float__'):
                max_dd = abs(float(drawdown_min_result))
            else:
                max_dd = abs(float(drawdown_min_result))
            return min(1.0, max(0.0, max_dd))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    async def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """
        Расчет критерия Келли.
        Args:
            returns: Временной ряд доходностей
        Returns:
            Критерий Келли
        """
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            # Разделяем доходности на положительные и отрицательные
            # Преобразуем в numpy массив для унификации
            returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
            positive_returns = returns_array[returns_array > 0.0]
            negative_returns = returns_array[returns_array < 0.0]
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return 0.0
            # Средние доходности
            if hasattr(positive_returns, 'mean') and hasattr(negative_returns, 'mean'):
                avg_win = float(positive_returns.mean())
                avg_loss = abs(float(negative_returns.mean()))
            else:
                avg_win = float(sum(positive_returns) / len(positive_returns))
                avg_loss = abs(float(sum(negative_returns) / len(negative_returns)))
            # Вероятности
            win_prob = len(positive_returns) / len(returns)
            loss_prob = len(negative_returns) / len(returns)
            # Критерий Келли
            if avg_loss == 0:
                return 0.0
            kelly = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
            return max(-1.0, min(1.0, kelly))
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return 0.0

    async def calculate_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """
        Расчет волатильности.
        Args:
            returns: Временной ряд доходностей
            window: Размер окна
        Returns:
            Волатильность
        """
        try:
            if returns.empty or len(returns) < window:
                return 0.0
            # Годовая волатильность (предполагаем 252 торговых дня)
            volatility = float(returns.rolling(window=window).std().iloc[-1] * np.sqrt(252))
            return min(1.0, max(0.0, volatility))
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Расчет коэффициента Шарпа."""
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            excess_returns = returns - risk_free_rate / 252
            if float(excess_returns.std()) == 0:
                return 0.0
            sharpe = float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            return max(-10.0, min(10.0, sharpe))
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Расчет коэффициента Сортино."""
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            excess_returns = returns - risk_free_rate / 252
            # Преобразуем в numpy массив для унификации
            excess_array = excess_returns.to_numpy() if hasattr(excess_returns, 'to_numpy') else np.asarray(excess_returns)
            downside_returns = excess_array[excess_array < 0]
            if len(downside_returns) == 0:
                return 0.0
            # Проверяем std для downside_returns
            if hasattr(downside_returns, 'std'):
                downside_std = float(downside_returns.std())
            else:
                downside_std = float(np.std(downside_returns))
            if downside_std == 0:
                return 0.0
            # Вычисляем среднее значение
            excess_mean = float(np.mean(excess_array))
            sortino = float(excess_mean / downside_std * np.sqrt(252))
            return max(-10.0, min(10.0, sortino))
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Расчет коэффициента Кальмара."""
        try:
            if returns.empty or max_drawdown == 0:
                return 0.0
            annual_return = float(returns.mean() * 252)
            calmar = annual_return / max_drawdown
            return max(-10.0, min(10.0, calmar))
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def _calculate_exposure_level(self, positions: Dict[str, Any]) -> float:
        """Расчет уровня экспозиции."""
        try:
            if not positions:
                return 0.0
            total_value = sum(
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in positions.values()
            )
            return min(1.0, total_value)
        except Exception as e:
            logger.error(f"Error calculating exposure level: {e}")
            return 0.0

    def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """Расчет риска концентрации."""
        try:
            if not positions:
                return 0.0
            total_value = sum(
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in positions.values()
            )
            if total_value == 0:
                return 0.0
            # Индекс Херфиндаля-Хиршмана
            hhi = sum(
                (abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) / total_value) ** 2
                for pos in positions.values()
            )
            return min(1.0, hhi)
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Расчет риска корреляции."""
        try:
            if len(positions) < 2:
                return 0.0
            # Простая оценка корреляции на основе размера позиций
            position_sizes = [
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in positions.values()
            ]
            if not position_sizes or sum(position_sizes) == 0:
                return 0.0
            # Нормализуем размеры позиций
            normalized_sizes = [size / sum(position_sizes) for size in position_sizes]
            # Оценка концентрации как мера корреляционного риска
            concentration = sum(size ** 2 for size in normalized_sizes)
            return min(1.0, concentration)
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0

    def _calculate_liquidity_risk(self, market_data: pd.DataFrame) -> float:
        """Расчет риска ликвидности."""
        try:
            if market_data.empty:
                return 0.0
            # Простая оценка на основе объема торгов
            if "volume" in market_data.columns:
                avg_volume = float(market_data["volume"].mean())
                current_volume = float(market_data["volume"].iloc[-1])
                if avg_volume > 0:
                    liquidity_ratio = current_volume / avg_volume
                    return max(0.0, min(1.0, 1.0 - liquidity_ratio))
            return 0.5  # Значение по умолчанию
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.5

    def _calculate_confidence_score(
        self, var_95: float, volatility: float, max_dd: float, data_points: int
    ) -> float:
        """Расчет скора уверенности."""
        try:
            # Базовый скор на основе количества данных
            base_score = min(1.0, data_points / 1000)
            
            # Корректировка на основе рисков
            risk_penalty = (var_95 + volatility + max_dd) / 3
            confidence_score = base_score * (1 - risk_penalty)
            
            return max(0.0, min(1.0, confidence_score))
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0


class RiskMetricsCalculator:
    """Расширенный калькулятор метрик риска."""

    def __init__(self) -> None:
        """Инициализация калькулятора."""
        pass

    async def initialize(self) -> None:
        """Инициализация калькулятора."""
        pass

    async def calculate_stress_test(
        self, portfolio: Dict[str, Any], stress_scenarios: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Расчет стресс-тестов."""
        try:
            results = {}
            for scenario in stress_scenarios:
                scenario_name = scenario.get("name", "unknown")
                stress_factor = scenario.get("factor", 1.0)
                
                # Простая оценка влияния стресса
                total_value = sum(
                    abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                    for pos in portfolio.values()
                )
                stressed_value = total_value * stress_factor
                results[scenario_name] = stressed_value
                
            return results
        except Exception as e:
            logger.error(f"Error calculating stress test: {e}")
            return {}

    async def calculate_monte_carlo_var(
        self, returns: pd.Series, confidence: float = 0.95, simulations: int = 10000
    ) -> float:
        """Расчет VaR методом Монте-Карло."""
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            
            # Параметры распределения
            mean_return = float(returns.mean())
            std_return = float(returns.std())
            
            # Генерация случайных доходностей
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            
            # Расчет VaR
            if callable(simulated_returns):
                simulated_returns_array = simulated_returns()
            else:
                simulated_returns_array = simulated_returns
            var = np.percentile(simulated_returns_array, (1 - confidence) * 100)
            return abs(float(var))
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0

    async def calculate_expected_shortfall(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Расчет Expected Shortfall."""
        try:
            if returns.empty or len(returns) < 20:
                return 0.0
            # Получаем значения returns
            if hasattr(returns, 'values'):
                returns_array: np.ndarray = returns.values
            else:
                returns_array: np.ndarray = np.asarray(returns)
            # Расчет VaR
            var = np.percentile(returns_array, (1 - confidence) * 100)
            # Расчет ES как среднее значений ниже VaR
            tail_returns = returns_array[returns_array <= float(var)]
            if len(tail_returns) == 0:
                return abs(var)
            es = float(np.mean(tail_returns))
            return abs(es)
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0

    async def calculate_tail_dependence(
        self, returns1: pd.Series, returns2: pd.Series
    ) -> float:
        """Расчет хвостовой зависимости."""
        try:
            if returns1.empty or returns2.empty or len(returns1) != len(returns2):
                return 0.0
            # Простая оценка хвостовой зависимости
            threshold = 0.05
            returns1_array = returns1.to_numpy() if hasattr(returns1, 'to_numpy') else np.asarray(returns1)
            returns2_array = returns2.to_numpy() if hasattr(returns2, 'to_numpy') else np.asarray(returns2)
            tail1 = returns1_array < np.percentile(returns1_array, threshold * 100)
            tail2 = returns2_array < np.percentile(returns2_array, threshold * 100)
            # Коэффициент хвостовой зависимости
            tail_dependence = float((tail1 & tail2).sum() / tail1.sum() if tail1.sum() > 0 else 0.0)
            return min(1.0, tail_dependence)
        except Exception as e:
            logger.error(f"Error calculating tail dependence: {e}")
            return 0.0


class RiskValidator:
    """Валидатор рисков."""

    def __init__(self, config: RiskConfig) -> None:
        """Инициализация валидатора."""
        self.config = config

    async def validate_position(
        self, position: Dict[str, Any], portfolio: Dict[str, Any], config: RiskConfig
    ) -> bool:
        """
        Валидация позиции.
        Args:
            position: Позиция для валидации
            portfolio: Портфель
            config: Конфигурация рисков
        Returns:
            True если позиция валидна
        """
        try:
            if not position:
                return False
            # Проверка размера позиции
            position_size = abs(float(position.get("size", 0)))
            if position_size > config.threshold:
                logger.warning(f"Position size {position_size} exceeds threshold {config.threshold}")
                return False
            # Проверка лимитов портфеля
            total_exposure = sum(
                abs(float(pos.get("size", 0))) for pos in portfolio.values()
            )
            if total_exposure + position_size > config.threshold * 10:
                logger.warning(f"Total exposure {total_exposure + position_size} exceeds limit")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False

    async def validate_portfolio(
        self, portfolio: Dict[str, Any], config: RiskConfig
    ) -> List[Dict[str, Any]]:
        """
        Валидация портфеля.
        Args:
            portfolio: Портфель для валидации
            config: Конфигурация рисков
        Returns:
            Список алертов о рисках
        """
        try:
            alerts: List[Dict[str, Any]] = []
            if not portfolio:
                return alerts
            # Расчет общего риска
            total_value = sum(
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in portfolio.values()
            )
            # Проверка лимитов
            if total_value > config.threshold:
                alerts.append({
                    "type": "exposure_limit",
                    "message": f"Portfolio exposure {total_value:.2%} exceeds limit {config.threshold:.2%}",
                    "level": "critical"
                })
            # Проверка количества позиций
            if len(portfolio) > 20:  # Максимум 20 позиций
                alerts.append({
                    "type": "position_count",
                    "message": f"Number of positions {len(portfolio)} exceeds limit 20",
                    "level": "warning"
                })
            # Проверка концентрации
            for symbol, position in portfolio.items():
                position_value = abs(float(position.get("size", 0))) * float(position.get("current_price", 0))
                if total_value > 0 and position_value / total_value > 0.3:
                    alerts.append({
                        "type": "concentration",
                        "message": f"High concentration in {symbol}: {position_value / total_value:.2%}",
                        "level": "warning",
                        "symbol": symbol
                    })
            return alerts
        except Exception as e:
            logger.error(f"Error validating portfolio: {e}")
            return []

    async def check_risk_limits(
        self, portfolio: Dict[str, Any], limits: RiskLimits
    ) -> List[Dict[str, Any]]:
        """
        Проверка лимитов риска.
        Args:
            portfolio: Портфель для проверки
            limits: Лимиты риска
        Returns:
            Список нарушений лимитов
        """
        try:
            alerts: List[Dict[str, Any]] = []
            if not portfolio:
                return alerts
            # Проверка лимитов позиций
            for symbol, position in portfolio.items():
                position_size = abs(float(position.get("size", 0)))
                if position_size > limits.max_loss:
                    alerts.append({
                        "type": "position_limit",
                        "message": f"Position size {position_size} for {symbol} exceeds limit {limits.max_loss}",
                        "level": "critical",
                        "symbol": symbol
                    })
                leverage = float(position.get("leverage", 1.0))
                if leverage > limits.max_loss:
                    alerts.append({
                        "type": "leverage_limit",
                        "message": f"Leverage {leverage} for {symbol} exceeds limit {limits.max_loss}",
                        "level": "critical",
                        "symbol": symbol
                    })
            # Проверка общего риска портфеля
            total_value = sum(
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in portfolio.values()
            )
            if total_value > limits.max_loss:
                alerts.append({
                    "type": "portfolio_limit",
                    "message": f"Portfolio risk {total_value:.2%} exceeds limit {limits.max_loss:.2%}",
                    "level": "critical"
                })
            return alerts
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return []


class RiskRecommender:
    """Рекомендатель по рискам."""

    def __init__(self, config: RiskConfig) -> None:
        """Инициализация рекомендателя."""
        self.config = config

    async def generate_recommendations(
        self,
        portfolio: Dict[str, Any],
        risk_metrics: RiskMetrics,
        config: RiskConfig,
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций.
        Args:
            portfolio: Портфель
            risk_metrics: Метрики риска
            config: Конфигурация
        Returns:
            Список рекомендаций
        """
        try:
            recommendations = []
            
            # Анализ метрик риска
            if risk_metrics.value < 0.3:
                recommendations.append({
                    "type": "risk_reduction",
                    "message": "Consider reducing portfolio risk due to low confidence score",
                    "priority": "high",
                    "action": "reduce_exposure"
                })
            
            # Анализ концентрации
            if risk_metrics.details and risk_metrics.details.get("concentration_risk", 0) > 0.5:
                recommendations.append({
                    "type": "diversification",
                    "message": "High concentration risk detected - consider diversifying portfolio",
                    "priority": "medium",
                    "action": "diversify"
                })
            
            # Анализ волатильности
            if risk_metrics.details and risk_metrics.details.get("volatility", 0) > 0.3:
                recommendations.append({
                    "type": "volatility_management",
                    "message": "High volatility detected - consider hedging strategies",
                    "priority": "medium",
                    "action": "hedge"
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def recommend_position_size(
        self,
        symbol: str,
        confidence: float,
        portfolio: Dict[str, Any],
        config: RiskConfig,
    ) -> float:
        """
        Рекомендация размера позиции.
        Args:
            symbol: Символ
            confidence: Уровень уверенности
            portfolio: Портфель
            config: Конфигурация
        Returns:
            Рекомендуемый размер позиции
        """
        try:
            # Базовый размер на основе уверенности
            base_size = confidence * config.threshold
            
            # Учет текущего риска портфеля
            current_exposure = sum(
                abs(float(pos.get("size", 0))) * float(pos.get("current_price", 0)) 
                for pos in portfolio.values()
            )
            
            # Уменьшаем размер при высоком риске
            if current_exposure > config.threshold * 5:
                base_size *= 0.5
            
            # Учитываем количество позиций
            if len(portfolio) > 10:
                base_size *= 0.8
            
            return max(0.0, min(config.threshold, base_size))
        except Exception as e:
            logger.error(f"Error recommending position size: {e}")
            return 0.0

    async def recommend_stop_loss(
        self, entry_price: float, symbol: str, volatility: float, config: RiskConfig
    ) -> float:
        """
        Рекомендация стоп-лосса.
        Args:
            entry_price: Цена входа
            symbol: Символ
            volatility: Волатильность
            config: Конфигурация
        Returns:
            Рекомендуемый стоп-лосс
        """
        try:
            # Базовый стоп-лосс на основе волатильности
            stop_distance = volatility * 2  # 2 стандартных отклонения
            
            # Учитываем конфигурацию рисков
            max_loss = config.threshold
            stop_distance = min(stop_distance, max_loss)
            
            # Расчет стоп-лосса
            stop_loss = entry_price * (1 - stop_distance)
            
            return max(0.0, stop_loss)
        except Exception as e:
            logger.error(f"Error recommending stop loss: {e}")
            return entry_price * 0.95  # 5% стоп-лосс по умолчанию
