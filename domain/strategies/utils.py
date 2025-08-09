"""
Утилиты для стратегий торговли.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.numpy_utils import np
from shared.decimal_utils import TradingDecimal, to_trading_decimal

from domain.type_definitions.risk_types import RiskLevel

logger = logging.getLogger(__name__)


class StrategyUtils:
    """Утилиты для работы со стратегиями."""

    @staticmethod
    def validate_strategy_name(name: Any) -> bool:
        """Валидировать название стратегии."""
        if not isinstance(name, str):
            return False
        if len(name) < 3 or len(name) > 50:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))

    @staticmethod
    def normalize_strategy_name(name: str) -> str:
        """Нормализовать название стратегии."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())

    @staticmethod
    def generate_strategy_id(name: str) -> str:
        """Генерировать уникальный ID стратегии."""
        normalized_name = StrategyUtils.normalize_strategy_name(name)
        unique_id = str(uuid.uuid4())[:8]
        return f"{normalized_name}_{unique_id}"

    @staticmethod
    def validate_parameters(parameters: Any) -> List[str]:
        """Валидировать параметры стратегии."""
        errors = []
        if not isinstance(parameters, dict):
            errors.append("Parameters must be a dictionary")
            return errors
        
        # Проверяем обязательные поля
        required_fields = ["confidence_threshold", "risk_level", "max_position_size"]
        for field in required_fields:
            if field not in parameters:
                errors.append(f"Missing required field: {field}")
            
            # Проверяем confidence_threshold
            if "confidence_threshold" in parameters:
                try:
                    conf = Decimal(str(parameters["confidence_threshold"]))
                    if conf < 0 or conf > 1:
                        errors.append("confidence_threshold must be between 0 and 1")
                except (ValueError, TypeError):
                    errors.append("confidence_threshold must be a valid number")
            
            # Проверяем risk_level
            if "risk_level" in parameters:
                try:
                    risk = Decimal(str(parameters["risk_level"]))
                    if risk < 0 or risk > 1:
                        errors.append("risk_level must be between 0 and 1")
                except (ValueError, TypeError):
                    errors.append("risk_level must be a valid number")
            
        return errors

    @staticmethod
    def merge_parameters(
        default_params: Dict[str, Any], custom_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Объединить параметры по умолчанию с пользовательскими."""
        result = default_params.copy()
        result.update(custom_params)
        return result

    @staticmethod
    def calculate_risk_adjusted_return(
        returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal:
        """Рассчитать доходность с поправкой на риск."""
        if not returns:
            return Decimal("0")
        avg_return = sum(returns) / len(returns)
        volatility = StrategyUtils._calculate_volatility(returns)
        if volatility == 0:
            return Decimal("0")
        risk_adjusted = (float(avg_return) - float(risk_free_rate)) / float(volatility)
        return Decimal(str(risk_adjusted)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    @staticmethod
    def _calculate_volatility(returns: List[Decimal]) -> Decimal:
        """Рассчитать волатильность."""
        if len(returns) < 2:
            return Decimal("0")
        mean_return = sum(returns) / len(returns)
        squared_diff_sum = sum([(float(r) - float(mean_return)) ** 2 for r in returns])
        variance = squared_diff_sum / (len(returns) - 1)
        # Используем math.sqrt для Decimal
        import math
        return Decimal(str(math.sqrt(float(variance)))).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )


# Функции расчета производительности
def calculate_sharpe_ratio(
    returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")
) -> Decimal:
    """
    Рассчитать коэффициент Шарпа.
    Args:
        returns: Список доходностей
        risk_free_rate: Безрисковая ставка
    Returns:
        Decimal: Коэффициент Шарпа
    """
    if not returns or len(returns) < 2:
        return Decimal("0")
    try:
        returns_array = np.array([float(r) for r in returns])
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        if std_return == 0:
            return Decimal("0")
        sharpe = (avg_return - float(risk_free_rate)) / std_return
        return Decimal(str(sharpe)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return Decimal("0")


def calculate_sortino_ratio(
    returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")
) -> Decimal:
    """
    Рассчитать коэффициент Сортино.
    Args:
        returns: Список доходностей
        risk_free_rate: Безрисковая ставка
    Returns:
        Decimal: Коэффициент Сортино
    """
    if not returns or len(returns) < 2:
        return Decimal("0")
    try:
        returns_array = np.array([float(r) for r in returns])
        avg_return = np.mean(returns_array)
        # Только отрицательные доходности для downside deviation
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return Decimal("0")
        downside_deviation = np.std(negative_returns, ddof=1)
        if downside_deviation == 0:
            return Decimal("0")
        sortino = (avg_return - float(risk_free_rate)) / downside_deviation
        return Decimal(str(sortino)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating Sortino ratio: {e}")
        return Decimal("0")


def calculate_max_drawdown(equity_curve: List[Decimal]) -> Decimal:
    """
    Рассчитать максимальную просадку.
    Args:
        equity_curve: Кривая капитала
    Returns:
        Decimal: Максимальная просадка
    """
    if not equity_curve or len(equity_curve) < 2:
        return Decimal("0")
    try:
        equity_array = np.array([float(e) for e in equity_curve])
        peak = equity_array[0]
        max_dd = Decimal("0")
        for value in equity_array:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else Decimal("0")
            if dd > max_dd:
                max_dd = dd
        return max_dd.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return Decimal("0")


def calculate_win_rate(trades: List[Dict[str, Any]]) -> Decimal:
    """
    Рассчитать процент выигрышных сделок.
    Args:
        trades: Список сделок
    Returns:
        Decimal: Процент выигрышных сделок
    """
    if not trades:
        return Decimal("0")
    try:
        winning_trades = sum(
            1 for trade in trades if float(trade.get("pnl", Decimal("0"))) > 0
        )
        win_rate = Decimal(str(winning_trades / len(trades)))
        return win_rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return Decimal("0")


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> Decimal:
    """
    Рассчитать фактор прибыли.
    Args:
        trades: Список сделок
    Returns:
        Decimal: Фактор прибыли
    """
    if not trades:
        return Decimal("0")
    try:
        gross_profit: Decimal = Decimal(str(sum(
            Decimal(str(trade.get("pnl", Decimal("0"))))
            for trade in trades
            if float(trade.get("pnl", Decimal("0"))) > 0
        )))
        gross_loss: Decimal = Decimal(str(abs(
            sum(
                Decimal(str(trade.get("pnl", Decimal("0"))))
                for trade in trades
                if float(trade.get("pnl", Decimal("0"))) < 0
            )
        )))
        if gross_loss == 0:
            return Decimal("0")
        profit_factor = gross_profit / gross_loss
        return profit_factor.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating profit factor: {e}")
        return Decimal("0")


def calculate_avg_trade(trades: List[Dict[str, Any]]) -> Decimal:
    """
    Рассчитать среднюю сделку.
    Args:
        trades: Список сделок
    Returns:
        Decimal: Средняя сделка
    """
    if not trades:
        return Decimal("0")
    try:
        total_pnl: Decimal = Decimal(str(sum(Decimal(str(trade.get("pnl", Decimal("0")))) for trade in trades)))
        avg_trade = total_pnl / len(trades)
        return avg_trade.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error calculating average trade: {e}")
        return Decimal("0")


# Функции валидации
def validate_trading_pair(trading_pair: Any) -> bool:
    """
    Валидировать торговую пару.
    Args:
        trading_pair: Торговая пара
    Returns:
        bool: True если валидна
    """
    if not isinstance(trading_pair, str):
        return False
    # Паттерн для валидации торговых пар
    pattern = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
    if not pattern.match(trading_pair):
        return False
    # Проверяем, что базовая и котируемая валюта разные
    base, quote = trading_pair.split("/")
    return base != quote


def validate_strategy_parameters(parameters: Dict[str, Any]) -> List[str]:
    """
    Валидировать параметры стратегии.
    Args:
        parameters: Параметры стратегии
    Returns:
        List[str]: Список ошибок валидации
    """
    errors = []
    # Проверяем обязательные параметры
    required_params = ["confidence_threshold", "risk_level", "max_position_size"]
    for param in required_params:
        if param not in parameters:
            errors.append(f"Required parameter {param} is missing")
    # Валидируем confidence_threshold
    if "confidence_threshold" in parameters:
        try:
            confidence = Decimal(str(parameters["confidence_threshold"]))
            if not (Decimal("0.0") <= confidence <= Decimal("1.0")):
                errors.append("Confidence threshold must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            errors.append("Confidence threshold must be a valid decimal")
    # Валидируем risk_level
    if "risk_level" in parameters:
        risk_level = parameters["risk_level"]
        if not isinstance(risk_level, (str, RiskLevel)):
            errors.append("Risk level must be a string or RiskLevel")
        elif isinstance(risk_level, str) and risk_level not in [
            "low",
            "medium",
            "high",
        ]:
            errors.append("Risk level must be one of: low, medium, high")
    # Валидируем max_position_size
    if "max_position_size" in parameters:
        try:
            position_size = Decimal(str(parameters["max_position_size"]))
            if not (Decimal("0.001") <= position_size <= Decimal("1.0")):
                errors.append("Max position size must be between 0.001 and 1.0")
        except (ValueError, TypeError):
            errors.append("Max position size must be a valid decimal")
    return errors


def normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализовать параметры стратегии.
    Args:
        parameters: Параметры стратегии
    Returns:
        Dict[str, Any]: Нормализованные параметры
    """
    normalized: Dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, (int, float)):
            normalized[key] = Decimal(str(value))
        elif isinstance(value, str) and key == "risk_level":
            normalized[key] = value.lower()
        elif isinstance(value, str) and key in ["confidence_threshold", "max_position_size", "stop_loss", "take_profit"]:
            try:
                normalized[key] = Decimal(value)
            except (ValueError, InvalidOperation):
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


@dataclass
class StrategyPerformanceCalculator:
    """Калькулятор производительности стратегии."""

    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Decimal] = field(default_factory=list)
    returns: List[Decimal] = field(default_factory=list)

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Добавить сделку."""
        self.trades.append(trade)
        # Обновляем кривую капитала
        if self.equity_curve and len(self.equity_curve) > 0:
            last_equity = self.equity_curve[-1]
            new_equity = last_equity + trade.get("pnl", Decimal("0"))
        else:
            new_equity = trade.get("pnl", Decimal("0"))
        self.equity_curve.append(new_equity)
        # Рассчитываем доходность
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            if prev_equity != 0:
                return_rate = (new_equity - prev_equity) / prev_equity
                self.returns.append(return_rate)

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Рассчитать все метрики производительности."""
        return {
            "total_trades": len(self.trades),
            "winning_trades": sum(
                1 for t in self.trades if t.get("pnl", Decimal("0")) > 0
            ),
            "losing_trades": sum(
                1 for t in self.trades if t.get("pnl", Decimal("0")) < 0
            ),
            "win_rate": calculate_win_rate(self.trades),
            "profit_factor": calculate_profit_factor(self.trades),
            "avg_trade": calculate_avg_trade(self.trades),
            "total_pnl": sum(t.get("pnl", Decimal("0")) for t in self.trades),
            "sharpe_ratio": calculate_sharpe_ratio(self.returns),
            "sortino_ratio": calculate_sortino_ratio(self.returns),
            "max_drawdown": calculate_max_drawdown(self.equity_curve),
            "avg_win": self._calculate_avg_win(),
            "avg_loss": self._calculate_avg_loss(),
            "largest_win": self._calculate_largest_win(),
            "largest_loss": self._calculate_largest_loss(),
            "consecutive_wins": self._calculate_consecutive_wins(),
            "consecutive_losses": self._calculate_consecutive_losses(),
        }

    def _calculate_avg_win(self) -> Decimal:
        """Рассчитать средний выигрыш."""
        wins = [
            t.get("pnl", Decimal("0"))
            for t in self.trades
            if t.get("pnl", Decimal("0")) > 0
        ]
        if not wins:
            return Decimal("0")
        return Decimal(str(sum(wins) / len(wins)))

    def _calculate_avg_loss(self) -> Decimal:
        """Рассчитать средний проигрыш."""
        losses = [
            t.get("pnl", Decimal("0"))
            for t in self.trades
            if t.get("pnl", Decimal("0")) < 0
        ]
        if not losses:
            return Decimal("0")
        return Decimal(str(sum(losses) / len(losses)))

    def _calculate_largest_win(self) -> Decimal:
        """Рассчитать самый большой выигрыш."""
        wins = [
            t.get("pnl", Decimal("0"))
            for t in self.trades
            if t.get("pnl", Decimal("0")) > 0
        ]
        return max(wins) if wins else Decimal("0")

    def _calculate_largest_loss(self) -> Decimal:
        """Рассчитать самый большой проигрыш."""
        losses = [
            t.get("pnl", Decimal("0"))
            for t in self.trades
            if t.get("pnl", Decimal("0")) < 0
        ]
        return min(losses) if losses else Decimal("0")

    def _calculate_consecutive_wins(self) -> int:
        """Рассчитать максимальное количество последовательных выигрышей."""
        max_consecutive = 0
        current_consecutive = 0
        for trade in self.trades:
            if trade.get("pnl", Decimal("0")) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive

    def _calculate_consecutive_losses(self) -> int:
        """Рассчитать максимальное количество последовательных проигрышей."""
        max_consecutive = 0
        current_consecutive = 0
        for trade in self.trades:
            if trade.get("pnl", Decimal("0")) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive


@dataclass
class StrategyRiskManager:
    """Менеджер рисков стратегии."""

    max_position_size: Decimal = Decimal("0.1")
    max_daily_loss: Decimal = Decimal("0.05")
    max_drawdown: Decimal = Decimal("0.2")
    position_sizing_method: str = "fixed"  # fixed, kelly, volatility
    risk_per_trade: Decimal = Decimal("0.02")

    def calculate_position_size(
        self,
        account_balance: Decimal,
        risk_per_trade: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None,
        confidence: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Рассчитать размер позиции.
        Args:
            account_balance: Баланс счета
            risk_per_trade: Риск на сделку
            volatility: Волатильность
            confidence: Уверенность
        Returns:
            Decimal: Размер позиции
        """
        risk_per_trade = risk_per_trade or self.risk_per_trade
        if self.position_sizing_method == "fixed":
            return account_balance * self.max_position_size
        elif self.position_sizing_method == "kelly":
            return self._calculate_kelly_position_size(account_balance, confidence)
        elif self.position_sizing_method == "volatility":
            return self._calculate_volatility_position_size(account_balance, volatility)
        else:
            return account_balance * risk_per_trade

    def _calculate_kelly_position_size(
        self, account_balance: Decimal, confidence: Optional[Decimal]
    ) -> Decimal:
        """Рассчитать размер позиции по формуле Келли."""
        if not confidence:
            return account_balance * self.max_position_size
        # Упрощенная формула Келли
        kelly_fraction = (confidence - Decimal("0.5")) * 2
        kelly_fraction = max(Decimal("0"), min(kelly_fraction, self.max_position_size))
        return account_balance * kelly_fraction

    def _calculate_volatility_position_size(
        self, account_balance: Decimal, volatility: Optional[Decimal]
    ) -> Decimal:
        """Рассчитать размер позиции на основе волатильности."""
        if not volatility:
            return account_balance * self.max_position_size
        # Обратная зависимость от волатильности
        volatility_factor = Decimal("1") / (Decimal("1") + volatility)
        position_size = self.max_position_size * volatility_factor
        return account_balance * position_size

    def check_risk_limits(
        self, current_drawdown: Decimal, daily_pnl: Decimal, total_exposure: Decimal
    ) -> Tuple[bool, List[str]]:
        """
        Проверить лимиты риска.
        Args:
            current_drawdown: Текущая просадка
            daily_pnl: Дневная прибыль/убыток
            total_exposure: Общая экспозиция
        Returns:
            Tuple[bool, List[str]]: (в пределах лимитов, список нарушений)
        """
        violations = []
        if current_drawdown > self.max_drawdown:
            violations.append(
                f"Drawdown {current_drawdown} exceeds limit {self.max_drawdown}"
            )
        if daily_pnl < -self.max_daily_loss:
            violations.append(
                f"Daily loss {daily_pnl} exceeds limit {self.max_daily_loss}"
            )
        if total_exposure > self.max_position_size:
            violations.append(
                f"Total exposure {total_exposure} exceeds limit {self.max_position_size}"
            )
        return len(violations) == 0, violations


@dataclass
class StrategyOptimizer:
    """Оптимизатор стратегии."""

    optimization_method: str = "grid_search"  # grid_search, genetic, bayesian
    max_iterations: int = 1000
    convergence_threshold: Decimal = Decimal("0.001")

    def optimize_parameters(
        self,
        strategy_class: type,
        parameter_ranges: Dict[str, List[Any]],
        evaluation_function: Callable[[Dict[str, Any]], float],
        constraints: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Оптимизировать параметры стратегии.
        Args:
            strategy_class: Класс стратегии
            parameter_ranges: Диапазоны параметров
            evaluation_function: Функция оценки
            constraints: Ограничения
        Returns:
            Tuple[Dict[str, Any], float]: (лучшие параметры, лучший результат)
        """
        if self.optimization_method == "grid_search":
            return self._grid_search_optimization(
                parameter_ranges, evaluation_function, constraints
            )
        elif self.optimization_method == "genetic":
            return self._genetic_optimization(
                parameter_ranges, evaluation_function, constraints
            )
        elif self.optimization_method == "bayesian":
            return self._bayesian_optimization(
                parameter_ranges, evaluation_function, constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _grid_search_optimization(
        self,
        parameter_ranges: Dict[str, List[Any]],
        evaluation_function: Callable[[Dict[str, Any]], float],
        constraints: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Оптимизация методом перебора."""
        best_params = {}
        best_score = float("-inf")
        # Генерируем все комбинации параметров
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        for params in param_combinations:
            # Проверяем ограничения
            if constraints and not self._check_constraints(params, constraints):
                continue
            try:
                score = evaluation_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        return best_params, best_score

    def _genetic_optimization(
        self,
        parameter_ranges: Dict[str, List[Any]],
        evaluation_function: Callable[[Dict[str, Any]], float],
        constraints: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Генетическая оптимизация."""
        # Упрощенная реализация генетического алгоритма
        population_size = 50
        generations = 20
        # Инициализируем популяцию
        population = [
            self._random_parameters(parameter_ranges) for _ in range(population_size)
        ]
        best_params = {}
        best_score = float("-inf")
        for generation in range(generations):
            # Оцениваем популяцию
            scores = []
            for params in population:
                if constraints and not self._check_constraints(params, constraints):
                    scores.append(float("-inf"))
                else:
                    try:
                        score = evaluation_function(params)
                        scores.append(score)
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                    except Exception:
                        scores.append(float("-inf"))
            # Селекция и скрещивание
            population = self._genetic_selection_crossover(
                population, scores, parameter_ranges
            )
        return best_params, best_score

    def _bayesian_optimization(
        self,
        parameter_ranges: Dict[str, List[Any]],
        evaluation_function: Callable[[Dict[str, Any]], float],
        constraints: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Байесовская оптимизация."""
        # Упрощенная реализация байесовской оптимизации
        best_params = {}
        best_score = float("-inf")
        # Начинаем с случайных точек
        for i in range(self.max_iterations):
            params = self._random_parameters(parameter_ranges)
            if constraints and not self._check_constraints(params, constraints):
                continue
            try:
                score = evaluation_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception:
                continue
        return best_params, best_score

    def _generate_parameter_combinations(
        self, parameter_ranges: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Генерировать комбинации параметров."""
        import itertools

        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        return combinations

    def _random_parameters(
        self, parameter_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Генерировать случайные параметры."""
        import random

        params = {}
        for key, values in parameter_ranges.items():
            params[key] = random.choice(values)
        return params

    def _check_constraints(
        self, params: Dict[str, Any], constraints: Dict[str, Callable]
    ) -> bool:
        """Проверить ограничения."""
        for constraint_name, constraint_func in constraints.items():
            if not constraint_func(params):
                return False
        return True

    def _genetic_selection_crossover(
        self,
        population: List[Dict[str, Any]],
        scores: List[float],
        parameter_ranges: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Селекция и скрещивание для генетического алгоритма."""
        import random

        # Сортируем по оценкам
        sorted_population = [
            x for _, x in sorted(zip(scores, population), reverse=True)
        ]
        # Отбираем лучших
        elite_size = len(sorted_population) // 4
        elite = sorted_population[:elite_size]
        # Скрещиваем
        new_population = elite.copy()
        while len(new_population) < len(population):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = self._crossover(parent1, parent2, parameter_ranges)
            new_population.append(child)
        return new_population

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        parameter_ranges: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Скрещивание двух родителей."""
        import random

        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        # Мутация
        if random.random() < 0.1:
            mutation_key = random.choice(list(parameter_ranges.keys()))
            child[mutation_key] = random.choice(parameter_ranges[mutation_key])
        return child


# Технические индикаторы
def calculate_sma(prices: List[float], period: int) -> List[float]:
    """
    Рассчитать простую скользящую среднюю.
    Args:
        prices: Список цен
        period: Период
    Returns:
        List[float]: Список значений SMA
    """
    if len(prices) < period:
        return []
    sma_values = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        sma = sum(window) / len(window)
        sma_values.append(sma)
    return sma_values


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """
    Рассчитать экспоненциальную скользящую среднюю.
    Args:
        prices: Список цен
        period: Период
    Returns:
        List[float]: Список значений EMA
    """
    if len(prices) < period:
        return []
    multiplier = 2.0 / (period + 1)
    ema_values = [prices[0]]
    for i in range(1, len(prices)):
        if len(ema_values) > 0:
            # Используем Decimal для точного расчета EMA
            price_decimal = to_trading_decimal(prices[i])
            multiplier_decimal = to_trading_decimal(multiplier)
            previous_ema_decimal = to_trading_decimal(ema_values[-1])
            
            ema_decimal = (price_decimal * multiplier_decimal) + (previous_ema_decimal * (to_trading_decimal(1) - multiplier_decimal))
            ema = float(ema_decimal)
        else:
            ema = prices[i]  # Fallback если ema_values пуст
        ema_values.append(ema)
    return ema_values


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Рассчитать RSI.
    Args:
        prices: Список цен
        period: Период
    Returns:
        List[float]: Список значений RSI
    """
    if len(prices) < period + 1:
        return []
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return [float(x) for x in rsi.tolist()]


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Рассчитать MACD.
    Args:
        prices: Список цен
        fast_period: Быстрый период
        slow_period: Медленный период
        signal_period: Период сигнала
    Returns:
        Tuple[List[float], List[float], List[float]]: (MACD, Signal, Histogram)
    """
    if len(prices) < slow_period:
        return [], [], []
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    # Выравниваем длины
    min_length = min(len(ema_fast), len(ema_slow))
    ema_fast = ema_fast[-min_length:]
    ema_slow = ema_slow[-min_length:]
    macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal_period)
    # Выравниваем длины для histogram
    min_length = min(len(macd_line), len(signal_line))
    macd_line = macd_line[-min_length:]
    signal_line = signal_line[-min_length:]
    histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: List[float], period: int = 20, std_dev: float = 2.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    Рассчитать полосы Боллинджера.
    Args:
        prices: Список цен
        period: Период
        std_dev: Стандартное отклонение
    Returns:
        Tuple[List[float], List[float], List[float]]: (Upper, Middle, Lower)
    """
    if len(prices) < period:
        return [], [], []
    sma_values = calculate_sma(prices, period)
    upper_band = []
    lower_band = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        std = float(np.std(window))
        sma = sma_values[i - period + 1]
        upper_band.append(sma + std_dev * std)
        lower_band.append(sma - std_dev * std)
    return upper_band, sma_values, lower_band


def calculate_atr(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> List[float]:
    """
    Рассчитать средний истинный диапазон.
    Args:
        highs: Список максимумов
        lows: Список минимумов
        closes: Список цен закрытия
        period: Период
    Returns:
        List[float]: Список значений ATR
    """
    if len(highs) < period + 1:
        return []
    true_ranges = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)
    atr_values = []
    for i in range(period - 1, len(true_ranges)):
        window = true_ranges[i - period + 1 : i + 1]
        atr = sum(window) / len(window)
        atr_values.append(atr)
    return atr_values


def calculate_volume_sma(volumes: List[float], period: int) -> List[float]:
    """
    Рассчитать скользящую среднюю объема.
    Args:
        volumes: Список объемов
        period: Период
    Returns:
        List[float]: Список значений Volume SMA
    """
    return calculate_sma(volumes, period)


def detect_support_resistance(
    prices: List[float], window: int = 20
) -> Tuple[List[float], List[float]]:
    """
    Обнаружить уровни поддержки и сопротивления.
    Args:
        prices: Список цен
        window: Размер окна
    Returns:
        Tuple[List[float], List[float]]: (поддержка, сопротивление)
    """
    if len(prices) < window:
        return [], []
    support_levels = []
    resistance_levels = []
    for i in range(window, len(prices) - window):
        window_prices = prices[i - window : i + window]
        current_price = prices[i]
        if current_price == min(window_prices):
            support_levels.append(current_price)
        if current_price == max(window_prices):
            resistance_levels.append(current_price)
    return support_levels, resistance_levels


def calculate_volatility(prices: List[float], period: int = 20) -> List[float]:
    """
    Рассчитать волатильность.
    Args:
        prices: Список цен
        period: Период
    Returns:
        List[float]: Список значений волатильности
    """
    if len(prices) < period:
        return []
    volatility_values = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        returns = np.diff(window) / window[:-1]
        volatility = float(np.std(returns))
        volatility_values.append(volatility)
    return volatility_values


def calculate_momentum(prices: List[float], period: int = 10) -> List[float]:
    """
    Рассчитать моментум.
    Args:
        prices: Список цен
        period: Период
    Returns:
        List[float]: Список значений моментума
    """
    if len(prices) < period:
        return []
    momentum_values = []
    for i in range(period - 1, len(prices)):
        momentum = prices[i] - prices[i - period]
        momentum_values.append(momentum)
    return momentum_values
