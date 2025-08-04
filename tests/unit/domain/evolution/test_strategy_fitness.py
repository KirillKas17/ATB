"""
Unit тесты для strategy_fitness.py.

Покрывает:
- TradeResult - результат торговой сделки
- StrategyEvaluationResult - результат оценки стратегии
- StrategyFitnessEvaluator - оценщик эффективности стратегий
- Методы расчета P&L и ROI
- Методы оценки риска и производительности
- Симуляция торговли и анализ сделок
"""

import pytest
from shared.numpy_utils import np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from domain.evolution.strategy_fitness import (
    TradeResult,
    StrategyEvaluationResult,
    StrategyFitnessEvaluator
)
from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate
)
from domain.types.evolution_types import (
    AccuracyScore,
    ConsistencyScore,
    EntryCondition,
    ExitCondition,
    FitnessScore,
    FitnessWeights,
    ProfitabilityScore,
    RiskScore,
    StrategyPerformance,
    TradePosition
)
from domain.exceptions.base_exceptions import ValidationError


class TestTradeResult:
    """Тесты для TradeResult."""

    @pytest.fixture
    def sample_trade(self) -> TradeResult:
        """Тестовая сделка."""
        return TradeResult(
            entry_time=datetime(2024, 1, 1, 10, 0, 0),
            exit_time=datetime(2024, 1, 1, 11, 0, 0),
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("1.0"),
            signal_type="buy",
            holding_time=3600,
            commission=Decimal("10")
        )

    def test_creation_valid(self, sample_trade: TradeResult) -> None:
        """Тест создания с валидными данными."""
        assert sample_trade.entry_price == Decimal("50000")
        assert sample_trade.exit_price == Decimal("51000")
        assert sample_trade.quantity == Decimal("1.0")
        assert sample_trade.signal_type == "buy"
        assert sample_trade.holding_time == 3600
        assert sample_trade.commission == Decimal("10")

    def test_calculate_pnl_buy_profit(self, sample_trade: TradeResult) -> None:
        """Тест расчета P&L для прибыльной покупки."""
        sample_trade.calculate_pnl()
        
        expected_pnl = (Decimal("51000") - Decimal("50000")) * Decimal("1.0") - Decimal("10")
        assert sample_trade.pnl == expected_pnl
        assert sample_trade.success is True

    def test_calculate_pnl_buy_loss(self) -> None:
        """Тест расчета P&L для убыточной покупки."""
        trade = TradeResult(
            entry_price=Decimal("50000"),
            exit_price=Decimal("49000"),
            quantity=Decimal("1.0"),
            signal_type="buy",
            commission=Decimal("10")
        )
        trade.calculate_pnl()
        
        expected_pnl = (Decimal("49000") - Decimal("50000")) * Decimal("1.0") - Decimal("10")
        assert trade.pnl == expected_pnl
        assert trade.success is False

    def test_calculate_pnl_sell_profit(self) -> None:
        """Тест расчета P&L для прибыльной продажи."""
        trade = TradeResult(
            entry_price=Decimal("50000"),
            exit_price=Decimal("49000"),
            quantity=Decimal("1.0"),
            signal_type="sell",
            commission=Decimal("10")
        )
        trade.calculate_pnl()
        
        expected_pnl = (Decimal("50000") - Decimal("49000")) * Decimal("1.0") - Decimal("10")
        assert trade.pnl == expected_pnl
        assert trade.success is True

    def test_get_roi(self, sample_trade: TradeResult) -> None:
        """Тест получения ROI."""
        sample_trade.calculate_pnl()
        roi = sample_trade.get_roi()
        
        expected_roi = (Decimal("1000") - Decimal("10")) / (Decimal("50000") * Decimal("1.0"))
        assert roi == expected_roi

    def test_get_risk_metrics(self, sample_trade: TradeResult) -> None:
        """Тест получения метрик риска."""
        sample_trade.calculate_pnl()
        metrics = sample_trade.get_risk_metrics()
        
        assert isinstance(metrics, dict)
        assert "pnl" in metrics
        assert "pnl_pct" in metrics
        assert "roi" in metrics
        assert "holding_time" in metrics
        assert "success" in metrics
        assert isinstance(metrics["pnl"], float)
        assert isinstance(metrics["success"], bool)

    def test_to_dict(self, sample_trade: TradeResult) -> None:
        """Тест преобразования в словарь."""
        result = sample_trade.to_dict()
        
        assert isinstance(result, dict)
        assert result["entry_price"] == "50000"
        assert result["exit_price"] == "51000"
        assert result["quantity"] == "1.0"
        assert result["signal_type"] == "buy"
        assert "roi" in result

    def test_from_dict(self) -> None:
        """Тест создания из словаря."""
        data = {
            "id": str(uuid4()),
            "entry_time": "2024-01-01T10:00:00",
            "exit_time": "2024-01-01T11:00:00",
            "entry_price": "50000",
            "exit_price": "51000",
            "quantity": "1.0",
            "pnl": "990",
            "pnl_pct": "0.0198",
            "commission": "10",
            "signal_type": "buy",
            "holding_time": 3600,
            "success": True,
            "metadata": {}
        }
        
        trade = TradeResult.from_dict(data)
        
        assert trade.entry_price == Decimal("50000")
        assert trade.exit_price == Decimal("51000")
        assert trade.quantity == Decimal("1.0")
        assert trade.signal_type == "buy"
        assert trade.success is True


class TestStrategyEvaluationResult:
    """Тесты для StrategyEvaluationResult."""

    @pytest.fixture
    def sample_evaluation(self) -> StrategyEvaluationResult:
        """Тестовая оценка стратегии."""
        return StrategyEvaluationResult(
            strategy_id=uuid4(),
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=Decimal("0.7"),
            total_pnl=Decimal("1000"),
            net_pnl=Decimal("990"),
            profitability=Decimal("0.099"),
            max_drawdown_pct=Decimal("0.05")
        )

    @pytest.fixture
    def sample_trades(self) -> List[TradeResult]:
        """Тестовые сделки."""
        trades = []
        for i in range(10):
            trade = TradeResult(
                entry_time=datetime(2024, 1, 1, 10 + i, 0, 0),
                exit_time=datetime(2024, 1, 1, 10 + i, 30, 0),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if i < 7 else Decimal("49000"),
                quantity=Decimal("1.0"),
                signal_type="buy",
                holding_time=1800,
                commission=Decimal("10")
            )
            trade.calculate_pnl()
            trades.append(trade)
        return trades

    def test_creation_valid(self, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест создания с валидными данными."""
        assert sample_evaluation.total_trades == 10
        assert sample_evaluation.winning_trades == 7
        assert sample_evaluation.losing_trades == 3
        assert sample_evaluation.win_rate == Decimal("0.7")
        assert sample_evaluation.total_pnl == Decimal("1000")
        assert sample_evaluation.net_pnl == Decimal("990")

    def test_add_trade(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест добавления сделки."""
        initial_trades = len(sample_evaluation.trades)
        sample_evaluation.add_trade(sample_trades[0])
        
        assert len(sample_evaluation.trades) == initial_trades + 1
        assert sample_evaluation.total_trades == 1

    def test_recalculate_metrics(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест пересчета метрик."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        assert sample_evaluation.total_trades == 10
        assert sample_evaluation.winning_trades == 7
        assert sample_evaluation.losing_trades == 3
        assert sample_evaluation.win_rate == Decimal("0.7")
        assert sample_evaluation.profit_factor > Decimal("0")

    def test_calculate_risk_metrics(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест расчета риск-метрик."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        risk_metrics = sample_evaluation.get_risk_metrics()
        
        assert isinstance(risk_metrics, dict)
        assert "max_drawdown" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "sortino_ratio" in risk_metrics
        assert "calmar_ratio" in risk_metrics
        assert "profit_factor" in risk_metrics
        assert "win_rate" in risk_metrics

    def test_check_approval_criteria(self, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест проверки критериев одобрения."""
        context = EvolutionContext(
            min_accuracy=Decimal("0.6"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.1"),
            min_sharpe=Decimal("1.0")
        )
        
        result = sample_evaluation.check_approval_criteria(context)
        
        assert isinstance(result, bool)
        assert isinstance(sample_evaluation.is_approved, bool)
        assert isinstance(sample_evaluation.approval_reason, str)

    def test_get_fitness_score(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест получения fitness score."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        fitness_score = sample_evaluation.get_fitness_score()
        
        assert isinstance(fitness_score, FitnessScore)
        assert fitness_score.value > Decimal("0")

    def test_get_fitness_score_custom_weights(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест получения fitness score с кастомными весами."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        weights = FitnessWeights(
            accuracy=Decimal("0.4"),
            profitability=Decimal("0.3"),
            risk=Decimal("0.2"),
            consistency=Decimal("0.1")
        )
        
        fitness_score = sample_evaluation.get_fitness_score(weights)
        
        assert isinstance(fitness_score, FitnessScore)
        assert fitness_score.value > Decimal("0")

    def test_get_performance_summary(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест получения сводки производительности."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        summary = sample_evaluation.get_performance_summary()
        
        assert isinstance(summary, StrategyPerformance)
        assert summary.total_trades == 10
        assert summary.winning_trades == 7
        assert summary.losing_trades == 3
        assert summary.win_rate == 0.7

    def test_get_trade_analysis(self, sample_evaluation: StrategyEvaluationResult, sample_trades: List[TradeResult]) -> None:
        """Тест получения анализа сделок."""
        for trade in sample_trades:
            sample_evaluation.add_trade(trade)
        
        analysis = sample_evaluation.get_trade_analysis()
        
        assert isinstance(analysis, dict)
        assert "signal_analysis" in analysis
        assert "holding_time_analysis" in analysis
        assert "monthly_performance" in analysis

    def test_to_dict(self, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест преобразования в словарь."""
        result = sample_evaluation.to_dict()
        
        assert isinstance(result, dict)
        assert result["total_trades"] == 10
        assert result["winning_trades"] == 7
        assert result["losing_trades"] == 3
        assert result["win_rate"] == "0.7"
        assert "fitness_score" in result

    def test_from_dict(self) -> None:
        """Тест создания из словаря."""
        data = {
            "id": str(uuid4()),
            "strategy_id": str(uuid4()),
            "total_trades": 10,
            "winning_trades": 7,
            "losing_trades": 3,
            "win_rate": "0.7",
            "accuracy": "0.7",
            "total_pnl": "1000",
            "net_pnl": "990",
            "profitability": "0.099",
            "profit_factor": "2.33",
            "max_drawdown": "500",
            "max_drawdown_pct": "0.05",
            "sharpe_ratio": "1.5",
            "sortino_ratio": "2.0",
            "calmar_ratio": "1.98",
            "average_trade": "99",
            "best_trade": "1000",
            "worst_trade": "-100",
            "average_win": "200",
            "average_loss": "-150",
            "largest_win": "1000",
            "largest_loss": "-100",
            "average_holding_time": 1800,
            "total_trading_time": 18000,
            "start_date": "2024-01-01T10:00:00",
            "end_date": "2024-01-01T15:00:00",
            "is_approved": True,
            "approval_reason": "All criteria met",
            "evaluation_time": "2024-01-01T16:00:00",
            "metadata": {},
            "trades": [],
            "equity_curve": []
        }
        
        evaluation = StrategyEvaluationResult.from_dict(data)
        
        assert evaluation.total_trades == 10
        assert evaluation.winning_trades == 7
        assert evaluation.losing_trades == 3
        assert evaluation.win_rate == Decimal("0.7")
        assert evaluation.is_approved is True


class TestStrategyFitnessEvaluator:
    """Тесты для StrategyFitnessEvaluator."""

    @pytest.fixture
    def sample_weights(self) -> FitnessWeights:
        """Тестовые веса."""
        return FitnessWeights(
            accuracy=Decimal("0.3"),
            profitability=Decimal("0.3"),
            risk=Decimal("0.2"),
            consistency=Decimal("0.2")
        )

    @pytest.fixture
    def evaluator(self, sample_weights: FitnessWeights) -> StrategyFitnessEvaluator:
        """Тестовый оценщик."""
        return StrategyFitnessEvaluator(weights=sample_weights)

    @pytest.fixture
    def sample_candidate(self) -> StrategyCandidate:
        """Тестовый кандидат стратегии."""
        return StrategyCandidate(
            id=uuid4(),
            name="Test Strategy",
            description="Test strategy for evaluation",
            entry_conditions=[
                EntryCondition(
                    indicator="sma",
                    operator="crossover",
                    value=50.0,
                    timeframe="1h"
                )
            ],
            exit_conditions=[
                ExitCondition(
                    indicator="profit_target",
                    operator="greater_than",
                    value=0.02,
                    timeframe="1h"
                )
            ],
            parameters={
                "position_size": 0.1,
                "stop_loss": 0.01,
                "take_profit": 0.02
            }
        )

    @pytest.fixture
    def sample_historical_data(self) -> pd.DataFrame:
        """Тестовые исторические данные."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = {
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50100,
            'low': np.random.randn(100).cumsum() + 49900,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 100),
            'sma_20': np.random.randn(100).cumsum() + 50000,
            'sma_50': np.random.randn(100).cumsum() + 49950
        }
        return pd.DataFrame(data, index=dates)

    def test_initialization_default_weights(self) -> None:
        """Тест инициализации с дефолтными весами."""
        evaluator = StrategyFitnessEvaluator()
        
        assert evaluator.weights is not None
        assert isinstance(evaluator.weights, FitnessWeights)

    def test_initialization_custom_weights(self, sample_weights: FitnessWeights) -> None:
        """Тест инициализации с кастомными весами."""
        evaluator = StrategyFitnessEvaluator(weights=sample_weights)
        
        assert evaluator.weights == sample_weights

    def test_evaluate_strategy(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест оценки стратегии."""
        result = evaluator.evaluate_strategy(
            sample_candidate,
            sample_historical_data,
            initial_capital=Decimal("10000")
        )
        
        assert isinstance(result, StrategyEvaluationResult)
        assert result.strategy_id == sample_candidate.id
        assert isinstance(result.total_trades, int)
        assert isinstance(result.win_rate, Decimal)
        assert isinstance(result.profitability, Decimal)

    def test_simulate_trading(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест симуляции торговли."""
        trades = evaluator._simulate_trading(
            sample_candidate,
            sample_historical_data,
            initial_capital=Decimal("10000")
        )
        
        assert isinstance(trades, list)
        for trade in trades:
            assert isinstance(trade, TradeResult)

    def test_check_entry_signals(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест проверки сигналов входа."""
        signal = evaluator._check_entry_signals(
            sample_candidate,
            sample_historical_data.iloc[50:60],
            50,
            sample_historical_data
        )
        
        # Сигнал может быть None или строкой
        assert signal is None or isinstance(signal, str)

    def test_check_exit_signals(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест проверки сигналов выхода."""
        position = TradePosition(
            entry_time=datetime.now(),
            entry_price=Decimal("50000"),
            quantity=Decimal("1.0"),
            signal_type="buy"
        )
        
        signal = evaluator._check_exit_signals(
            sample_candidate,
            sample_historical_data.iloc[50:60],
            position,
            50,
            sample_historical_data
        )
        
        # Сигнал может быть None или строкой
        assert signal is None or isinstance(signal, str)

    def test_evaluate_conditions(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест оценки условий."""
        conditions = [
            EntryCondition(
                indicator="sma",
                operator="crossover",
                value=50.0,
                timeframe="1h"
            )
        ]
        
        result = evaluator._evaluate_conditions(conditions)
        
        assert isinstance(result, bool)

    def test_evaluate_single_condition(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест оценки одного условия."""
        condition = EntryCondition(
            indicator="sma",
            operator="crossover",
            value=50.0,
            timeframe="1h"
        )
        
        result = evaluator._evaluate_single_condition(condition)
        
        assert isinstance(result, bool)

    def test_evaluate_dict_condition(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест оценки условия в виде словаря."""
        condition = {
            "indicator": "price",
            "operator": "greater_than",
            "value": 50000
        }
        
        result = evaluator._evaluate_dict_condition(condition)
        
        assert isinstance(result, bool)

    def test_open_position(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест открытия позиции."""
        position = evaluator._open_position(
            sample_candidate,
            sample_historical_data.iloc[50:51],
            "buy",
            Decimal("10000")
        )
        
        assert isinstance(position, TradePosition)
        assert position.signal_type == "buy"
        assert position.quantity > Decimal("0")

    def test_close_position(self, evaluator: StrategyFitnessEvaluator, sample_historical_data: pd.DataFrame) -> None:
        """Тест закрытия позиции."""
        position = TradePosition(
            entry_time=datetime.now(),
            entry_price=Decimal("50000"),
            quantity=Decimal("1.0"),
            signal_type="buy"
        )
        
        trade = evaluator._close_position(
            position,
            sample_historical_data.iloc[50:51],
            "profit_target"
        )
        
        assert isinstance(trade, TradeResult)
        assert trade.entry_price == Decimal("50000")
        assert trade.quantity == Decimal("1.0")

    def test_get_evaluation_result(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест получения результата оценки."""
        # Сначала выполняем оценку
        evaluator.evaluate_strategy(sample_candidate, sample_historical_data)
        
        result = evaluator.get_evaluation_result(sample_candidate.id)
        
        assert result is None or isinstance(result, StrategyEvaluationResult)

    def test_get_all_results(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест получения всех результатов."""
        results = evaluator.get_all_results()
        
        assert isinstance(results, list)
        assert all(isinstance(r, StrategyEvaluationResult) for r in results)

    def test_get_approved_strategies(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест получения одобренных стратегий."""
        strategies = evaluator.get_approved_strategies()
        
        assert isinstance(strategies, list)
        assert all(isinstance(s, StrategyEvaluationResult) for s in strategies)

    def test_get_top_strategies(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест получения топ стратегий."""
        strategies = evaluator.get_top_strategies(n=5)
        
        assert isinstance(strategies, list)
        assert len(strategies) <= 5
        assert all(isinstance(s, StrategyEvaluationResult) for s in strategies)

    def test_clear_results(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест очистки результатов."""
        evaluator.clear_results()
        
        results = evaluator.get_all_results()
        assert len(results) == 0

    def test_get_evaluation_statistics(self, evaluator: StrategyFitnessEvaluator) -> None:
        """Тест получения статистики оценки."""
        stats = evaluator.get_evaluation_statistics()
        
        assert isinstance(stats, dict)
        assert "total_evaluations" in stats
        assert "approved_strategies" in stats
        assert "average_fitness_score" in stats

    def test_error_handling_invalid_data(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate) -> None:
        """Тест обработки ошибок с невалидными данными."""
        invalid_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'close': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
        
        result = evaluator.evaluate_strategy(sample_candidate, invalid_data)
        
        assert isinstance(result, StrategyEvaluationResult)
        assert result.total_trades == 0

    def test_performance_with_large_data(self, evaluator: StrategyFitnessEvaluator, sample_candidate: StrategyCandidate) -> None:
        """Тест производительности с большими данными."""
        # Создаем большие исторические данные
        large_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 50000,
            'high': np.random.randn(1000).cumsum() + 50100,
            'low': np.random.randn(1000).cumsum() + 49900,
            'close': np.random.randn(1000).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 1000),
            'sma_20': np.random.randn(1000).cumsum() + 50000,
            'sma_50': np.random.randn(1000).cumsum() + 49950
        })
        
        start_time = datetime.now()
        result = evaluator.evaluate_strategy(sample_candidate, large_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Обработка должна быть быстрой (менее 5 секунд)
        assert processing_time < 5.0
        assert isinstance(result, StrategyEvaluationResult) 