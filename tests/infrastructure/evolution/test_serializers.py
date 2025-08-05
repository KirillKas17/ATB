"""
Юнит-тесты для сериализаторов.
"""
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from domain.evolution.strategy_fitness import StrategyEvaluationResult, TradeResult
from domain.evolution.strategy_model import (
    EvolutionContext, EvolutionStatus, StrategyCandidate,
    EntryRule, ExitRule, FilterConfig, FilterType,
    IndicatorConfig, IndicatorType, SignalType, StrategyType
)
from infrastructure.evolution.serializers import (
    candidate_to_model,
    model_to_candidate,
    evaluation_to_model,
    model_to_evaluation,
    context_to_model,
    model_to_context
)
from infrastructure.evolution.models import (
    StrategyCandidateModel,
    StrategyEvaluationModel,
    EvolutionContextModel
)
class TestCandidateSerializers:
    """Тесты для сериализации StrategyCandidate"""
    def test_candidate_to_model_conversion(self: "TestCandidateSerializers") -> None:
        """Тест конвертации StrategyCandidate в StrategyCandidateModel"""
        # Arrange
        candidate = StrategyCandidate(
            id=uuid4(),
            name="Test Candidate",
            description="Test description",
            strategy_type=StrategyType.ARBITRAGE,
            status=EvolutionStatus.GENERATED,
            generation=1,
            parent_ids=[uuid4(), uuid4()],
            mutation_count=2,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            indicators=[],
            filters=[],
            entry_rules=[],
            exit_rules=[],
            position_size_pct=Decimal("0.1"),
            max_positions=5,
            min_holding_time=300,
            max_holding_time=3600,
            metadata={"meta1": "value1"}
        )
        # Act
        model = candidate_to_model(candidate)
        # Assert
        assert model.id == str(candidate.id)
        assert model.name == "Test Candidate"
        assert model.description == "Test description"
        assert model.strategy_type == StrategyType.ARBITRAGE.value
        assert model.status == EvolutionStatus.GENERATED.value
        assert model.generation == 1
        assert model.mutation_count == 2
        assert model.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert model.updated_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert model.position_size_pct == "0.1"
        assert model.max_positions == 5
        assert model.min_holding_time == 300
        assert model.max_holding_time == 3600
    def test_model_to_candidate_conversion(self: "TestCandidateSerializers") -> None:
        """Тест конвертации StrategyCandidateModel в StrategyCandidate"""
        # Arrange
        candidate_id = uuid4()
        parent_id1 = uuid4()
        parent_id2 = uuid4()
        model = StrategyCandidateModel(
            id=str(candidate_id),
            name="Test Candidate",
            description="Test description",
            strategy_type=StrategyType.ARBITRAGE.value,
            status=EvolutionStatus.GENERATED.value,
            generation=1,
            parent_ids=json.dumps([str(parent_id1), str(parent_id2)]),
            mutation_count=2,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            indicators_config=json.dumps([]),
            filters_config=json.dumps([]),
            entry_rules_config=json.dumps([]),
            exit_rules_config=json.dumps([]),
            position_size_pct="0.1",
            max_positions=5,
            min_holding_time=300,
            max_holding_time=3600,
            meta_data=json.dumps({"meta1": "value1"})
        )
        # Act
        candidate = model_to_candidate(model)
        # Assert
        assert candidate.id == candidate_id
        assert candidate.name == "Test Candidate"
        assert candidate.description == "Test description"
        assert candidate.strategy_type == StrategyType.ARBITRAGE
        assert candidate.status == EvolutionStatus.GENERATED
        assert candidate.generation == 1
        assert candidate.parent_ids == [parent_id1, parent_id2]
        assert candidate.mutation_count == 2
        assert candidate.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert candidate.updated_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert candidate.position_size_pct == Decimal("0.1")
        assert candidate.max_positions == 5
        assert candidate.min_holding_time == 300
        assert candidate.max_holding_time == 3600
        assert candidate.metadata == {"meta1": "value1"}
    def test_candidate_round_trip(self: "TestCandidateSerializers") -> None:
        """Тест полного цикла сериализации StrategyCandidate"""
        # Arrange
        original_candidate = StrategyCandidate(
            id=uuid4(),
            name="Test Candidate",
            description="Test description",
            strategy_type=StrategyType.ARBITRAGE,
            status=EvolutionStatus.GENERATED,
            generation=1,
            parent_ids=[uuid4(), uuid4()],
            mutation_count=2,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            indicators=[],
            filters=[],
            entry_rules=[],
            exit_rules=[],
            position_size_pct=Decimal("0.1"),
            max_positions=5,
            min_holding_time=300,
            max_holding_time=3600,
            metadata={"meta1": "value1"}
        )
        # Act
        model = candidate_to_model(original_candidate)
        restored_candidate = model_to_candidate(model)
        # Assert
        assert restored_candidate.id == original_candidate.id
        assert restored_candidate.name == original_candidate.name
        assert restored_candidate.description == original_candidate.description
        assert restored_candidate.strategy_type == original_candidate.strategy_type
        assert restored_candidate.status == original_candidate.status
        assert restored_candidate.generation == original_candidate.generation
        assert restored_candidate.parent_ids == original_candidate.parent_ids
        assert restored_candidate.mutation_count == original_candidate.mutation_count
        assert restored_candidate.created_at == original_candidate.created_at
        assert restored_candidate.updated_at == original_candidate.updated_at
        assert restored_candidate.position_size_pct == original_candidate.position_size_pct
        assert restored_candidate.max_positions == original_candidate.max_positions
        assert restored_candidate.min_holding_time == original_candidate.min_holding_time
        assert restored_candidate.max_holding_time == original_candidate.max_holding_time
        assert restored_candidate.metadata == original_candidate.metadata
class TestEvaluationSerializers:
    """Тесты для сериализации StrategyEvaluationResult"""
    def test_evaluation_to_model_conversion(self: "TestEvaluationSerializers") -> None:
        """Тест конвертации StrategyEvaluationResult в StrategyEvaluationModel"""
        # Arrange
        evaluation = StrategyEvaluationResult(
            id=uuid4(),
            strategy_id=uuid4(),
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            win_rate=Decimal("0.65"),
            accuracy=Decimal("0.70"),
            total_pnl=Decimal("1000.50"),
            net_pnl=Decimal("950.25"),
            profitability=Decimal("0.095"),
            profit_factor=Decimal("2.1"),
            max_drawdown=Decimal("-150.00"),
            max_drawdown_pct=Decimal("-0.15"),
            sharpe_ratio=Decimal("1.85"),
            sortino_ratio=Decimal("2.1"),
            calmar_ratio=Decimal("1.2"),
            average_trade=Decimal("9.50"),
            best_trade=Decimal("50.00"),
            worst_trade=Decimal("-25.00"),
            average_win=Decimal("15.00"),
            average_loss=Decimal("-10.00"),
            largest_win=Decimal("100.00"),
            largest_loss=Decimal("-50.00"),
            average_holding_time=1800,
            total_trading_time=86400,
            start_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            is_approved=True,
            approval_reason="Good performance",
            evaluation_time=datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"meta1": "value1"}
        )
        # Act
        model = evaluation_to_model(evaluation)
        # Assert
        assert model.id == str(evaluation.id)
        assert model.strategy_id == str(evaluation.strategy_id)
        assert model.total_trades == 100
        assert model.winning_trades == 65
        assert model.losing_trades == 35
        assert model.win_rate == "0.65"
        assert model.accuracy == "0.70"
        assert model.total_pnl == "1000.50"
        assert model.net_pnl == "950.25"
        assert model.profitability == "0.095"
        assert model.profit_factor == "2.1"
        assert model.max_drawdown == "-150.00"
        assert model.max_drawdown_pct == "-0.15"
        assert model.sharpe_ratio == "1.85"
        assert model.sortino_ratio == "2.1"
        assert model.calmar_ratio == "1.2"
        assert model.average_trade == "9.50"
        assert model.best_trade == "50.00"
        assert model.worst_trade == "-25.00"
        assert model.average_win == "15.00"
        assert model.average_loss == "-10.00"
        assert model.largest_win == "100.00"
        assert model.largest_loss == "-50.00"
        assert model.average_holding_time == 1800
        assert model.total_trading_time == 86400
        assert model.start_date == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert model.end_date == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert model.is_approved is True
        assert model.approval_reason == "Good performance"
        assert model.evaluation_time == datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
    def test_model_to_evaluation_conversion(self: "TestEvaluationSerializers") -> None:
        """Тест конвертации StrategyEvaluationModel в StrategyEvaluationResult"""
        # Arrange
        evaluation_id = uuid4()
        strategy_id = uuid4()
        model = StrategyEvaluationModel(
            id=str(evaluation_id),
            strategy_id=str(strategy_id),
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            win_rate="0.65",
            accuracy="0.70",
            total_pnl="1000.50",
            net_pnl="950.25",
            profitability="0.095",
            profit_factor="2.1",
            max_drawdown="-150.00",
            max_drawdown_pct="-0.15",
            sharpe_ratio="1.85",
            sortino_ratio="2.1",
            calmar_ratio="1.2",
            average_trade="9.50",
            best_trade="50.00",
            worst_trade="-25.00",
            average_win="15.00",
            average_loss="-10.00",
            largest_win="100.00",
            largest_loss="-50.00",
            average_holding_time=1800,
            total_trading_time=86400,
            start_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            is_approved=True,
            approval_reason="Good performance",
            fitness_score="0.85",
            evaluation_time=datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
            meta_data=json.dumps({"meta1": "value1"})
        )
        # Act
        evaluation = model_to_evaluation(model)
        # Assert
        assert evaluation.id == evaluation_id
        assert evaluation.strategy_id == strategy_id
        assert evaluation.total_trades == 100
        assert evaluation.winning_trades == 65
        assert evaluation.losing_trades == 35
        assert evaluation.win_rate == Decimal("0.65")
        assert evaluation.accuracy == Decimal("0.70")
        assert evaluation.total_pnl == Decimal("1000.50")
        assert evaluation.net_pnl == Decimal("950.25")
        assert evaluation.profitability == Decimal("0.095")
        assert evaluation.profit_factor == Decimal("2.1")
        assert evaluation.max_drawdown == Decimal("-150.00")
        assert evaluation.max_drawdown_pct == Decimal("-0.15")
        assert evaluation.sharpe_ratio == Decimal("1.85")
        assert evaluation.sortino_ratio == Decimal("2.1")
        assert evaluation.calmar_ratio == Decimal("1.2")
        assert evaluation.average_trade == Decimal("9.50")
        assert evaluation.best_trade == Decimal("50.00")
        assert evaluation.worst_trade == Decimal("-25.00")
        assert evaluation.average_win == Decimal("15.00")
        assert evaluation.average_loss == Decimal("-10.00")
        assert evaluation.largest_win == Decimal("100.00")
        assert evaluation.largest_loss == Decimal("-50.00")
        assert evaluation.average_holding_time == 1800
        assert evaluation.total_trading_time == 86400
        assert evaluation.start_date == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert evaluation.end_date == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert evaluation.is_approved is True
        assert evaluation.approval_reason == "Good performance"
        assert evaluation.evaluation_time == datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        assert evaluation.metadata == {"meta1": "value1"}
class TestContextSerializers:
    """Тесты для сериализации EvolutionContext"""
    def test_context_to_model_conversion(self: "TestContextSerializers") -> None:
        """Тест конвертации EvolutionContext в EvolutionContextModel"""
        # Arrange
        context = EvolutionContext(
            id=uuid4(),
            name="Test Context",
            description="Test evolution context",
            population_size=100,
            generations=50,
            mutation_rate=Decimal("0.1"),
            crossover_rate=Decimal("0.8"),
            elite_size=5,
            min_accuracy=Decimal("0.9"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.15"),
            min_sharpe=Decimal("1.0"),
            max_indicators=10,
            max_filters=5,
            max_entry_rules=3,
            max_exit_rules=3,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"meta1": "value1"}
        )
        # Act
        model = context_to_model(context)
        # Assert
        assert model.id == str(context.id)
        assert model.name == "Test Context"
        assert model.description == "Test evolution context"
        assert model.population_size == 100
        assert model.generations == 50
        assert model.mutation_rate == "0.1"
        assert model.crossover_rate == "0.8"
        assert model.elite_size == 5
        assert model.min_accuracy == "0.9"
        assert model.min_profitability == "0.05"
        assert model.max_drawdown == "0.15"
        assert model.min_sharpe == "1.0"
        assert model.max_indicators == 10
        assert model.max_filters == 5
        assert model.max_entry_rules == 3
        assert model.max_exit_rules == 3
        assert model.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert model.updated_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    def test_model_to_context_conversion(self: "TestContextSerializers") -> None:
        """Тест конвертации EvolutionContextModel в EvolutionContext"""
        # Arrange
        context_id = uuid4()
        model = EvolutionContextModel(
            id=str(context_id),
            name="Test Context",
            description="Test evolution context",
            population_size=100,
            generations=50,
            mutation_rate="0.1",
            crossover_rate="0.8",
            elite_size=5,
            min_accuracy="0.9",
            min_profitability="0.05",
            max_drawdown="0.15",
            min_sharpe="1.0",
            max_indicators=10,
            max_filters=5,
            max_entry_rules=3,
            max_exit_rules=3,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            meta_data=json.dumps({"meta1": "value1"})
        )
        # Act
        context = model_to_context(model)
        # Assert
        assert context.id == context_id
        assert context.name == "Test Context"
        assert context.description == "Test evolution context"
        assert context.population_size == 100
        assert context.generations == 50
        assert context.mutation_rate == Decimal("0.1")
        assert context.crossover_rate == Decimal("0.8")
        assert context.elite_size == 5
        assert context.min_accuracy == Decimal("0.9")
        assert context.min_profitability == Decimal("0.05")
        assert context.max_drawdown == Decimal("0.15")
        assert context.min_sharpe == Decimal("1.0")
        assert context.max_indicators == 10
        assert context.max_filters == 5
        assert context.max_entry_rules == 3
        assert context.max_exit_rules == 3
        assert context.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert context.updated_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert context.metadata == {"meta1": "value1"}
    def test_context_round_trip(self: "TestContextSerializers") -> None:
        """Тест полного цикла сериализации EvolutionContext"""
        # Arrange
        original_context = EvolutionContext(
            id=uuid4(),
            name="Test Context",
            description="Test evolution context",
            population_size=100,
            generations=50,
            mutation_rate=Decimal("0.1"),
            crossover_rate=Decimal("0.8"),
            elite_size=5,
            min_accuracy=Decimal("0.9"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.15"),
            min_sharpe=Decimal("1.0"),
            max_indicators=10,
            max_filters=5,
            max_entry_rules=3,
            max_exit_rules=3,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"meta1": "value1"}
        )
        # Act
        model = context_to_model(original_context)
        restored_context = model_to_context(model)
        # Assert
        assert restored_context.id == original_context.id
        assert restored_context.name == original_context.name
        assert restored_context.description == original_context.description
        assert restored_context.population_size == original_context.population_size
        assert restored_context.generations == original_context.generations
        assert restored_context.mutation_rate == original_context.mutation_rate
        assert restored_context.crossover_rate == original_context.crossover_rate
        assert restored_context.elite_size == original_context.elite_size
        assert restored_context.min_accuracy == original_context.min_accuracy
        assert restored_context.min_profitability == original_context.min_profitability
        assert restored_context.max_drawdown == original_context.max_drawdown
        assert restored_context.min_sharpe == original_context.min_sharpe
        assert restored_context.max_indicators == original_context.max_indicators
        assert restored_context.max_filters == original_context.max_filters
        assert restored_context.max_entry_rules == original_context.max_entry_rules
        assert restored_context.max_exit_rules == original_context.max_exit_rules
        assert restored_context.created_at == original_context.created_at
        assert restored_context.updated_at == original_context.updated_at
        assert restored_context.metadata == original_context.metadata
class TestSerializersEdgeCases:
    """Тесты граничных случаев для сериализаторов"""
    def test_candidate_with_complex_metadata(self: "TestSerializersEdgeCases") -> None:
        """Тест сериализации StrategyCandidate со сложными метаданными"""
        # Arrange
        complex_metadata = {
            "nested_dict": {"key1": {"key2": [1, 2, 3]}},
            "list_of_dicts": [{"a": 1}, {"b": 2}],
            "mixed_types": [1, "string", 3.14, True, None]
        }
        candidate = StrategyCandidate(
            id=uuid4(),
            name="Test Candidate",
            description="Test description",
            strategy_type=StrategyType.ARBITRAGE,
            status=EvolutionStatus.GENERATED,
            generation=1,
            parent_ids=[uuid4(), uuid4()],
            mutation_count=2,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            indicators=[],
            filters=[],
            entry_rules=[],
            exit_rules=[],
            position_size_pct=Decimal("0.1"),
            max_positions=5,
            min_holding_time=300,
            max_holding_time=3600,
            metadata=complex_metadata
        )
        # Act
        model = candidate_to_model(candidate)
        restored_candidate = model_to_candidate(model)
        # Assert
        assert restored_candidate.metadata == complex_metadata
    def test_evaluation_with_extreme_values(self: "TestSerializersEdgeCases") -> None:
        """Тест сериализации StrategyEvaluationResult с экстремальными значениями"""
        # Arrange
        evaluation = StrategyEvaluationResult(
            id=uuid4(),
            strategy_id=uuid4(),
            total_trades=10000,
            winning_trades=9999,
            losing_trades=1,
            win_rate=Decimal("0.9999"),
            accuracy=Decimal("0.9999"),
            total_pnl=Decimal("999999.99"),
            net_pnl=Decimal("999999.99"),
            profitability=Decimal("0.9999"),
            profit_factor=Decimal("999.9"),
            max_drawdown=Decimal("-0.01"),
            max_drawdown_pct=Decimal("-0.0001"),
            sharpe_ratio=Decimal("99.9"),
            sortino_ratio=Decimal("99.9"),
            calmar_ratio=Decimal("99.9"),
            average_trade=Decimal("999.99"),
            best_trade=Decimal("9999.99"),
            worst_trade=Decimal("-0.01"),
            average_win=Decimal("999.99"),
            average_loss=Decimal("-0.01"),
            largest_win=Decimal("99999.99"),
            largest_loss=Decimal("-0.01"),
            average_holding_time=86400,
            total_trading_time=31536000,
            start_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, 12, 0, 0, tzinfo=timezone.utc),
            is_approved=True,
            approval_reason="Excellent performance",
            evaluation_time=datetime(2024, 12, 31, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"meta1": "value1"}
        )
        # Act
        model = evaluation_to_model(evaluation)
        restored_evaluation = model_to_evaluation(model)
        # Assert
        assert restored_evaluation.total_trades == 10000
        assert restored_evaluation.winning_trades == 9999
        assert restored_evaluation.losing_trades == 1
        assert restored_evaluation.win_rate == Decimal("0.9999")
        assert restored_evaluation.accuracy == Decimal("0.9999")
        assert restored_evaluation.total_pnl == Decimal("999999.99")
        assert restored_evaluation.net_pnl == Decimal("999999.99")
        assert restored_evaluation.profitability == Decimal("0.9999")
        assert restored_evaluation.profit_factor == Decimal("999.9")
        assert restored_evaluation.max_drawdown == Decimal("-0.01")
        assert restored_evaluation.max_drawdown_pct == Decimal("-0.0001")
        assert restored_evaluation.sharpe_ratio == Decimal("99.9")
        assert restored_evaluation.sortino_ratio == Decimal("99.9")
        assert restored_evaluation.calmar_ratio == Decimal("99.9")
        assert restored_evaluation.average_trade == Decimal("999.99")
        assert restored_evaluation.best_trade == Decimal("9999.99")
        assert restored_evaluation.worst_trade == Decimal("-0.01")
        assert restored_evaluation.average_win == Decimal("999.99")
        assert restored_evaluation.average_loss == Decimal("-0.01")
        assert restored_evaluation.largest_win == Decimal("99999.99")
        assert restored_evaluation.largest_loss == Decimal("-0.01")
        assert restored_evaluation.average_holding_time == 86400
        assert restored_evaluation.total_trading_time == 31536000 
