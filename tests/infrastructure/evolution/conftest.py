"""
Фикстуры для тестов infrastructure/evolution модуля.
"""
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Generator
from uuid import uuid4
import pytest
from domain.evolution.strategy_fitness import StrategyEvaluationResult, TradeResult
from domain.evolution.strategy_model import (
    EvolutionContext, EvolutionStatus, StrategyCandidate,
    EntryRule, ExitRule, FilterConfig, FilterType,
    IndicatorConfig, IndicatorType, SignalType, StrategyType
)
from infrastructure.evolution.cache import EvolutionCache
from infrastructure.evolution.exceptions import (
    BackupError, CacheError, ConnectionError, MigrationError,
    QueryError, SerializationError, StorageError, ValidationError
)
from infrastructure.evolution.models import (
    EvolutionContextModel, StrategyCandidateModel, StrategyEvaluationModel
)
from infrastructure.evolution.storage import StrategyStorage


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Временный путь к БД для тестов."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Очистка после тестов
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def storage(temp_db_path: str) -> Generator[StrategyStorage, None, None]:
    """Фикстура хранилища стратегий."""
    storage = StrategyStorage(temp_db_path)
    yield storage


@pytest.fixture
def cache() -> Generator[EvolutionCache, None, None]:
    """Фикстура кэша эволюции."""
    cache = EvolutionCache({
        "cache_size": 100,
        "cache_ttl": 60,
        "cache_strategy": "lru"
    })
    yield cache


@pytest.fixture
def sample_indicator() -> IndicatorConfig:
    """Образец индикатора."""
    return IndicatorConfig(
        id=uuid4(),
        name="SMA",
        indicator_type=IndicatorType.TREND,
        parameters={"period": 20},
        weight=Decimal("1.0"),
        is_active=True
    )


@pytest.fixture
def sample_filter() -> FilterConfig:
    """Образец фильтра."""
    return FilterConfig(
        id=uuid4(),
        name="Volatility Filter",
        filter_type=FilterType.VOLATILITY,
        parameters={"min_atr": 0.01, "max_atr": 0.05},
        threshold=Decimal("0.5"),
        is_active=True
    )


@pytest.fixture
def sample_entry_rule() -> EntryRule:
    """Образец правила входа."""
    return EntryRule(
        id=uuid4(),
        conditions=[
            {
                "indicator": "SMA",
                "condition": "above",
                "period": 20,
                "threshold": 0.0,
                "direction": "up",
                "operator": "gt",
                "value": 0.0
            }
        ],
        signal_type=SignalType.BUY,
        confidence_threshold=Decimal("0.7"),
        volume_ratio=Decimal("1.0"),
        is_active=True
    )


@pytest.fixture
def sample_exit_rule() -> ExitRule:
    """Образец правила выхода."""
    return ExitRule(
        id=uuid4(),
        conditions=[
            {
                "indicator": "SMA",
                "condition": "below",
                "period": 20,
                "threshold": 0.0,
                "operator": "lt",
                "value": 0.0
            }
        ],
        signal_type=SignalType.SELL,
        stop_loss_pct=Decimal("0.02"),
        take_profit_pct=Decimal("0.04"),
        trailing_stop=False,
        trailing_distance=Decimal("0.01"),
        is_active=True
    )


@pytest.fixture
def sample_candidate(
    sample_indicator: IndicatorConfig,
    sample_filter: FilterConfig,
    sample_entry_rule: EntryRule,
    sample_exit_rule: ExitRule
) -> StrategyCandidate:
    """Образец кандидата стратегии."""
    return StrategyCandidate(
        id=uuid4(),
        name="Test Strategy",
        description="Test strategy for unit tests",
        strategy_type=StrategyType.TREND,  # Исправлено: используем правильное значение enum
        status=EvolutionStatus.GENERATED,
        indicators=[sample_indicator],
        filters=[sample_filter],
        entry_rules=[sample_entry_rule],
        exit_rules=[sample_exit_rule],
        position_size_pct=Decimal("0.1"),
        max_positions=3,
        min_holding_time=60,
        max_holding_time=86400,
        generation=1,
        parent_ids=[uuid4()],
        mutation_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"test": True}
    )


@pytest.fixture
def sample_trade() -> TradeResult:
    """Образец торговой сделки."""
    return TradeResult(
        id=uuid4(),
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now(),
        entry_price=Decimal("50000"),
        exit_price=Decimal("51000"),
        quantity=Decimal("0.1"),
        pnl=Decimal("100"),
        pnl_pct=Decimal("0.02"),
        commission=Decimal("1"),
        signal_type="buy",
        holding_time=3600,
        success=True,
        metadata={"test": True}
    )


@pytest.fixture
def sample_evaluation(sample_candidate: StrategyCandidate, sample_trade: TradeResult) -> StrategyEvaluationResult:
    """Образец результата оценки."""
    evaluation = StrategyEvaluationResult(
        id=uuid4(),
        strategy_id=sample_candidate.id,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=Decimal("1.0"),
        accuracy=Decimal("1.0"),
        total_pnl=Decimal("100"),
        net_pnl=Decimal("99"),
        profitability=Decimal("0.02"),
        profit_factor=Decimal("10.0"),
        max_drawdown=Decimal("0"),
        max_drawdown_pct=Decimal("0"),
        sharpe_ratio=Decimal("2.0"),
        sortino_ratio=Decimal("2.5"),
        calmar_ratio=Decimal("1.5"),
        average_trade=Decimal("99"),
        best_trade=Decimal("99"),
        worst_trade=Decimal("99"),
        average_win=Decimal("99"),
        average_loss=Decimal("0"),
        largest_win=Decimal("99"),
        largest_loss=Decimal("0"),
        average_holding_time=3600,
        total_trading_time=3600,
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now(),
        is_approved=True,
        approval_reason="All criteria met",
        evaluation_time=datetime.now(),
        metadata={"test": True}
    )
    evaluation.trades = [sample_trade]
    return evaluation


@pytest.fixture
def sample_context() -> EvolutionContext:
    """Образец контекста эволюции."""
    return EvolutionContext(
        id=uuid4(),
        name="Test Evolution Context",
        description="Test context for unit tests",
        population_size=50,
        generations=100,
        mutation_rate=Decimal("0.1"),
        crossover_rate=Decimal("0.8"),
        elite_size=5,
        min_accuracy=Decimal("0.8"),
        min_profitability=Decimal("0.05"),
        max_drawdown=Decimal("0.15"),
        min_sharpe=Decimal("1.0"),
        max_indicators=10,
        max_filters=5,
        max_entry_rules=3,
        max_exit_rules=3,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"test": True}
    )


@pytest.fixture
def temp_backup_dir() -> Generator[Path, None, None]:
    """Временная директория для бэкапов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_dir = Path(temp_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        yield backup_dir


@pytest.fixture
def temp_migration_dir() -> Generator[Path, None, None]:
    """Временная директория для миграций."""
    with tempfile.TemporaryDirectory() as temp_dir:
        migration_dir = Path(temp_dir) / "migrations"
        migration_dir.mkdir(parents=True, exist_ok=True)
        yield migration_dir


@pytest.fixture
def sample_migration_data() -> dict:
    """Образец данных миграции."""
    return {
        "version": "1.0",
        "description": "Test migration",
        "scripts": [
            "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
        ],
        "rollback_scripts": [
            "DROP TABLE IF EXISTS test_table"
        ],
        "rollback_supported": True,
        "dependencies": []
    }


@pytest.fixture
def mock_engine() -> Any:
    """Mock для SQLAlchemy engine."""
    class MockEngine:
        def __init__(self) -> Any:
            self.closed = False
        def dispose(self) -> Any:
            self.closed = True
    return MockEngine()


@pytest.fixture
def mock_session() -> Any:
    """Mock для SQLModel Session."""
    class MockSession:
        def __init__(self) -> Any:
            self.committed = False
            self.rolled_back = False
            self.closed = False
        def commit(self) -> Any:
            self.committed = True
        def rollback(self) -> Any:
            self.rolled_back = True
        def close(self) -> Any:
            self.closed = True
        def __enter__(self) -> Any:
            return self
        def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
            self.close()
    return MockSession() 
