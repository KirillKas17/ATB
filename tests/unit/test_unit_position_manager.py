"""
Unit тесты для PositionManager.
Тестирует управление позициями, включая открытие, закрытие,
отслеживание P&L и управление рисками позиций.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from infrastructure.core.position_manager import PositionManager, Position, PositionSide, PositionStatus

class TestPositionManager:
    """Тесты для PositionManager."""
    
    @pytest.fixture
    def position_manager(self) -> PositionManager:
        """Фикстура для PositionManager."""
        config = {
            "max_positions": 10,
            "max_position_size": 1.0,
            "min_position_size": 0.01,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "default_leverage": 1.0,
            "max_leverage": 10.0,
            "min_leverage": 1.0,
            "min_diversification": 3,
            "max_correlation": 0.8
        }
        return PositionManager(config)
    
    @pytest.fixture
    def sample_position(self) -> Position:
        """Фикстура с тестовой позицией."""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=1.5,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            entry_time=datetime.now(),
            leverage=1.0,
            margin=75000.0,
            fees=0.0,
            tags=[],
            metadata={}
        )
    
    @pytest.fixture
    def sample_positions_list(self) -> list:
        """Фикстура со списком тестовых позиций."""
        return [
            Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                size=1.5,
                entry_price=50000.0,
                stop_loss=48000.0,
                take_profit=55000.0,
                entry_time=datetime.now() - timedelta(hours=1),
                leverage=1.0,
                margin=75000.0,
                fees=0.0,
                tags=[],
                metadata={}
            ),
            Position(
                symbol="ETHUSDT",
                side=PositionSide.SHORT,
                size=10.0,
                entry_price=3000.0,
                stop_loss=3200.0,
                take_profit=2800.0,
                entry_time=datetime.now() - timedelta(minutes=30),
                leverage=1.0,
                margin=30000.0,
                fees=0.0,
                tags=[],
                metadata={}
            )
        ]
    
    def test_initialization(self, position_manager: PositionManager) -> None:
        """Тест инициализации менеджера позиций."""
        assert position_manager is not None
        assert hasattr(position_manager, 'positions')
        assert hasattr(position_manager, 'position_history')
        assert hasattr(position_manager, 'config')
        assert hasattr(position_manager, 'max_positions')
    
    def test_open_position(self, position_manager: PositionManager) -> None:
        """Тест открытия позиции."""
        # Открытие позиции
        position = position_manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            leverage=2.0
        )
        # Проверки
        assert position is not None
        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.size == 1.0
        assert position.entry_price == 50000.0
        assert position.status == PositionStatus.OPEN
        assert position.entry_time is not None
    
    def test_close_position(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест закрытия позиции."""
        # Добавление позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        
        # Закрытие позиции
        close_result = position_manager.close_position(position_id, 53000.0)
        
        # Проверки
        assert close_result is not None
        assert close_result.exit_price == 53000.0
        assert close_result.status == PositionStatus.CLOSED
        assert close_result.exit_time is not None
        assert close_result.pnl is not None
    
    def test_update_position(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест обновления позиции."""
        # Добавление позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        
        # Обновление позиции
        update_result = position_manager.update_position(position_id, 53000.0)
        
        # Проверки
        assert update_result is not None
        assert update_result.pnl is not None
        # current_price не является атрибутом Position в этом классе
    
    def test_get_position(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест получения позиции."""
        # Добавление позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        
        # Получение позиции
        retrieved_position = position_manager.get_position(position_id)
        
        # Проверки
        assert retrieved_position is not None
        assert retrieved_position.symbol == sample_position.symbol
        assert retrieved_position.side == sample_position.side
        assert retrieved_position.size == sample_position.size
    
    def test_get_positions(self, position_manager: PositionManager, sample_positions_list: list) -> None:
        """Тест получения списка позиций."""
        # Добавление позиций
        for position in sample_positions_list:
            position_manager.add_position(position)
        
        # Получение всех позиций
        all_positions = position_manager.get_positions()
        
        # Проверки
        assert all_positions is not None
        assert isinstance(all_positions, list)
        assert len(all_positions) >= len(sample_positions_list)
    
    def test_get_positions_by_symbol(self, position_manager: PositionManager, sample_positions_list: list) -> None:
        """Тест получения позиций по символу."""
        # Добавление позиций
        for position in sample_positions_list:
            position_manager.add_position(position)
        
        # Получение позиций по символу
        btc_positions = position_manager.get_positions_by_symbol("BTCUSDT")
        eth_positions = position_manager.get_positions_by_symbol("ETHUSDT")
        
        # Проверки
        assert btc_positions is not None
        assert eth_positions is not None
        assert isinstance(btc_positions, list)
        assert isinstance(eth_positions, list)
        
        # Проверка фильтрации
        for position in btc_positions:
            assert position.symbol == "BTCUSDT"
        for position in eth_positions:
            assert position.symbol == "ETHUSDT"
    
    def test_get_open_positions(self, position_manager: PositionManager, sample_positions_list: list) -> None:
        """Тест получения открытых позиций."""
        # Добавление позиций
        for position in sample_positions_list:
            position_manager.add_position(position)
        
        # Получение открытых позиций
        open_positions = position_manager.get_open_positions()
        
        # Проверки
        assert open_positions is not None
        assert isinstance(open_positions, list)
        for position in open_positions:
            assert position.status == PositionStatus.OPEN
    
    def test_get_closed_positions(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест получения закрытых позиций."""
        # Добавление и закрытие позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        position_manager.close_position(position_id, 53000.0)
        
        # Получение закрытых позиций
        closed_positions = position_manager.get_closed_positions()
        
        # Проверки
        assert closed_positions is not None
        assert isinstance(closed_positions, list)
        for position in closed_positions:
            assert position.status == PositionStatus.CLOSED
    
    def test_get_total_pnl(self, position_manager: PositionManager, sample_positions_list: list) -> None:
        """Тест получения общего P&L."""
        # Добавление позиций
        for position in sample_positions_list:
            position_manager.add_position(position)
        
        # Получение общего P&L
        total_pnl = position_manager.get_total_pnl()
        
        # Проверки
        assert isinstance(total_pnl, float)
    
    def test_get_win_rate(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест получения процента прибыльных сделок."""
        # Добавление и закрытие позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        position_manager.close_position(position_id, 53000.0)  # Прибыльная сделка
        
        # Получение процента прибыльных сделок
        win_rate = position_manager.get_win_rate()
        
        # Проверки
        assert isinstance(win_rate, float)
        assert 0.0 <= win_rate <= 1.0
    
    def test_get_average_roi(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест получения среднего ROI."""
        # Добавление и закрытие позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        position_manager.close_position(position_id, 53000.0)
        
        # Получение среднего ROI
        avg_roi = position_manager.get_average_roi()
        
        # Проверки
        assert isinstance(avg_roi, float)
    
    def test_get_max_drawdown(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест получения максимальной просадки."""
        # Добавление и закрытие позиции
        position_manager.add_position(sample_position)
        position_id = position_manager._generate_position_id(sample_position)
        position_manager.close_position(position_id, 53000.0)
        
        # Получение максимальной просадки
        max_drawdown = position_manager.get_max_drawdown()
        
        # Проверки
        assert isinstance(max_drawdown, float)
        assert max_drawdown >= 0.0
    
    def test_position_limits(self, position_manager: PositionManager) -> None:
        """Тест лимитов позиций."""
        # Попытка открыть позицию с превышением лимита размера
        result = position_manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=2.0,  # Превышает max_position_size
            entry_price=50000.0
        )
        
        # Проверки
        assert result is None  # Должно вернуть None из-за превышения лимита
    
    def test_risk_limits(self, position_manager: PositionManager) -> None:
        """Тест лимитов риска."""
        # Попытка открыть позицию с превышением лимита плеча
        result = position_manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            leverage=20.0  # Превышает max_leverage
        )
        
        # Проверки
        assert result is None  # Должно вернуть None из-за превышения лимита
    
    def test_position_correlation(self, position_manager: PositionManager, sample_positions_list: list) -> None:
        """Тест расчета корреляции между позициями."""
        # Добавление позиций
        for position in sample_positions_list:
            position_manager.add_position(position)
        
        # Расчет корреляции
        correlation = position_manager.get_position_correlation("BTCUSDT", "ETHUSDT")
        
        # Проверки
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0
    
    def test_error_handling(self, position_manager: PositionManager) -> None:
        """Тест обработки ошибок."""
        # Попытка закрыть несуществующую позицию
        result = position_manager.close_position("nonexistent_id", 50000.0)
        
        # Проверки
        assert result is None  # Должно вернуть None для несуществующей позиции
    
    def test_edge_cases(self, position_manager: PositionManager) -> None:
        """Тест граничных случаев."""
        # Попытка открыть позицию с None в качестве символа
        result = position_manager.open_position(
            symbol=None,  # type: ignore
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0
        )
        
        # Проверки
        assert result is None  # Должно вернуть None из-за некорректных параметров
    
    def test_position_metrics(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест метрик позиции."""
        # Добавление позиции
        position_manager.add_position(sample_position)
        
        # Проверка метрик
        assert hasattr(position_manager, 'risk_metrics')
        assert isinstance(position_manager.risk_metrics, dict)
    
    def test_position_validation(self, position_manager: PositionManager) -> None:
        """Тест валидации позиции."""
        # Проверка лимитов позиции
        result = position_manager._check_position_limits("BTCUSDT", 0.5)
        
        # Проверки
        assert isinstance(result, bool)
    
    def test_position_cleanup(self, position_manager: PositionManager, sample_position: Position) -> None:
        """Тест очистки позиций."""
        # Добавление позиции
        position_manager.add_position(sample_position)
        initial_count = len(position_manager.get_positions())
        
        # Удаление позиции
        position_manager.remove_position(sample_position.symbol)
        final_count = len(position_manager.get_positions())
        
        # Проверки
        assert final_count < initial_count 
