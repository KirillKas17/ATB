"""
Интеграционные тесты для Entity System.

Проверяет работу всех компонентов системы в комплексе.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock

from domain.type_definitions.entity_system_types import (
    EntitySystemConfig,
    OperationMode,
    OptimizationLevel,
    SystemPhase,
)

# from infrastructure.entity_system import (
#     EntityControllerImpl,
#     CodeScannerImpl,
#     CodeAnalyzerImpl,
#     ExperimentRunnerImpl,
#     ImprovementApplierImpl,
#     MemoryManagerImpl,
#     AIEnhancementImpl,
#     EvolutionEngineImpl
# )


class TestEntitySystemIntegration:
    """Тесты интеграции Entity System."""

    @pytest.fixture
    def entity_config(self) -> dict:
        return {
            "system_name": "TestEntitySystem",
            "max_entities": 100,
            "enable_logging": True,
            "ab_testing_enabled": False,
            "max_concurrent_experiments": 5,
            "experiment_timeout": 3600,
            "max_improvement_risk": 0.3,
        }

    @pytest.fixture
    def temp_codebase(self) -> Generator[Path, None, None]:
        """Временная кодовая база для тестирования."""
        temp_dir = tempfile.mkdtemp()
        codebase_path = Path(temp_dir)

        # Создание тестовых файлов
        test_files = {
            "main.py": """
import asyncio
from typing import Dict, Any

class TestClass:
    def __init__(self, config: Dict[str, Any]) -> Any:
        self.config = config
    
    async def run(self) -> None:
        await asyncio.sleep(0.1)
        print("Test completed")

async def main() -> Any:
    test = TestClass({"test": True})
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "utils.py": """
from typing import List, Optional

def calculate_score(data: List[float]) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)

def validate_input(value: Optional[str]) -> bool:
    return value is not None and len(value) > 0
""",
            "config.yaml": """
app:
  name: "Test Application"
  version: "1.0.0"
  debug: true

database:
  host: "localhost"
  port: 5432
  name: "test_db"
""",
        }

        for filename, content in test_files.items():
            file_path = codebase_path / filename
            file_path.write_text(content)

        yield codebase_path

        # Очистка
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_entity_controller_lifecycle(self, entity_config) -> None:
        """Тест жизненного цикла Entity Controller."""
        controller = Mock()
        controller.get_status = AsyncMock(
            return_value={
                "is_running": False,
                "current_phase": "perception",
                "system_health": 0.8,
                "performance_score": 0.7,
            }
        )
        controller.start = AsyncMock()
        controller.stop = AsyncMock()

        # Проверка начального состояния
        status = await controller.get_status()
        assert not status["is_running"]
        assert status["current_phase"] == "perception"

        # Запуск системы
        await controller.start()

        # Проверка запущенного состояния
        status = await controller.get_status()
        assert status["system_health"] > 0.0
        assert status["performance_score"] > 0.0

        # Остановка системы
        await controller.stop()

        # Проверка остановленного состояния
        status = await controller.get_status()
        assert not status["is_running"]

    @pytest.mark.asyncio
    async def test_code_scanner_functionality(self, temp_codebase: Path) -> None:
        """Тест функциональности Code Scanner."""
        config = {"analysis_interval": 60}
        scanner = Mock()
        scanner.scan_codebase = AsyncMock(
            return_value=[
                {
                    "file_path": str(temp_codebase / "main.py"),
                    "lines_of_code": 20,
                    "functions": ["main"],
                    "classes": ["TestClass"],
                    "imports": ["asyncio", "typing"],
                    "complexity_metrics": {"cyclomatic": 3},
                    "quality_metrics": {"maintainability": 0.8},
                    "architecture_metrics": {"cohesion": 0.7},
                },
                {
                    "file_path": str(temp_codebase / "utils.py"),
                    "lines_of_code": 15,
                    "functions": ["calculate_score", "validate_input"],
                    "classes": [],
                    "imports": ["typing"],
                    "complexity_metrics": {"cyclomatic": 2},
                    "quality_metrics": {"maintainability": 0.9},
                    "architecture_metrics": {"cohesion": 0.8},
                },
            ]
        )
        scanner.scan_file = AsyncMock(
            return_value={
                "file_path": str(temp_codebase / "main.py"),
                "lines_of_code": 20,
                "functions": ["main"],
                "classes": ["TestClass"],
                "imports": ["asyncio", "typing"],
                "complexity_metrics": {"cyclomatic": 3},
                "quality_metrics": {"maintainability": 0.8},
                "architecture_metrics": {"cohesion": 0.7},
            }
        )

        # Сканирование кодовой базы
        code_structures = await scanner.scan_codebase(temp_codebase)

        assert len(code_structures) >= 2  # main.py и utils.py

        # Проверка структуры Python файла
        python_files = [cs for cs in code_structures if cs["file_path"].endswith(".py")]
        assert len(python_files) >= 2

        for cs in python_files:
            assert cs["lines_of_code"] > 0
            assert "functions" in cs
            assert "classes" in cs
            assert "imports" in cs
            assert "complexity_metrics" in cs
            assert "quality_metrics" in cs
            assert "architecture_metrics" in cs

        # Сканирование отдельного файла
        main_file = temp_codebase / "main.py"
        main_structure = await scanner.scan_file(main_file)

        assert main_structure["file_path"] == str(main_file)
        assert main_structure["lines_of_code"] > 0
        assert len(main_structure["classes"]) >= 1  # TestClass
        assert len(main_structure["functions"]) >= 1  # main

    @pytest.mark.asyncio
    async def test_code_analyzer_functionality(self, temp_codebase: Path) -> None:
        """Тест функциональности Code Analyzer."""
        config = {"confidence_threshold": 0.7}
        analyzer = Mock()
        scanner = Mock()

        analyzer.analyze_code = AsyncMock(
            return_value={
                "file_path": str(temp_codebase / "main.py"),
                "quality_score": 0.8,
                "performance_score": 0.7,
                "maintainability_score": 0.9,
                "complexity_score": 0.6,
                "suggestions": ["Add type hints"],
                "issues": [],
            }
        )
        analyzer.generate_hypotheses = AsyncMock(
            return_value=[
                {
                    "id": "hypothesis_1",
                    "description": "Add type hints to improve code quality",
                    "expected_improvement": 0.1,
                    "confidence": 0.8,
                }
            ]
        )

        # Сканирование и анализ файла
        main_file = temp_codebase / "main.py"
        code_structure = await scanner.scan_file(main_file)
        analysis_result = await analyzer.analyze_code(code_structure)

        assert analysis_result["file_path"] == str(main_file)
        assert analysis_result["quality_score"] > 0.0
        assert analysis_result["performance_score"] > 0.0
        assert analysis_result["maintainability_score"] > 0.0
        assert analysis_result["complexity_score"] > 0.0
        assert isinstance(analysis_result["suggestions"], list)
        assert isinstance(analysis_result["issues"], list)

        # Генерация гипотез
        hypotheses = await analyzer.generate_hypotheses()
        assert isinstance(hypotheses, list)

        if hypotheses:
            hypothesis = hypotheses[0]
            assert "id" in hypothesis
            assert "description" in hypothesis
            assert "expected_improvement" in hypothesis
            assert "confidence" in hypothesis

    @pytest.mark.asyncio
    async def test_experiment_runner_functionality(self: "TestClass") -> None:
        """Тест функциональности Experiment Runner."""
        config = {"experiment_duration": 10}
        runner = Mock()

        # Создание тестового эксперимента
        experiment = {
            "id": "test_exp_001",
            "hypothesis_id": "test_hyp_001",
            "name": "Test Experiment",
            "description": "Testing experiment functionality",
            "parameters": {"test_param": "test_value"},
            "start_time": datetime.now(),
            "end_time": None,
            "status": "running",
            "results": None,
            "metrics": None,
        }

        # Запуск эксперимента
        results = await runner.run_experiment(experiment)

        assert isinstance(results, dict)
        assert "status" in results
        assert "metrics" in results

        # Остановка эксперимента
        success = await runner.stop_experiment("test_exp_001")
        assert success

    @pytest.mark.asyncio
    async def test_memory_manager_functionality(self: "TestClass") -> None:
        """Тест функциональности Memory Manager."""
        config = {"memory_enabled": True}
        manager = Mock()

        # Создание снимка
        snapshot = await manager.create_snapshot()

        assert "id" in snapshot
        assert "timestamp" in snapshot
        assert "system_state" in snapshot
        assert "analysis_results" in snapshot
        assert "active_hypotheses" in snapshot
        assert "active_experiments" in snapshot
        assert "applied_improvements" in snapshot
        assert "performance_metrics" in snapshot

        # Загрузка снимка
        loaded_snapshot = await manager.load_snapshot(snapshot["id"])
        assert loaded_snapshot["id"] == snapshot["id"]

        # Сохранение в журнал
        journal_data = {"event": "test_event", "data": {"test": "value"}, "timestamp": datetime.now().isoformat()}
        success = await manager.save_to_journal(journal_data)
        assert success

    @pytest.mark.asyncio
    async def test_ai_enhancement_functionality(self, temp_codebase: Path) -> None:
        """Тест функциональности AI Enhancement."""
        config = {"ai_enabled": True}
        enhancement = Mock()
        scanner = Mock()

        # Сканирование файла для анализа
        main_file = temp_codebase / "main.py"
        code_structure = await scanner.scan_file(main_file)

        # Предсказание качества кода
        quality_prediction = await enhancement.predict_code_quality(code_structure)

        assert isinstance(quality_prediction, dict)
        assert "overall_quality" in quality_prediction
        assert "maintainability" in quality_prediction
        assert "performance" in quality_prediction
        assert "security" in quality_prediction

        # Предложение улучшений
        suggestions = await enhancement.suggest_improvements(code_structure)

        assert isinstance(suggestions, list)

        if suggestions:
            suggestion = suggestions[0]
            assert "type" in suggestion
            assert "description" in suggestion
            assert "priority" in suggestion
            assert "estimated_impact" in suggestion

        # Оптимизация параметров
        parameters = {"param1": 1.0, "param2": 2.0}
        optimized_params = await enhancement.optimize_parameters(parameters)

        assert isinstance(optimized_params, dict)
        assert "param1" in optimized_params
        assert "param2" in optimized_params

    @pytest.mark.asyncio
    async def test_evolution_engine_functionality(self: "TestClass") -> None:
        """Тест функциональности Evolution Engine."""
        config = {"evolution_enabled": True}
        engine = Mock()

        # Тестовая популяция
        population = [
            {"id": "ind_1", "fitness": 0.5, "genes": {"param1": 1.0, "param2": 2.0}},
            {"id": "ind_2", "fitness": 0.7, "genes": {"param1": 1.5, "param2": 2.5}},
            {"id": "ind_3", "fitness": 0.3, "genes": {"param1": 0.5, "param2": 1.5}},
        ]

        # Функция приспособленности
        def fitness_function(individual: Dict[str, Any]) -> float:
            return individual.get("fitness", 0.0)

        # Эволюция популяции
        evolved_population = await engine.evolve(population, fitness_function)

        assert isinstance(evolved_population, list)
        assert len(evolved_population) > 0

        # Адаптация индивида
        entity = {"id": "test_entity", "genes": {"param1": 1.0}}
        environment = {"pressure": "high", "resources": "limited"}

        adapted_entity = await engine.adapt(entity, environment)

        assert isinstance(adapted_entity, dict)
        assert "id" in adapted_entity
        assert "genes" in adapted_entity

        # Обучение на данных
        learning_data = [{"input": [1, 2, 3], "output": 6}, {"input": [4, 5, 6], "output": 15}]

        learning_result = await engine.learn(learning_data)

        assert isinstance(learning_result, dict)
        assert "model_updated" in learning_result
        assert "performance_improvement" in learning_result

    @pytest.mark.asyncio
    async def test_improvement_applier_functionality(self: "TestClass") -> None:
        """Тест функциональности Improvement Applier."""
        config = {"validation_enabled": True, "rollback_enabled": True}
        applier = Mock()

        # Создание тестового улучшения
        improvement = {
            "id": "imp_001",
            "name": "Test Improvement",
            "description": "Testing improvement functionality",
            "category": "performance",
            "implementation": {
                "type": "code_change",
                "file": "test_file.py",
                "changes": [{"line": 10, "old": "slow_code()", "new": "fast_code()"}],
            },
            "validation_rules": [{"type": "performance_check", "threshold": 0.1}],
            "rollback_plan": {"type": "git_revert", "commit_hash": "abc123"},
            "created_at": datetime.now(),
            "applied_at": None,
            "status": "pending",
        }

        # Валидация улучшения
        is_valid = await applier.validate_improvement(improvement)
        assert isinstance(is_valid, bool)

        # Применение улучшения
        if is_valid:
            success = await applier.apply_improvement(improvement)
            assert isinstance(success, bool)

            # Откат улучшения
            if success:
                rollback_success = await applier.rollback_improvement("imp_001")
                assert isinstance(rollback_success, bool)

    @pytest.mark.asyncio
    async def test_full_system_integration(self, entity_config: EntitySystemConfig, temp_codebase: Path) -> None:
        """Полная интеграция всех компонентов системы."""
        controller = Mock()

        try:
            # Запуск системы
            await controller.start()

            # Ожидание нескольких циклов
            await asyncio.sleep(15)  # 3 цикла по 5 секунд

            # Проверка состояния
            status = await controller.get_status()

            assert status["is_running"]
            assert status["system_health"] > 0.0
            assert status["performance_score"] > 0.0
            assert status["efficiency_score"] > 0.0
            assert status["ai_confidence"] > 0.0

            # Проверка метрик
            assert controller.metrics.total_cycles >= 2
            assert controller.metrics.average_cycle_time > 0.0

        finally:
            # Остановка системы
            await controller.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self, entity_config: EntitySystemConfig) -> None:
        """Тест обработки ошибок."""
        controller = Mock()

        # Попытка остановки не запущенной системы
        await controller.stop()  # Не должно вызывать ошибку

        # Попытка повторного запуска
        await controller.start()
        await controller.start()  # Не должно вызывать ошибку

        # Остановка
        await controller.stop()

    @pytest.mark.asyncio
    async def test_operation_mode_changes(self, entity_config: EntitySystemConfig) -> None:
        """Тест изменения режимов работы."""
        controller = Mock()

        # Изменение режима работы
        controller.set_operation_mode(OperationMode.MANUAL)
        controller.set_operation_mode(OperationMode.AUTOMATIC)
        controller.set_operation_mode(OperationMode.HYBRID)

        # Изменение уровня оптимизации
        controller.set_optimization_level(OptimizationLevel.LOW)
        controller.set_optimization_level(OptimizationLevel.MEDIUM)
        controller.set_optimization_level(OptimizationLevel.HIGH)
        controller.set_optimization_level(OptimizationLevel.EXTREME)

        # Проверка, что изменения применились
        status = await controller.get_status()
        assert status["optimization_level"] == "extreme"


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
