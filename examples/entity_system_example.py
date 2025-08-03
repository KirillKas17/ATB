"""
Пример использования Entity System.

Демонстрирует работу всех компонентов системы в реальном сценарии.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Добавление корневой директории в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.types.entity_system_types import (
    EntitySystemConfig,
    OperationMode,
    OptimizationLevel,
    SystemPhase
)

from infrastructure.entity_system import (
    EntityControllerImpl,
    CodeScannerImpl,
    CodeAnalyzerImpl,
    ExperimentRunnerImpl,
    ImprovementApplierImpl,
    MemoryManagerImpl,
    AIEnhancementImpl,
    EvolutionEngineImpl
)


async def demonstrate_code_scanning():
    """Демонстрация сканирования кода."""
    print("\n=== Демонстрация сканирования кода ===")
    
    config = {"analysis_interval": 60}
    scanner = CodeScannerImpl(config)
    
    # Сканирование текущей директории
    codebase_path = Path(".")
    code_structures = await scanner.scan_codebase(codebase_path)
    
    print(f"Найдено файлов: {len(code_structures)}")
    
    # Анализ Python файлов
    python_files = [cs for cs in code_structures if cs["file_path"].endswith(".py")]
    print(f"Python файлов: {len(python_files)}")
    
    for cs in python_files[:3]:  # Показываем первые 3 файла
        print(f"\nФайл: {cs['file_path']}")
        print(f"  Строк кода: {cs['lines_of_code']}")
        print(f"  Функций: {len(cs['functions'])}")
        print(f"  Классов: {len(cs['classes'])}")
        print(f"  Импортов: {len(cs['imports'])}")
        print(f"  Сложность: {cs['complexity_metrics'].get('cyclomatic_complexity', 'N/A')}")


async def demonstrate_code_analysis():
    """Демонстрация анализа кода."""
    print("\n=== Демонстрация анализа кода ===")
    
    config = {"confidence_threshold": 0.7}
    analyzer = CodeAnalyzerImpl(config)
    scanner = CodeScannerImpl(config)
    
    # Сканирование и анализ файла
    test_file = Path("examples/entity_system_example.py")
    if test_file.exists():
        code_structure = await scanner.scan_file(test_file)
        analysis_result = await analyzer.analyze_code(code_structure)
        
        print(f"Анализ файла: {analysis_result['file_path']}")
        print(f"  Качество: {analysis_result['quality_score']:.2f}")
        print(f"  Производительность: {analysis_result['performance_score']:.2f}")
        print(f"  Поддерживаемость: {analysis_result['maintainability_score']:.2f}")
        print(f"  Сложность: {analysis_result['complexity_score']:.2f}")
        
        # Показать предложения
        if analysis_result['suggestions']:
            print(f"  Предложения: {len(analysis_result['suggestions'])}")
            for suggestion in analysis_result['suggestions'][:2]:
                print(f"    - {suggestion.get('description', 'N/A')}")
        
        # Показать проблемы
        if analysis_result['issues']:
            print(f"  Проблемы: {len(analysis_result['issues'])}")
            for issue in analysis_result['issues'][:2]:
                print(f"    - {issue.get('description', 'N/A')}")


async def demonstrate_experiments():
    """Демонстрация экспериментов."""
    print("\n=== Демонстрация экспериментов ===")
    
    config = {"experiment_duration": 5}
    runner = ExperimentRunnerImpl(config)
    
    # Создание тестового эксперимента
    experiment = {
        "id": "demo_exp_001",
        "hypothesis_id": "demo_hyp_001",
        "name": "Демонстрационный эксперимент",
        "description": "Тестирование функциональности экспериментов",
        "parameters": {
            "algorithm": "genetic",
            "population_size": 100,
            "mutation_rate": 0.1
        },
        "start_time": datetime.now(),
        "end_time": None,
        "status": "running",
        "results": None,
        "metrics": None
    }
    
    # Запуск эксперимента
    print("Запуск эксперимента...")
    results = await runner.run_experiment(experiment)
    
    print(f"Результаты эксперимента:")
    print(f"  Статус: {results.get('status', 'N/A')}")
    print(f"  Метрики: {len(results.get('metrics', {}))}")
    
    # Остановка эксперимента
    success = await runner.stop_experiment("demo_exp_001")
    print(f"Эксперимент остановлен: {success}")


async def demonstrate_memory_management():
    """Демонстрация управления памятью."""
    print("\n=== Демонстрация управления памятью ===")
    
    config = {"memory_enabled": True}
    manager = MemoryManagerImpl(config)
    
    # Создание снимка состояния
    print("Создание снимка состояния...")
    snapshot = await manager.create_snapshot()
    
    print(f"Снимок создан: {snapshot['id']}")
    print(f"  Временная метка: {snapshot['timestamp']}")
    print(f"  Состояние системы: {snapshot['system_state']['is_running']}")
    print(f"  Результаты анализа: {len(snapshot['analysis_results'])}")
    print(f"  Активные гипотезы: {len(snapshot['active_hypotheses'])}")
    print(f"  Активные эксперименты: {len(snapshot['active_experiments'])}")
    
    # Загрузка снимка
    print("\nЗагрузка снимка...")
    loaded_snapshot = await manager.load_snapshot(snapshot["id"])
    print(f"Снимок загружен: {loaded_snapshot['id']}")
    
    # Сохранение в журнал
    print("\nСохранение в журнал...")
    journal_data = {
        "event": "demo_event",
        "data": {"demo": "value", "timestamp": datetime.now().isoformat()},
        "timestamp": datetime.now().isoformat()
    }
    success = await manager.save_to_journal(journal_data)
    print(f"Данные сохранены в журнал: {success}")


async def demonstrate_ai_enhancement():
    """Демонстрация AI улучшений."""
    print("\n=== Демонстрация AI улучшений ===")
    
    config = {"ai_enabled": True}
    enhancement = AIEnhancementImpl(config)
    scanner = CodeScannerImpl(config)
    
    # Сканирование файла для анализа
    test_file = Path("examples/entity_system_example.py")
    if test_file.exists():
        code_structure = await scanner.scan_file(test_file)
        
        # Предсказание качества кода
        print("Предсказание качества кода...")
        quality_prediction = await enhancement.predict_code_quality(code_structure)
        
        print(f"Качество кода:")
        print(f"  Общее качество: {quality_prediction.get('overall_quality', 0):.2f}")
        print(f"  Поддерживаемость: {quality_prediction.get('maintainability', 0):.2f}")
        print(f"  Производительность: {quality_prediction.get('performance', 0):.2f}")
        print(f"  Безопасность: {quality_prediction.get('security', 0):.2f}")
        
        # Предложение улучшений
        print("\nПредложения улучшений...")
        suggestions = await enhancement.suggest_improvements(code_structure)
        
        print(f"Найдено предложений: {len(suggestions)}")
        for suggestion in suggestions[:3]:
            print(f"  - {suggestion.get('type', 'N/A')}: {suggestion.get('description', 'N/A')}")
            print(f"    Приоритет: {suggestion.get('priority', 'N/A')}")
            print(f"    Ожидаемый эффект: {suggestion.get('estimated_impact', 'N/A')}")
        
        # Оптимизация параметров
        print("\nОптимизация параметров...")
        parameters = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.5
        }
        
        optimized_params = await enhancement.optimize_parameters(parameters)
        
        print("Оптимизированные параметры:")
        for key, value in optimized_params.items():
            print(f"  {key}: {value}")


async def demonstrate_evolution():
    """Демонстрация эволюционного развития."""
    print("\n=== Демонстрация эволюционного развития ===")
    
    config = {"evolution_enabled": True}
    engine = EvolutionEngineImpl(config)
    
    # Создание тестовой популяции
    population = [
        {"id": "ind_1", "fitness": 0.5, "genes": {"param1": 1.0, "param2": 2.0, "param3": 0.5}},
        {"id": "ind_2", "fitness": 0.7, "genes": {"param1": 1.5, "param2": 2.5, "param3": 0.7}},
        {"id": "ind_3", "fitness": 0.3, "genes": {"param1": 0.5, "param2": 1.5, "param3": 0.3}},
        {"id": "ind_4", "fitness": 0.8, "genes": {"param1": 2.0, "param2": 3.0, "param3": 0.8}},
        {"id": "ind_5", "fitness": 0.4, "genes": {"param1": 0.8, "param2": 1.8, "param3": 0.4}}
    ]
    
    print(f"Начальная популяция: {len(population)} особей")
    
    # Функция приспособленности
    def fitness_function(individual: Dict[str, Any]) -> float:
        return individual.get("fitness", 0.0)
    
    # Эволюция популяции
    print("Запуск эволюции...")
    evolved_population = await engine.evolve(population, fitness_function)
    
    print(f"Эволюционированная популяция: {len(evolved_population)} особей")
    
    # Анализ результатов
    best_individual = max(evolved_population, key=lambda x: x.get("fitness", 0))
    print(f"Лучшая особь: {best_individual['id']} (приспособленность: {best_individual['fitness']:.2f})")
    
    # Адаптация индивида
    print("\nАдаптация индивида...")
    entity = {"id": "test_entity", "genes": {"param1": 1.0, "param2": 2.0}}
    environment = {"pressure": "high", "resources": "limited", "competition": "intense"}
    
    adapted_entity = await engine.adapt(entity, environment)
    print(f"Адаптированная особь: {adapted_entity['id']}")
    print(f"Новые гены: {adapted_entity['genes']}")
    
    # Обучение на данных
    print("\nОбучение на данных...")
    learning_data = [
        {"input": [1, 2, 3], "output": 6, "performance": 0.8},
        {"input": [4, 5, 6], "output": 15, "performance": 0.9},
        {"input": [7, 8, 9], "output": 24, "performance": 0.7}
    ]
    
    learning_result = await engine.learn(learning_data)
    print(f"Результат обучения:")
    print(f"  Модель обновлена: {learning_result.get('model_updated', False)}")
    print(f"  Улучшение производительности: {learning_result.get('performance_improvement', 0):.2f}")


async def demonstrate_improvement_application():
    """Демонстрация применения улучшений."""
    print("\n=== Демонстрация применения улучшений ===")
    
    config = {"validation_enabled": True, "rollback_enabled": True}
    applier = ImprovementApplierImpl(config)
    
    # Создание тестового улучшения
    improvement = {
        "id": "demo_imp_001",
        "name": "Демонстрационное улучшение",
        "description": "Оптимизация алгоритма сортировки",
        "category": "performance",
        "implementation": {
            "type": "code_change",
            "file": "utils.py",
            "changes": [
                {
                    "line": 10,
                    "old": "return sorted(data)",
                    "new": "return sorted(data, key=lambda x: x, reverse=True)"
                }
            ]
        },
        "validation_rules": [
            {"type": "performance_check", "threshold": 0.1},
            {"type": "functionality_check", "test_cases": ["test1", "test2"]}
        ],
        "rollback_plan": {
            "type": "git_revert",
            "commit_hash": "abc123def456"
        },
        "created_at": datetime.now(),
        "applied_at": None,
        "status": "pending"
    }
    
    # Валидация улучшения
    print("Валидация улучшения...")
    is_valid = await applier.validate_improvement(improvement)
    print(f"Улучшение валидно: {is_valid}")
    
    if is_valid:
        # Применение улучшения
        print("Применение улучшения...")
        success = await applier.apply_improvement(improvement)
        print(f"Улучшение применено: {success}")
        
        if success:
            # Откат улучшения
            print("Откат улучшения...")
            rollback_success = await applier.rollback_improvement("demo_imp_001")
            print(f"Улучшение откачено: {rollback_success}")


async def demonstrate_full_system():
    """Демонстрация полной системы."""
    print("\n=== Демонстрация полной Entity System ===")
    
    # Конфигурация системы
    config = {
        "analysis_interval": 10,  # Быстрые циклы для демонстрации
        "experiment_duration": 5,
        "confidence_threshold": 0.7,
        "improvement_threshold": 0.1,
        "ai_enabled": True,
        "evolution_enabled": True,
        "memory_enabled": True,
        "validation_enabled": True,
        "rollback_enabled": True
    }
    
    controller = EntityControllerImpl(config)
    
    try:
        print("Запуск Entity System...")
        await controller.start()
        
        # Ожидание нескольких циклов
        print("Ожидание выполнения циклов...")
        await asyncio.sleep(25)  # 2-3 цикла
        
        # Получение статуса
        status = await controller.get_status()
        
        print(f"\nСтатус системы:")
        print(f"  Работает: {status['is_running']}")
        print(f"  Текущая фаза: {status['current_phase']}")
        print(f"  Здоровье системы: {status['system_health']:.2f}")
        print(f"  Производительность: {status['performance_score']:.2f}")
        print(f"  Эффективность: {status['efficiency_score']:.2f}")
        print(f"  AI уверенность: {status['ai_confidence']:.2f}")
        
        # Метрики системы
        print(f"\nМетрики системы:")
        print(f"  Всего циклов: {controller.metrics.total_cycles}")
        print(f"  Среднее время цикла: {controller.metrics.average_cycle_time:.2f} сек")
        print(f"  Успешных улучшений: {controller.metrics.successful_improvements}")
        print(f"  Завершенных экспериментов: {controller.metrics.experiments_completed}")
        
    finally:
        print("\nОстановка Entity System...")
        await controller.stop()
        print("Entity System остановлена")


async def main():
    """Основная функция демонстрации."""
    print("🚀 Демонстрация Entity System")
    print("=" * 50)
    
    try:
        # Демонстрация отдельных компонентов
        await demonstrate_code_scanning()
        await demonstrate_code_analysis()
        await demonstrate_experiments()
        await demonstrate_memory_management()
        await demonstrate_ai_enhancement()
        await demonstrate_evolution()
        await demonstrate_improvement_application()
        
        # Демонстрация полной системы
        await demonstrate_full_system()
        
        print("\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время демонстрации: {e}")
        logging.exception("Детали ошибки:")


if __name__ == "__main__":
    # Запуск демонстрации
    asyncio.run(main()) 