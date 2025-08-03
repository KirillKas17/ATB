"""Unit тесты для StrategyScanner."""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import tempfile
import json
from pathlib import Path
from infrastructure.entity_system.perception.strategy_scanner import StrategyScanner
class TestStrategyScanner:
    """Тесты для StrategyScanner."""
    @pytest.fixture
    def scanner(self) -> StrategyScanner:
        """Создание экземпляра StrategyScanner."""
        return StrategyScanner()
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Создание временной директории для тестов."""
        temp_dir = tempfile.mkdtemp()
        return Path(temp_dir)
    @pytest.fixture
    def sample_strategy_file(self, temp_dir: Path) -> Path:
        """Создание тестового файла стратегии."""
        strategy_content = '''
"""Тестовая стратегия."""
import numpy as np
import pandas as pd
class TrendFollowingStrategy:
    """Стратегия следования за трендом."""
    def __init__(self, moving_average_period: int = 20) -> Any:
        self.moving_average_period = moving_average_period
        self.stop_loss = 0.02
        self.take_profit = 0.05
        self.position_size = 0.1
        self.risk_per_trade = 0.01
        self.max_drawdown = 0.1
    def calculate_moving_average(self, prices: pd.Series) -> pd.Series:
        """Расчет скользящего среднего."""
        return prices.rolling(window=self.moving_average_period).mean()
    def detect_breakout(self, current_price: float, ma: float) -> bool:
        """Определение пробоя."""
        return current_price > ma * 1.02
    def calculate_momentum(self, prices: pd.Series) -> float:
        """Расчет моментума."""
        return (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ рынка."""
        if market_data.get('rsi', 50) < 30:
            return {'action': 'buy', 'confidence': 0.8}
        elif market_data.get('rsi', 50) > 70:
            return {'action': 'sell', 'confidence': 0.7}
        else:
            return {'action': 'hold', 'confidence': 0.5}
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Расчет коэффициента Шарпа."""
        return returns.mean() / returns.std() if returns.std() > 0 else 0
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Расчет процента выигрышных сделок."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades)
'''
        strategy_file = temp_dir / "trend_following_strategy.py"
        strategy_file.write_text(strategy_content, encoding='utf-8')
        return strategy_file
    @pytest.fixture
    def sample_non_strategy_file(self, temp_dir: Path) -> Path:
        """Создание тестового файла, не являющегося стратегией."""
        non_strategy_content = '''
"""Обычный утилитарный модуль."""
import os
import sys
def read_config(config_path: str) -> dict:
    """Чтение конфигурации."""
    return {'setting': 'value'}
def validate_data(data: List[float]) -> bool:
    """Валидация данных."""
    return all(isinstance(x, (int, float)) for x in data)
'''
        non_strategy_file = temp_dir / "utils.py"
        non_strategy_file.write_text(non_strategy_content, encoding='utf-8')
        return non_strategy_file
    def test_scanner_initialization(self, scanner: StrategyScanner) -> None:
        """Тест инициализации сканера."""
        assert scanner is not None
        assert hasattr(scanner, 'strategy_patterns')
        assert hasattr(scanner, 'risk_patterns')
        assert hasattr(scanner, 'performance_patterns')
        assert hasattr(scanner, '_analysis_cache')
        assert hasattr(scanner, '_cache_lock')
        assert hasattr(scanner, '_thread_lock')
        # Проверка паттернов
        assert 'trend_following' in scanner.strategy_patterns
        assert 'mean_reversion' in scanner.strategy_patterns
        assert 'arbitrage' in scanner.strategy_patterns
        assert 'market_making' in scanner.strategy_patterns
        assert len(scanner.risk_patterns) > 0
        assert len(scanner.performance_patterns) > 0
    def test_is_strategy_file_true(self, scanner: StrategyScanner, sample_strategy_file: Path) -> None:
        """Тест определения файла стратегии (положительный случай)."""
        assert scanner._is_strategy_file(sample_strategy_file) is True
    def test_is_strategy_file_false(self, scanner: StrategyScanner, sample_non_strategy_file: Path) -> None:
        """Тест определения файла стратегии (отрицательный случай)."""
        assert scanner._is_strategy_file(sample_non_strategy_file) is False
    def test_is_strategy_file_by_name(self, scanner: StrategyScanner, temp_dir: Path) -> None:
        """Тест определения стратегии по имени файла."""
        strategy_file = temp_dir / "my_strategy.py"
        strategy_file.write_text("def test() -> Any: pass", encoding='utf-8')
        assert scanner._is_strategy_file(strategy_file) is True
    def test_is_strategy_file_nonexistent(self, scanner: StrategyScanner, temp_dir: Path) -> None:
        """Тест обработки несуществующего файла."""
        nonexistent_file = temp_dir / "nonexistent.py"
        assert scanner._is_strategy_file(nonexistent_file) is False
    @pytest.mark.asyncio
    async def test_analyze_strategy_file(self, scanner: StrategyScanner, sample_strategy_file: Path) -> None:
        """Тест анализа файла стратегии."""
        result = await scanner._analyze_strategy_file(sample_strategy_file)
        assert result is not None
        assert result['file_path'] == str(sample_strategy_file)
        assert result['file_name'] == sample_strategy_file.name
        assert 'trend_following' in result['strategy_type']
        assert result['lines_of_code'] > 0
        assert result['file_size_bytes'] > 0
        assert 'timestamp' in result
        # Проверка анализа сложности
        complexity = result['complexity']
        assert 'cyclomatic' in complexity
        assert 'cognitive' in complexity
        assert 'nesting_depth' in complexity
        assert 'max_nesting' in complexity
        # Проверка извлечения функций
        functions = result['functions']
        assert len(functions) > 0
        assert any(f['name'] == 'analyze_market' for f in functions)
        # Проверка извлечения классов
        classes = result['classes']
        assert len(classes) > 0
        assert any(c['name'] == 'TrendFollowingStrategy' for c in classes)
        # Проверка извлечения импортов
        imports = result['imports']
        assert len(imports) > 0
        assert any('numpy' in imp for imp in imports)
        assert any('pandas' in imp for imp in imports)
    @pytest.mark.asyncio
    async def test_analyze_strategy_file_caching(self, scanner: StrategyScanner, sample_strategy_file: Path) -> None:
        """Тест кэширования результатов анализа."""
        # Первый анализ
        result1 = await scanner._analyze_strategy_file(sample_strategy_file)
        assert result1 is not None
        # Второй анализ (должен быть из кэша)
        result2 = await scanner._analyze_strategy_file(sample_strategy_file)
        assert result2 is not None
        assert result1 == result2
        # Проверка статистики кэша
        cache_stats = scanner.get_cache_stats()
        assert cache_stats['cache_size'] > 0
        assert cache_stats['cache_memory_bytes'] > 0
        assert str(sample_strategy_file) in cache_stats['cached_files']
    def test_detect_strategy_type(self, scanner: StrategyScanner) -> None:
        """Тест определения типа стратегии."""
        content = """
        def calculate_moving_average() -> Any:
            pass
        def detect_breakout() -> Any:
            pass
        def calculate_momentum() -> Any:
            pass
        """
        types = scanner._detect_strategy_type(content)
        assert 'trend_following' in types
    def test_detect_risk_management(self, scanner: StrategyScanner) -> None:
        """Тест определения управления рисками."""
        content = """
        self.stop_loss = 0.02
        self.take_profit = 0.05
        self.position_size = 0.1
        """
        risk_info = scanner._detect_risk_management(content)
        assert 'features' in risk_info
        assert 'score' in risk_info
        assert risk_info['score'] > 0
        assert 'stop_loss' in risk_info['features']
        assert 'take_profit' in risk_info['features']
        assert 'position_size' in risk_info['features']
    def test_detect_performance_metrics(self, scanner: StrategyScanner) -> None:
        """Тест определения метрик производительности."""
        content = """
        def calculate_sharpe_ratio() -> Any:
            pass
        def calculate_win_rate() -> Any:
            pass
        def calculate_profit_factor() -> Any:
            pass
        """
        perf_info = scanner._detect_performance_metrics(content)
        assert 'features' in perf_info
        assert 'score' in perf_info
        assert perf_info['score'] > 0
        assert 'sharpe_ratio' in perf_info['features']
        assert 'win_rate' in perf_info['features']
    def test_calculate_complexity(self, scanner: StrategyScanner) -> None:
        """Тест расчета сложности кода."""
        import ast
        code = """
def complex_function(x) -> Any:
    if x > 0:
        if x > 10:
            for i in range(x):
                if i % 2 == 0:
                    try:
                        result = i * 2
                    except Exception:
                        result = 0
                else:
                    result = i * 3
        else:
            result = x * 2
    else:
        result = 0
    return result
"""
        tree = ast.parse(code)
        complexity = scanner._calculate_complexity(tree)
        assert complexity['cyclomatic'] > 0
        assert complexity['cognitive'] > 0
        assert complexity['nesting_depth'] > 0
        assert complexity['max_nesting'] > 0
    def test_extract_functions(self, scanner: StrategyScanner) -> None:
        """Тест извлечения функций."""
        import ast
        code = """
def simple_func() -> Any:
    pass
def func_with_args(a, b, c=10, *args, **kwargs) -> Any:
    pass
"""
        tree = ast.parse(code)
        functions = scanner._extract_functions(tree)
        assert len(functions) == 2
        simple_func = next(f for f in functions if f['name'] == 'simple_func')
        assert simple_func['args'] == 0
        assert simple_func['kwargs'] == 0
        assert simple_func['varargs'] == 0
        assert simple_func['kwarg'] == 0
        assert simple_func['total_params'] == 0
        complex_func = next(f for f in functions if f['name'] == 'func_with_args')
        assert complex_func['args'] == 3
        assert complex_func['kwargs'] == 0
        assert complex_func['varargs'] == 1
        assert complex_func['kwarg'] == 1
        assert complex_func['total_params'] == 5
    def test_extract_classes(self, scanner: StrategyScanner) -> None:
        """Тест извлечения классов."""
        import ast
        code = """
class SimpleClass:
    class_var = 10
    def method1(self) -> Any:
        pass
    def method2(self) -> Any:
        pass
"""
        tree = ast.parse(code)
        classes = scanner._extract_classes(tree)
        assert len(classes) == 1
        cls = classes[0]
        assert cls['name'] == 'SimpleClass'
        assert cls['methods'] == 2
        assert cls['class_variables'] == 1
    def test_extract_imports(self, scanner: StrategyScanner) -> None:
        """Тест извлечения импортов."""
        import ast
        code = """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
"""
        tree = ast.parse(code)
        imports = scanner._extract_imports(tree)
        assert 'numpy' in imports
        assert 'pandas' in imports
        assert 'typing.List' in imports
        assert 'typing.Dict' in imports
        assert 'sklearn.ensemble.RandomForestClassifier' in imports
    def test_get_strategy_statistics(self, scanner: StrategyScanner) -> None:
        """Тест получения статистики стратегий."""
        strategies = [
            {
                'strategy_type': ['trend_following'],
                'risk_management': {'score': 0.8},
                'performance_metrics': {'score': 0.7},
                'complexity': {'cyclomatic': 10},
                'lines_of_code': 100,
                'file_size_bytes': 1024
            },
            {
                'strategy_type': ['mean_reversion'],
                'risk_management': {'score': 0.6},
                'performance_metrics': {'score': 0.9},
                'complexity': {'cyclomatic': 15},
                'lines_of_code': 150,
                'file_size_bytes': 2048
            }
        ]
        stats = scanner.get_strategy_statistics(strategies)
        assert stats['total_strategies'] == 2
        assert stats['strategy_types']['trend_following'] == 1
        assert stats['strategy_types']['mean_reversion'] == 1
        assert stats['average_risk_score'] == 0.7
        assert stats['average_performance_score'] == 0.8
        assert stats['average_complexity'] == 12.5
        assert stats['total_lines_of_code'] == 250
        assert stats['total_size_bytes'] == 3072
    def test_get_strategy_statistics_empty(self, scanner: StrategyScanner) -> None:
        """Тест статистики для пустого списка стратегий."""
        stats = scanner.get_strategy_statistics([])
        assert stats == {}
    def test_clear_cache(self, scanner: StrategyScanner) -> None:
        """Тест очистки кэша."""
        # Добавляем данные в кэш
        scanner._analysis_cache['test'] = {'data': 'value'}
        assert len(scanner._analysis_cache) == 1
        # Очищаем кэш
        scanner.clear_cache()
        assert len(scanner._analysis_cache) == 0
    def test_get_cache_stats(self, scanner: StrategyScanner) -> None:
        """Тест получения статистики кэша."""
        # Пустой кэш
        stats = scanner.get_cache_stats()
        assert stats['cache_size'] == 0
        assert stats['cache_memory_bytes'] == 0
        assert len(stats['cached_files']) == 0
        # Кэш с данными
        scanner._analysis_cache['test1'] = {'data': 'value1'}
        scanner._analysis_cache['test2'] = {'data': 'value2'}
        stats = scanner.get_cache_stats()
        assert stats['cache_size'] == 2
        assert stats['cache_memory_bytes'] > 0
        assert len(stats['cached_files']) == 2
    @pytest.mark.asyncio
    async def test_scan_file_success(self, scanner: StrategyScanner, sample_strategy_file: Path) -> None:
        """Тест успешного сканирования файла."""
        result = await scanner.scan_file(sample_strategy_file)
        assert result is not None
        assert result['file_path'] == str(sample_strategy_file)
    @pytest.mark.asyncio
    async def test_scan_file_nonexistent(self, scanner: StrategyScanner, temp_dir: Path) -> None:
        """Тест сканирования несуществующего файла."""
        nonexistent_file = temp_dir / "nonexistent.py"
        result = await scanner.scan_file(nonexistent_file)
        assert result is None
    @pytest.mark.asyncio
    async def test_scan_file_not_python(self, scanner: StrategyScanner, temp_dir: Path) -> None:
        """Тест сканирования не-Python файла."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("This is not Python code", encoding='utf-8')
        result = await scanner.scan_file(text_file)
        assert result is None
    @pytest.mark.asyncio
    async def test_scan_strategies(self, scanner: StrategyScanner, temp_dir: Path, sample_strategy_file: Path, sample_non_strategy_file: Path) -> None:
        """Тест сканирования стратегий в директории."""
        # Создаем дополнительный файл стратегии
        strategy2 = temp_dir / "arbitrage_strategy.py"
        strategy2.write_text("""
        class ArbitrageStrategy:
            def find_spread(self) -> Any:
                pass
            def calculate_correlation(self) -> Any:
                pass
        """, encoding='utf-8')
        results = await scanner.scan_strategies(temp_dir)
        assert len(results) == 2  # trend_following_strategy.py и arbitrage_strategy.py
        file_names = [r['file_name'] for r in results]
        assert 'trend_following_strategy.py' in file_names
        assert 'arbitrage_strategy.py' in file_names
    @pytest.mark.asyncio
    async def test_scan_strategies_nonexistent_dir(self, scanner: StrategyScanner) -> None:
        """Тест сканирования несуществующей директории."""
        nonexistent_dir = Path("/nonexistent/directory")
        results = await scanner.scan_strategies(nonexistent_dir)
        assert results == []
    @pytest.mark.asyncio
    async def test_export_analysis_results(self, scanner: StrategyScanner, temp_dir: Path) -> None:
        """Тест экспорта результатов анализа."""
        strategies = [
            {
                'file_path': '/path/to/strategy1.py',
                'file_name': 'strategy1.py',
                'strategy_type': ['trend_following'],
                'risk_management': {'score': 0.8},
                'performance_metrics': {'score': 0.7},
                'complexity': {'cyclomatic': 10},
                'lines_of_code': 100,
                'file_size_bytes': 1024,
                'timestamp': '2023-01-01T00:00:00'
            }
        ]
        export_dir = temp_dir / "exports"
        success = await scanner.export_analysis_results(strategies, export_dir)
        assert success is True
        assert export_dir.exists()
        # Проверяем, что файл создан
        export_files = list(export_dir.glob("strategy_analysis_*.json"))
        assert len(export_files) == 1
        # Проверяем содержимое файла
        with open(export_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'timestamp' in data
        assert data['total_strategies'] == 1
        assert 'statistics' in data
        assert 'strategies' in data
        assert len(data['strategies']) == 1
    @pytest.mark.asyncio
    async def test_export_analysis_results_error(self, scanner: StrategyScanner) -> None:
        """Тест экспорта с ошибкой (недоступная директория)."""
        strategies = [{'test': 'data'}]
        export_dir = Path("/nonexistent/export/dir")
        success = await scanner.export_analysis_results(strategies, export_dir)
        assert success is False
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, scanner: StrategyScanner, sample_strategy_file: Path) -> None:
        """Тест retry-декоратора при сбоях."""
        # Мокаем метод, который будет падать первые 2 раза
        original_method = scanner._analyze_strategy_file
        call_count = 0
        
        async def failing_method(file_path: Path) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return await original_method(file_path)
        
        # Используем monkeypatch для замены метода
        pytest.MonkeyPatch().setattr(scanner, '_analyze_strategy_file', failing_method)
        
        result = await scanner.scan_file(sample_strategy_file)
        assert result is not None
        assert call_count == 3  # Метод вызвался 3 раза (2 неудачных + 1 успешный)
    def test_thread_safety(self, scanner: StrategyScanner) -> None:
        """Тест thread-safety."""
        import threading
        import time
        results = []
        errors = []
        def worker() -> Any:
            try:
                # Симулируем работу с кэшем
                with scanner._cache_lock:
                    scanner._analysis_cache[f'key_{threading.current_thread().ident}'] = {'data': 'value'}
                    time.sleep(0.01)  # Небольшая задержка
                    results.append(len(scanner._analysis_cache))
            except Exception as e:
                errors.append(e)
        # Запускаем несколько потоков
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что нет ошибок
        assert len(errors) == 0
        assert len(results) == 5
        assert all(r > 0 for r in results) 
