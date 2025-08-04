"""Сканер стратегий Entity System."""

import ast
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from shared.numpy_utils import np

from ..memory.utils import estimate_object_size, retry_on_failure


class StrategyScanner:
    """Сканер торговых стратегий."""

    def __init__(self) -> None:
        self.strategy_patterns = {
            "trend_following": [
                r"moving_average",
                r"trend_line",
                r"breakout",
                r"momentum",
            ],
            "mean_reversion": [
                r"bollinger_bands",
                r"rsi",
                r"stochastic",
                r"oversold",
                r"overbought",
            ],
            "arbitrage": [
                r"arbitrage",
                r"spread",
                r"correlation",
                r"statistical_arbitrage",
            ],
            "market_making": [r"bid_ask", r"spread", r"inventory", r"order_book"],
        }
        self.risk_patterns = [
            r"stop_loss",
            r"take_profit",
            r"position_size",
            r"risk_per_trade",
            r"max_drawdown",
        ]
        self.performance_patterns = [
            r"sharpe_ratio",
            r"sortino_ratio",
            r"calmar_ratio",
            r"win_rate",
            r"profit_factor",
        ]
        # Кэш для результатов анализа
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._thread_lock = threading.RLock()
        logger.info("StrategyScanner инициализирован")

    @retry_on_failure(max_attempts=3, delay=0.5)
    async def scan_strategies(self, codebase_path: Path) -> List[Dict[str, Any]]:
        """Сканирование торговых стратегий в кодовой базе."""
        with self._thread_lock:
            if not codebase_path.exists():
                logger.error(f"Путь к кодовой базе не существует: {codebase_path}")
                return []
            strategies = []
            python_files = list(codebase_path.rglob("*.py"))
            logger.info(f"Найдено {len(python_files)} Python файлов для анализа")
            for file_path in python_files:
                if self._is_strategy_file(file_path):
                    strategy_info = await self._analyze_strategy_file(file_path)
                    if strategy_info:
                        strategies.append(strategy_info)
            logger.info(f"Найдено {len(strategies)} стратегий")
            return strategies

    @retry_on_failure(max_attempts=2, delay=0.2)
    async def scan_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Сканирование отдельного файла на предмет стратегий."""
        with self._thread_lock:
            if not file_path.exists():
                logger.error(f"Файл не существует: {file_path}")
                return None
            if file_path.suffix != ".py":
                logger.warning(f"Файл не является Python файлом: {file_path}")
                return None
            return await self._analyze_strategy_file(file_path)

    def _is_strategy_file(self, file_path: Path) -> bool:
        """Определение, является ли файл стратегией."""
        strategy_indicators = ["strategy", "trader", "signal", "indicator", "algorithm"]
        try:
            file_content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_name = file_path.name.lower()
            # Проверка имени файла
            for indicator in strategy_indicators:
                if indicator in file_name:
                    return True
            # Проверка содержимого файла
            for indicator in strategy_indicators:
                if indicator in file_content.lower():
                    return True
            return False
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return False

    async def _analyze_strategy_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Анализ файла стратегии."""
        cache_key = str(file_path)
        # Проверяем кэш
        with self._cache_lock:
            if cache_key in self._analysis_cache:
                logger.debug(f"Результат анализа для {file_path} найден в кэше")
                return self._analysis_cache[cache_key]
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            # Парсинг AST
            tree = ast.parse(content)
            strategy_info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "strategy_type": self._detect_strategy_type(content),
                "risk_management": self._detect_risk_management(content),
                "performance_metrics": self._detect_performance_metrics(content),
                "complexity": self._calculate_complexity(tree),
                "functions": self._extract_functions(tree),
                "classes": self._extract_classes(tree),
                "imports": self._extract_imports(tree),
                "lines_of_code": len(content.splitlines()),
                "file_size_bytes": len(content.encode("utf-8")),
                "timestamp": datetime.now().isoformat(),
            }
            # Кэшируем результат
            with self._cache_lock:
                self._analysis_cache[cache_key] = strategy_info
            return strategy_info
        except Exception as e:
            logger.error(f"Ошибка анализа файла {file_path}: {e}")
            return None

    def _detect_strategy_type(self, content: str) -> List[str]:
        """Определение типа стратегии."""
        detected_types = []
        content_lower = content.lower()
        for strategy_type, patterns in self.strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    detected_types.append(strategy_type)
                    break
        return detected_types

    def _detect_risk_management(self, content: str) -> Dict[str, Any]:
        """Определение управления рисками."""
        risk_features = {}
        content_lower = content.lower()
        for pattern in self.risk_patterns:
            if re.search(pattern, content_lower):
                risk_features[pattern] = True
        return {
            "features": risk_features,
            "score": len(risk_features) / len(self.risk_patterns),
        }

    def _detect_performance_metrics(self, content: str) -> Dict[str, Any]:
        """Определение метрик производительности."""
        performance_features = {}
        content_lower = content.lower()
        for pattern in self.performance_patterns:
            if re.search(pattern, content_lower):
                performance_features[pattern] = True
        return {
            "features": performance_features,
            "score": len(performance_features) / len(self.performance_patterns),
        }

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Расчет сложности кода."""
        complexity = {
            "cyclomatic": 0,
            "cognitive": 0,
            "nesting_depth": 0,
            "max_nesting": 0,
        }

        def calculate_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
            complexity["max_nesting"] = max(complexity["max_nesting"], current_depth)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    calculate_nesting_depth(child, current_depth + 1)
                else:
                    calculate_nesting_depth(child, current_depth)
            return complexity["max_nesting"]

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity["cyclomatic"] += 1
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity["cognitive"] += 1
        calculate_nesting_depth(tree)
        complexity["nesting_depth"] = complexity["max_nesting"]
        return complexity

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Извлечение функций."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Подсчет параметров
                args_count = len(node.args.args)
                kwargs_count = (
                    len(node.args.kwonlyargs) if hasattr(node.args, "kwonlyargs") else 0
                )
                varargs = 1 if node.args.vararg else 0
                kwarg = 1 if node.args.kwarg else 0
                functions.append(
                    {
                        "name": node.name,
                        "args": args_count,
                        "kwargs": kwargs_count,
                        "varargs": varargs,
                        "kwarg": kwarg,
                        "total_params": args_count + kwargs_count + varargs + kwarg,
                        "lines": (
                            (node.end_lineno - node.lineno)
                            if hasattr(node, "end_lineno")
                            and node.end_lineno is not None
                            and node.lineno is not None
                            else 0
                        ),
                    }
                )
        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Извлечение классов."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                class_vars = [n for n in node.body if isinstance(n, ast.Assign)]
                classes.append(
                    {
                        "name": node.name,
                        "methods": len(methods),
                        "class_variables": len(class_vars),
                        "lines": (
                            (node.end_lineno - node.lineno)
                            if hasattr(node, "end_lineno")
                            and node.end_lineno is not None
                            and node.lineno is not None
                            else 0
                        ),
                    }
                )
        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Извлечение импортов."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def get_strategy_statistics(
        self, strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Получение статистики стратегий."""
        with self._thread_lock:
            if not strategies:
                return {}
            strategy_types: Dict[str, int] = {}
            risk_scores = []
            performance_scores = []
            complexity_scores = []
            total_size_bytes = 0
            for strategy in strategies:
                # Типы стратегий
                for strategy_type in strategy.get("strategy_type", []):
                    strategy_types[strategy_type] = (
                        strategy_types.get(strategy_type, 0) + 1
                    )
                # Оценки риска и производительности
                risk_scores.append(strategy.get("risk_management", {}).get("score", 0))
                performance_scores.append(
                    strategy.get("performance_metrics", {}).get("score", 0)
                )
                # Сложность
                complexity = strategy.get("complexity", {})
                complexity_scores.append(complexity.get("cyclomatic", 0))
                # Размер файла
                total_size_bytes += strategy.get("file_size_bytes", 0)
            return {
                "total_strategies": len(strategies),
                "strategy_types": strategy_types,
                "average_risk_score": (
                    sum(risk_scores) / len(risk_scores) if risk_scores else 0
                ),
                "average_performance_score": (
                    sum(performance_scores) / len(performance_scores)
                    if performance_scores
                    else 0
                ),
                "average_complexity": (
                    sum(complexity_scores) / len(complexity_scores)
                    if complexity_scores
                    else 0
                ),
                "total_lines_of_code": sum(
                    s.get("lines_of_code", 0) for s in strategies
                ),
                "total_size_bytes": total_size_bytes,
                "cache_size": len(self._analysis_cache),
            }

    def clear_cache(self) -> None:
        """Очистка кэша результатов анализа."""
        with self._cache_lock:
            self._analysis_cache.clear()
            logger.info("Кэш результатов анализа очищен")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        with self._cache_lock:
            total_size = sum(
                estimate_object_size(result) for result in self._analysis_cache.values()
            )
            return {
                "cache_size": len(self._analysis_cache),
                "cache_memory_bytes": total_size,
                "cached_files": list(self._analysis_cache.keys()),
            }

    async def export_analysis_results(
        self, strategies: List[Dict[str, Any]], export_path: Path
    ) -> bool:
        """Экспорт результатов анализа в JSON файл."""
        try:
            export_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_path / f"strategy_analysis_{timestamp}.json"
            import json

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "total_strategies": len(strategies),
                        "statistics": self.get_strategy_statistics(strategies),
                        "strategies": strategies,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Результаты анализа экспортированы в {export_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта результатов анализа: {e}")
            return False
