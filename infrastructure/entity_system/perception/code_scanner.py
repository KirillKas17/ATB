"""
Модуль сканирования кода проекта.

Предоставляет функциональность для анализа структуры кода,
вычисления метрик сложности, оценки качества кода
и анализа архитектурных зависимостей.
"""

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Set
from shared.numpy_utils import np

from loguru import logger


class CodeScanner:
    """Сканер кода проекта."""

    def __init__(self) -> None:
        self.project_root = Path(".")
        self.excluded_dirs = {
            ".git",
            "__pycache__",
            "venv",
            "node_modules",
            "build",
            "dist",
        }
        self.excluded_files = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe"}

    async def scan_project(self) -> Dict[str, Any]:
        logger.info("Начало сканирования проекта")
        structure: Dict[str, Any] = {
            "files": [],
            "directories": [],
            "complexity_metrics": {},
            "code_quality": {},
            "dependencies": {},
            "architecture": {},
        }
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            rel_root = Path(root).relative_to(self.project_root)
            structure["directories"].append(str(rel_root))
            for file in files:
                if not any(file.endswith(ext) for ext in self.excluded_files):
                    file_path = rel_root / file
                    structure["files"].append(str(file_path))
                    if file.endswith(".py"):
                        await self._analyze_python_file(file_path, structure)
        structure["architecture"] = await self._analyze_architecture(structure)
        logger.info(
            f"Сканирование завершено: {len(structure['files'])} файлов, {len(structure['directories'])} директорий"
        )
        return structure

    async def _analyze_python_file(
        self, file_path: Path, structure: Dict[str, Any]
    ) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            metrics: Dict[str, Any] = {
                "lines": len(content.splitlines()),
                "classes": len(
                    [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                ),
                "functions": len(
                    [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                    ]
                ),
                "imports": len(
                    [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
                ),
                "import_from": len(
                    [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom)
                    ]
                ),
            }
            complexity = self._calculate_complexity(tree)
            metrics["complexity"] = complexity
            quality = self._assess_code_quality(content, tree)
            metrics["quality"] = quality
            structure["complexity_metrics"][str(file_path)] = metrics
        except Exception as e:
            logger.warning(f"Ошибка анализа файла {file_path}: {e}")

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        complexity: Dict[str, int] = {"cyclomatic": 1, "cognitive": 0, "nesting": 0}
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity["cyclomatic"] += 1
            elif isinstance(node, ast.BoolOp):
                complexity["cyclomatic"] += len(node.values) - 1
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity["cognitive"] += 1
            elif isinstance(node, ast.BoolOp):
                complexity["cognitive"] += len(node.values) - 1
            if hasattr(node, "lineno"):
                complexity["nesting"] = max(
                    complexity["nesting"], self._get_nesting_level(node)
                )
        return complexity

    def _get_nesting_level(self, node: ast.AST) -> int:
        level = 0
        current = node
        while hasattr(current, "parent"):
            current = current.parent
            level += 1
        return level

    def _assess_code_quality(self, content: str, tree: ast.AST) -> Dict[str, Any]:
        quality: Dict[str, Any] = {"score": 100, "issues": [], "suggestions": []}
        lines = content.splitlines()
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            if isinstance(quality["issues"], list):
                quality["issues"].append(f"Длинные строки в строках: {long_lines[:5]}")
            quality["score"] = quality["score"] - len(long_lines) * 2
        docstrings = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)
        ]
        if not docstrings:
            if isinstance(quality["suggestions"], list):
                quality["suggestions"].append("Добавить документацию")
            quality["score"] = quality["score"] - 10
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if len(node.id) < 2:
                    if isinstance(quality["issues"], list):
                        quality["issues"].append(f"Короткое имя переменной: {node.id}")
                    quality["score"] = quality["score"] - 5
        quality["score"] = max(0, quality["score"])
        return quality

    async def _analyze_architecture(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Промышленный анализ архитектуры с построением графа зависимостей."""
        architecture: Dict[str, Any] = {
            "modules": {},
            "dependencies": {},
            "layers": {},
            "patterns": {},
        }

        # Анализ модулей
        for file_path in structure["files"]:
            if file_path.endswith(".py"):
                module_name = file_path.replace("/", ".").replace(".py", "")
                architecture["modules"][module_name] = {
                    "path": file_path,
                    "metrics": structure["complexity_metrics"].get(file_path, {}),
                }

        # Промышленный анализ зависимостей
        dependencies: Dict[str, Any] = await self._analyze_dependencies(structure)
        architecture["dependencies"] = dependencies

        # Анализ слоёв архитектуры
        layers: Dict[str, Any] = {
            "domain": [],
            "application": [],
            "infrastructure": [],
            "interfaces": [],
            "shared": [],
            "unknown": [],
        }
        for file_path in structure["files"]:
            if not file_path.endswith(".py"):
                continue

            layer = self._classify_module_layer(file_path)
            layers[layer].append(file_path)

        # Расчёт метрик слоёв
        layer_metrics: Dict[str, Dict[str, Any]] = {}
        for layer_name, layer_files in layers.items():
            if isinstance(layer_files, list):  # Проверяем, что это список
                layer_metrics[layer_name] = {
                    "file_count": len(layer_files),
                    "total_lines": sum(
                        structure["complexity_metrics"].get(f, {}).get("lines", 0)
                        for f in layer_files
                    ),
                    "average_complexity": self._calculate_average_complexity(
                        [structure["complexity_metrics"].get(f, {}) for f in layer_files]
                    ),
                }

        layers["metrics"] = layer_metrics
        architecture["layers"] = layers

        # Выявление архитектурных паттернов
        patterns: Dict[str, Any] = {
            "repository_pattern": [],
            "factory_pattern": [],
            "strategy_pattern": [],
            "observer_pattern": [],
            "singleton_pattern": [],
            "dependency_injection": [],
            "command_pattern": [],
            "adapter_pattern": [],
        }
        for file_path, metrics in structure["complexity_metrics"].items():
            if not file_path.endswith(".py"):
                continue

            detected_patterns = await self._detect_patterns_in_file(file_path)
            for pattern in detected_patterns:
                if pattern in patterns:
                    patterns[pattern].append(file_path)

        architecture["patterns"] = patterns

        # Анализ метрик архитектуры
        architecture_metrics: Dict[str, Any] = await self._calculate_architecture_metrics(architecture)
        architecture["metrics"] = architecture_metrics

        return architecture

    async def _analyze_dependencies(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Промышленный анализ зависимостей между модулями."""
        dependencies: Dict[str, Any] = {
            "graph": {},
            "incoming": {},
            "outgoing": {},
            "circular": [],
            "metrics": {},
        }

        # Построение графа зависимостей
        for file_path, metrics in structure["complexity_metrics"].items():
            if not file_path.endswith(".py"):
                continue

            module_name = file_path.replace("/", ".").replace(".py", "")
            imports = await self._extract_imports(file_path)

            dependencies["graph"][module_name] = {
                "imports": imports,
                "import_count": len(imports),
                "file_path": file_path,
            }

            # Подсчёт входящих и исходящих зависимостей
            dependencies["outgoing"][module_name] = len(imports)

            for imported_module in imports:
                if imported_module not in dependencies["incoming"]:
                    dependencies["incoming"][imported_module] = []
                dependencies["incoming"][imported_module].append(module_name)

        # Детекция циклических зависимостей
        circular_deps = await self._detect_circular_dependencies(dependencies["graph"])
        dependencies["circular"] = circular_deps

        # Расчёт метрик зависимостей
        dep_metrics = await self._calculate_dependency_metrics(dependencies)
        dependencies["metrics"] = dep_metrics

        return dependencies

    async def _extract_imports(self, file_path: str) -> List[str]:
        """Извлечение импортов из Python файла."""
        try:
            full_path = Path(file_path)
            if not full_path.exists():
                return []

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports

        except Exception as e:
            logger.warning(f"Ошибка извлечения импортов из {file_path}: {e}")
            return []

    async def _detect_circular_dependencies(
        self, graph: Dict[str, Any]
    ) -> List[List[str]]:
        """Детекция циклических зависимостей с использованием алгоритма поиска циклов."""

        def dfs(
            node: str, visited: Set[str], rec_stack: Set[str], path: List[str]
        ) -> List[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            cycles: List[List[str]] = []
            for neighbor in graph.get(node, {}).get("imports", []):
                if neighbor in graph:  # Проверяем только модули нашего проекта
                    if neighbor not in visited:
                        cycles.extend(dfs(neighbor, visited, rec_stack, path))
                    elif neighbor in rec_stack:
                        # Найден цикл
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)

            rec_stack.remove(node)
            path.pop()
            return cycles

        visited: Set[str] = set()
        all_cycles: List[List[str]] = []

        for node in graph:
            if node not in visited:
                cycles = dfs(node, visited, set(), [])
                all_cycles.extend(cycles)

        return all_cycles

    async def _calculate_dependency_metrics(
        self, dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Расчёт метрик зависимостей."""
        graph = dependencies["graph"]
        incoming = dependencies["incoming"]
        outgoing = dependencies["outgoing"]

        metrics = {
            "total_modules": len(graph),
            "total_dependencies": sum(outgoing.values()),
            "average_dependencies_per_module": (
                sum(outgoing.values()) / len(graph) if graph else 0
            ),
            "max_incoming": (
                max(len(deps) for deps in incoming.values()) if incoming else 0
            ),
            "max_outgoing": max(outgoing.values()) if outgoing else 0,
            "circular_dependencies_count": len(dependencies["circular"]),
            "modules_without_dependencies": len(
                [m for m, deps in outgoing.items() if deps == 0]
            ),
            "highly_coupled_modules": len(
                [m for m, deps in outgoing.items() if deps > 10]
            ),
        }

        # Расчёт метрик связности
        if graph:
            total_possible_edges = len(graph) * (len(graph) - 1)
            actual_edges = sum(outgoing.values())
            metrics["coupling_coefficient"] = (
                actual_edges / total_possible_edges if total_possible_edges > 0 else 0
            )

        return metrics

    def _classify_module_layer(self, file_path: str) -> str:
        """Классификация модуля по архитектурному слою."""
        path_parts = file_path.split("/")

        if "domain" in path_parts:
            return "domain"
        elif "application" in path_parts:
            return "application"
        elif "infrastructure" in path_parts:
            return "infrastructure"
        elif "interfaces" in path_parts:
            return "interfaces"
        elif "shared" in path_parts:
            return "shared"
        else:
            return "unknown"

    def _calculate_average_complexity(
        self, metrics_list: List[Dict[str, Any]]
    ) -> float:
        """Расчёт средней сложности."""
        if not metrics_list:
            return 0.0

        total_complexity = 0
        count = 0

        for metrics in metrics_list:
            if "complexity" in metrics and "cyclomatic" in metrics["complexity"]:
                total_complexity += metrics["complexity"]["cyclomatic"]
                count += 1

        return total_complexity / count if count > 0 else 0.0

    async def _detect_patterns_in_file(self, file_path: str) -> List[str]:
        """Детекция паттернов в файле."""
        detected_patterns = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Простые эвристики для детекции паттернов
            if "class.*Repository" in content or "Repository" in file_path:
                detected_patterns.append("repository_pattern")

            if "class.*Factory" in content or "Factory" in file_path:
                detected_patterns.append("factory_pattern")

            if "class.*Strategy" in content or "Strategy" in file_path:
                detected_patterns.append("strategy_pattern")

            if "Observer" in content or "subscribe" in content:
                detected_patterns.append("observer_pattern")

            if "get_instance" in content or "instance" in content:
                detected_patterns.append("singleton_pattern")

            if "inject" in content or "dependency" in content:
                detected_patterns.append("dependency_injection")

            if "execute" in content and "command" in content.lower():
                detected_patterns.append("command_pattern")

            if "adapt" in content or "adapter" in file_path:
                detected_patterns.append("adapter_pattern")

        except Exception as e:
            logger.warning(f"Ошибка детекции паттернов в {file_path}: {e}")

        return detected_patterns

    async def _calculate_architecture_metrics(
        self, architecture: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Расчёт метрик архитектуры."""
        metrics = {
            "modularity": 0.0,
            "coupling": 0.0,
            "cohesion": 0.0,
            "complexity": 0.0,
            "maintainability": 0.0,
        }

        # Расчёт модульности
        total_modules = len(architecture["modules"])
        if total_modules > 0:
            metrics["modularity"] = min(1.0, total_modules / 100)  # Нормализация

        # Расчёт связности (coupling)
        dependencies = architecture.get("dependencies", {})
        dep_metrics = dependencies.get("metrics", {})
        if dep_metrics:
            metrics["coupling"] = dep_metrics.get("coupling_coefficient", 0.0)

        # Расчёт связности (cohesion) - упрощённо
        layers = architecture.get("layers", {})
        layer_metrics = layers.get("metrics", {})
        if layer_metrics:
            total_files = sum(
                layer.get("file_count", 0) for layer in layer_metrics.values()
            )
            if total_files > 0:
                metrics["cohesion"] = 1.0 - (len(layer_metrics) / total_files)

        # Расчёт сложности
        all_metrics: List[float] = []
        for module_metrics in architecture["modules"].values():
            if (
                "metrics" in module_metrics
                and "complexity" in module_metrics["metrics"]
            ):
                complexity = module_metrics["metrics"]["complexity"]
                if "cyclomatic" in complexity:
                    all_metrics.append(complexity["cyclomatic"])

        if all_metrics:
            metrics["complexity"] = sum(all_metrics) / len(all_metrics)

        # Расчёт поддерживаемости
        metrics["maintainability"] = (
            (1.0 - metrics["coupling"]) * 0.4
            + metrics["cohesion"] * 0.3
            + (1.0 - min(metrics["complexity"] / 10, 1.0)) * 0.3
        )

        return metrics
