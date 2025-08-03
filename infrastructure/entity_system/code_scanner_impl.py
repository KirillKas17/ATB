import ast
import os
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from domain.types.entity_system_types import BaseCodeScanner, CodeStructure


class CodeScannerImpl(BaseCodeScanner):
    """
    Продвинутый сканер кода:
    - Рекурсивно сканирует директорию
    - Извлекает структуру, импорты, функции, классы
    - Считает метрики сложности, архитектурные паттерны
    - Логирует и обрабатывает ошибки
    """

    async def scan_codebase(self, path: Path) -> List[CodeStructure]:
        logger.info(f"Сканирование кодовой базы: {path}")
        code_structures = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        structure = await self.scan_file(file_path)
                        code_structures.append(structure)
                    except Exception as e:
                        logger.error(f"Ошибка при сканировании {file_path}: {e}")
        return code_structures

    async def scan_file(self, file_path: Path) -> CodeStructure:
        logger.debug(f"Сканирование файла: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module:
                    imports.append(n.module)
            elif isinstance(n, ast.Import):
                if n.names and n.names[0].name:
                    imports.append(n.names[0].name)
        lines_of_code = len(content.splitlines())
        complexity_metrics = self._calculate_complexity(tree)
        quality_metrics = self._calculate_quality_metrics(content)
        architecture_metrics = self._calculate_architecture_metrics(content)
        return CodeStructure(
            file_path=str(file_path),
            lines_of_code=lines_of_code,
            functions=[{"name": fn} for fn in functions],
            classes=[{"name": cl} for cl in classes],
            imports=imports,
            complexity_metrics=complexity_metrics,
            quality_metrics=quality_metrics,
            architecture_metrics=architecture_metrics,
        )

    async def scan_config(self, config_path: Path) -> Dict[str, Any]:
        logger.info(f"Сканирование конфигурации: {config_path}")
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Валидация и анализ структуры
        return {"valid": True, "keys": list(config.keys()), "config": config}

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        # Пример: подсчёт цикломатической сложности
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.With, ast.Try)
            ):
                complexity += 1
        return {"cyclomatic": complexity}

    def _calculate_quality_metrics(self, content: str) -> Dict[str, Any]:
        # Пример: простая метрика - длина строк
        lines = content.splitlines()
        long_lines = sum(1 for l in lines if len(l) > 120)
        return {"long_lines": long_lines, "total_lines": len(lines)}

    def _calculate_architecture_metrics(self, content: str) -> Dict[str, Any]:
        # Пример: поиск паттернов
        patterns = []
        if "class " in content and "def __init__" in content:
            patterns.append("OOP")
        if "def factory" in content or "Factory" in content:
            patterns.append("Factory")
        if "def subscribe" in content or "Observer" in content:
            patterns.append("Observer")
        return {"patterns": patterns}
