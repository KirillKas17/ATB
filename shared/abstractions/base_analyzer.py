"""
Базовый класс для анализаторов кода.
Устраняет дублирование кода между различными анализаторами.
"""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class ProblemFunction:
    """Представляет проблемную функцию."""
    
    def __init__(
        self,
        file_path: Path,
        function_name: str,
        line_number: int,
        issue_type: str,
        description: str,
        severity: str = "medium"
    ):
        self.file_path = file_path
        self.function_name = function_name
        self.line_number = line_number
        self.issue_type = issue_type
        self.description = description
        self.severity = severity


class BaseAnalyzer(ABC):
    """Базовый класс для всех анализаторов кода."""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.issues: List[ProblemFunction] = []
        self.stats = {
            "total_functions": 0,
            "problem_functions": 0,
            "files_analyzed": 0
        }
    
    @abstractmethod
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализ конкретного файла."""
        pass
    
    def analyze_directory(self, directory_path: Path) -> List[ProblemFunction]:
        """Анализ всей директории."""
        logger.info(f"Анализ директории: {directory_path}")
        
        all_issues = []
        python_files = list(directory_path.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
                self.stats["files_analyzed"] += 1
            except Exception as e:
                logger.error(f"Ошибка анализа файла {file_path}: {e}")
        
        self.issues = all_issues
        self.stats["problem_functions"] = len(all_issues)
        
        return all_issues
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Проверка, нужно ли пропустить файл."""
        skip_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            "node_modules",
            "tests",
            "test_",
            "_test.py"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Парсинг Python файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content)
        except Exception as e:
            logger.error(f"Ошибка парсинга файла {file_path}: {e}")
            return None
    
    def _is_abstract_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверить, является ли метод абстрактным."""
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Name) and 
                decorator.id == 'abstractmethod'):
                return True
            elif (isinstance(decorator, ast.Attribute) and 
                  decorator.attr == 'abstractmethod'):
                return True
        
        if node.name.startswith('_'):
            return True
        
        return False
    
    def _is_valid_simple_return(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверить, является ли простой возврат валидным."""
        if node.name.startswith('get_') or node.name.startswith('set_'):
            return True
        
        if node.name == '__init__':
            return True
        
        if self._is_abstract_method(node):
            return True
        
        return False
    
    def _get_class_name(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Получить имя класса для функции."""
        parent = getattr(node, 'parent', None)
        while parent:
            if isinstance(parent, ast.ClassDef):
                return parent.name
            parent = getattr(parent, 'parent', None)
        return None
    
    def _check_todo_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить TODO комментарии."""
        issues = []
        
        todo_patterns = [
            (r'#.*TODO.*implement', "TODO - требуется реализация"),
            (r'#.*FIXME.*implement', "FIXME - требуется исправление"),
            (r'#.*HACK', "HACK - временное решение"),
        ]
        
        for pattern, description in todo_patterns:
            import re
            if re.search(pattern, function_code, re.IGNORECASE):
                issues.append(ProblemFunction(
                    file_path=file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    issue_type="TODO/FIXME",
                    description=description,
                    severity="medium"
                ))
        
        return issues
    
    def _check_simple_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверить простые возвраты."""
        issues = []
        
        if self._is_valid_simple_return(node):
            return issues
        
        # Проверяем, есть ли только простые возвраты
        has_complex_logic = False
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                has_complex_logic = True
                break
        
        if not has_complex_logic and len(node.body) <= 2:
            # Проверяем, есть ли только return или pass
            simple_returns = 0
            for stmt in node.body:
                if isinstance(stmt, ast.Return) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
                    simple_returns += 1
                elif isinstance(stmt, ast.Pass):
                    simple_returns += 1
            
            if simple_returns == len(node.body):
                issues.append(ProblemFunction(
                    file_path=file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    issue_type="Simple Return",
                    description="Функция содержит только простые возвраты",
                    severity="low"
                ))
        
        return issues
    
    def _check_unused_parameters(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверить неиспользуемые параметры."""
        issues = []
        
        if not node.args.args:
            return issues
        
        # Получаем имена параметров
        param_names = {arg.arg for arg in node.args.args}
        
        # Исключаем self и cls
        param_names.discard('self')
        param_names.discard('cls')
        
        if not param_names:
            return issues
        
        # Проверяем использование параметров в теле функции
        used_names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in param_names:
                used_names.add(child.id)
        
        unused_params = param_names - used_names
        
        for param in unused_params:
            issues.append(ProblemFunction(
                file_path=file_path,
                function_name=node.name,
                line_number=node.lineno,
                issue_type="Unused Parameter",
                description=f"Неиспользуемый параметр: {param}",
                severity="medium"
            ))
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику анализа."""
        return {
            "layer": self.layer_name,
            "total_files": self.stats["files_analyzed"],
            "total_issues": self.stats["problem_functions"],
            "issues_by_type": self._group_issues_by_type(),
            "issues_by_severity": self._group_issues_by_severity()
        }
    
    def _group_issues_by_type(self) -> Dict[str, int]:
        """Группировка проблем по типу."""
        grouped = {}
        for issue in self.issues:
            grouped[issue.issue_type] = grouped.get(issue.issue_type, 0) + 1
        return grouped
    
    def _group_issues_by_severity(self) -> Dict[str, int]:
        """Группировка проблем по серьезности."""
        grouped = {}
        for issue in self.issues:
            grouped[issue.severity] = grouped.get(issue.severity, 0) + 1
        return grouped
    
    def generate_report(self) -> str:
        """Генерация отчета."""
        stats = self.get_statistics()
        
        report = f"""
# Отчет анализа {self.layer_name} слоя

## Общая статистика
- Файлов проанализировано: {stats['total_files']}
- Всего проблем: {stats['total_issues']}

## Проблемы по типу
"""
        
        for issue_type, count in stats['issues_by_type'].items():
            report += f"- {issue_type}: {count}\n"
        
        report += "\n## Проблемы по серьезности\n"
        for severity, count in stats['issues_by_severity'].items():
            report += f"- {severity}: {count}\n"
        
        if self.issues:
            report += "\n## Детальный список проблем\n"
            for issue in self.issues:
                report += f"- {issue.file_path}:{issue.line_number} - {issue.function_name} ({issue.issue_type}): {issue.description}\n"
        
        return report 