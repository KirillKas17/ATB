#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор нереализованных функций в Infrastructure слое.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass

from shared_analyzer import SharedLayerAnalyzer, ProblemFunction


class InfrastructureLayerAnalyzer(SharedLayerAnalyzer):
    """Анализатор для infrastructure слоя."""
    
    def __init__(self, project_root: str = "."):
        super().__init__(project_root)
        self.infrastructure_dir = self.project_root / "infrastructure"
    
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализировать один файл."""
        issues: List[ProblemFunction] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return issues

        try:
            tree = ast.parse(content)
            issues.extend(self._analyze_ast(tree, file_path, lines))
        except SyntaxError:
            print(f"Ошибка парсинга AST в файле {file_path}")

        return issues

    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[ProblemFunction]:
        """Анализировать AST."""
        issues: List[ProblemFunction] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
            elif isinstance(node, ast.AsyncFunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
        
        return issues

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str]) -> List[ProblemFunction]:
        """Анализировать отдельную функцию."""
        issues: List[ProblemFunction] = []
        
        # Получаем код функции
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        function_lines = lines[start_line:end_line]
        function_code = '\n'.join(function_lines)
        
        # Проверяем различные проблемы
        issues.extend(self._check_empty_implementation(node, file_path, function_code))
        issues.extend(self._check_not_implemented(node, file_path, function_code))
        issues.extend(self._check_placeholder_comments(node, file_path, function_code))
        issues.extend(self._check_todo_comments(node, file_path, function_code))
        issues.extend(self._check_simple_returns(node, file_path, function_code))
        
        return issues

    def _check_empty_implementation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить пустые реализации."""
        issues: List[ProblemFunction] = []
        
        # Функция содержит только pass
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Pass)):
            
            if not self._is_abstract_method(node):
                issues.append(ProblemFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Пустая реализация",
                    description="Функция содержит только pass",
                    location=f"{file_path.relative_to(self.project_root)}:{node.lineno}"
                ))
        
        return issues

    def _check_not_implemented(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить NotImplementedError."""
        issues: List[ProblemFunction] = []
        
        for child in ast.walk(node):
            if (isinstance(child, ast.Raise) and 
                isinstance(child.exc, ast.Name) and 
                child.exc.id == 'NotImplementedError'):
                
                issues.append(ProblemFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=child.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Не реализовано",
                    description="Функция вызывает NotImplementedError",
                    location=f"{file_path.relative_to(self.project_root)}:{child.lineno}"
                ))
                break
        
        return issues

    def _check_placeholder_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить комментарии-заглушки."""
        issues: List[ProblemFunction] = []
        
        placeholder_patterns = [
            (r'#.*заглушка', "Заглушка"),
            (r'#.*временно', "Временная реализация"),
            (r'#.*простая', "Простая реализация"),
            (r'#.*базовая', "Базовая реализация"),
            (r'#.*placeholder', "Placeholder"),
            (r'#.*stub', "Stub"),
            (r'#.*dummy', "Dummy"),
            (r'#.*fake', "Fake"),
            (r'#.*mock', "Mock"),
        ]
        
        for pattern, description in placeholder_patterns:
            if re.search(pattern, function_code, re.IGNORECASE):
                issues.append(ProblemFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Заглушка",
                    description=f"Функция содержит комментарий: {description}",
                    location=f"{file_path.relative_to(self.project_root)}:{node.lineno}"
                ))
                break
        
        return issues

    def _check_todo_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить TODO комментарии."""
        issues: List[ProblemFunction] = []
        
        todo_patterns = [
            (r'#.*TODO.*implement', "TODO - требуется реализация"),
            (r'#.*FIXME.*implement', "FIXME - требуется исправление"),
            (r'#.*HACK', "HACK - временное решение"),
        ]
        
        for pattern, description in todo_patterns:
            if re.search(pattern, function_code, re.IGNORECASE):
                issues.append(ProblemFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="TODO/FIXME",
                    description=description,
                    location=f"{file_path.relative_to(self.project_root)}:{node.lineno}"
                ))
                break
        
        return issues

    def _check_simple_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, function_code: str) -> List[ProblemFunction]:
        """Проверить простые возвраты."""
        issues: List[ProblemFunction] = []
        
        # Только если функция действительно простая
        if len(function_code.split('\n')) <= 5:
            simple_returns = [
                (r'return\s+None\s*$', "return None"),
                (r'return\s+0\s*$', "return 0"),
                (r'return\s+False\s*$', "return False"),
                (r'return\s+True\s*$', "return True"),
                (r'return\s+\[\]\s*$', "return []"),
                (r'return\s+\{\}\s*$', "return {}"),
            ]
            
            for pattern, description in simple_returns:
                if re.search(pattern, function_code, re.MULTILINE):
                    if not self._is_valid_simple_return(node):
                        issues.append(ProblemFunction(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=node.lineno,
                            function_name=node.name,
                            class_name=self._get_class_name(node),
                            issue_type="Простой возврат",
                            description=f"Функция возвращает {description}",
                            location=f"{file_path.relative_to(self.project_root)}:{node.lineno}"
                        ))
                        break
        
        return issues

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

    def analyze_infrastructure_layer(self) -> List[ProblemFunction]:
        """Анализировать весь infrastructure слой."""
        all_issues: List[ProblemFunction] = []
        
        if not self.infrastructure_dir.exists():
            print(f"Директория {self.infrastructure_dir} не найдена")
            return all_issues
        
        python_files = self.find_python_files(self.infrastructure_dir)
        print(f"Найдено {len(python_files)} Python файлов в infrastructure слое")
        
        for file_path in python_files:
            print(f"Анализирую: {file_path.relative_to(self.project_root)}")
            issues = self.analyze_file(file_path)
            all_issues.extend(issues)
        
        return all_issues

    def generate_report(self, issues: List[ProblemFunction]) -> str:
        """Сгенерировать отчет."""
        if not issues:
            return "[OK] Не найдено проблемных функций в infrastructure слое"
        
        report = []
        report.append("# Отчет: Нереализованные функции в Infrastructure слое")
        report.append(f"## Общая статистика")
        report.append(f"- Всего найдено проблем: {len(issues)}")
        
        # Группировка по типам
        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        report.append("### Распределение по типам:")
        for issue_type, count in sorted(type_counts.items()):
            report.append(f"- {issue_type}: {count}")
        
        report.append("\n## [Список] Список проблемных функций:")
        report.append("")
        report.append("| Файл | Строка | Функция | Класс | Тип проблемы | Описание |")
        report.append("|------|--------|---------|-------|--------------|----------|")
        
        for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
            class_name = issue.class_name or "-"
            report.append(f"| {issue.file_path} | {issue.line_number} | {issue.function_name} | {class_name} | {issue.issue_type} | {issue.description} |")
        
        return '\n'.join(report)


def main() -> None:
    """Основная функция."""
    analyzer = InfrastructureLayerAnalyzer()
    
    print("Анализ infrastructure слоя...")
    
    issues = analyzer.analyze_infrastructure_layer()
    
    report = analyzer.generate_report(issues)
    
    # Сохраняем отчет в файл
    report_file = "infrastructure_issues_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[Отчет] Отчет сохранен в файл: {report_file}")
    print(f"[Статистика] Найдено проблем: {len(issues)}")
    
    if issues:
        print("\n[Проблемы] Найдены проблемы:")
        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        for issue_type, count in sorted(type_counts.items()):
            print(f"  - {issue_type}: {count}")
        
        print(f"\n[Список] Список конкретных проблемных функций:")
        for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
            class_info = f" (класс: {issue.class_name})" if issue.class_name else ""
            print(f"  - {issue.location}: {issue.function_name}{class_info} - {issue.issue_type}")
    else:
        print("\n[OK] Проблем не найдено!")


if __name__ == "__main__":
    main() 