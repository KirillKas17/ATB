#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальный анализ application слоя с поиском конкретных проблемных функций.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class FunctionIssue:
    """Информация о проблемной функции."""
    file_path: str
    line_number: int
    function_name: str
    class_name: Optional[str]
    issue_type: str
    description: str
    code_snippet: str
    full_function_code: str


class DetailedAnalyzer:
    """Детальный анализатор функций."""
    
    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root)
        self.application_dir = self.project_root / "application"

    def find_python_files(self, directory: Path) -> List[Path]:
        """Найти все Python файлы в директории."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    python_files.append(Path(root) / file)
        return python_files

    def analyze_file(self, file_path: Path) -> List[FunctionIssue]:
        return []
        """Анализировать один файл."""
        issues: List[FunctionIssue] = []
        
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

    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[FunctionIssue]:
        return []
        """Анализировать AST."""
        issues: List[FunctionIssue] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
            elif isinstance(node, ast.AsyncFunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
        
        return issues

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str]) -> List[FunctionIssue]:
        return []
        """Анализировать отдельную функцию."""
        issues: List[FunctionIssue] = []
        
        # Получаем полный код функции
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        function_lines = lines[start_line:end_line]
        full_function_code = '\n'.join(function_lines)
        
        # Проверяем различные проблемы
        issues.extend(self._check_empty_body(node, file_path, lines, full_function_code))
        issues.extend(self._check_simple_returns(node, file_path, lines, full_function_code))
        issues.extend(self._check_not_implemented(node, file_path, lines, full_function_code))
        issues.extend(self._check_placeholder_comments(node, file_path, lines, full_function_code))
        issues.extend(self._check_todo_comments(node, file_path, lines, full_function_code))
        
        return issues

    def _check_empty_body(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], full_code: str) -> List[FunctionIssue]:
        return []
        """Проверить пустое тело функции."""
        issues: List[FunctionIssue] = []
        
        # Функция содержит только pass
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Pass)):
            
            if not self._is_abstract_method(node):
                issues.append(FunctionIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Пустая реализация",
                    description="Функция содержит только pass",
                    code_snippet=self._get_context(lines, node.lineno, 3),
                    full_function_code=full_code
                ))
        
        # Функция пустая
        elif len(node.body) == 0:
            issues.append(FunctionIssue(
                file_path=str(file_path.relative_to(self.project_root)),
                line_number=node.lineno,
                function_name=node.name,
                class_name=self._get_class_name(node),
                issue_type="Пустая реализация",
                description="Функция не имеет тела",
                code_snippet=self._get_context(lines, node.lineno, 3),
                full_function_code=full_code
            ))
        
        return issues

    def _check_simple_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], full_code: str) -> List[FunctionIssue]:
        """Проверить простые возвраты."""
        issues: List[FunctionIssue] = []
        
        # Паттерны простых возвратов
        simple_returns = [
            (r'return\s+None\s*$', "return None"),
            (r'return\s+0\s*$', "return 0"),
            (r'return\s+False\s*$', "return False"),
            (r'return\s+True\s*$', "return True"),
            (r'return\s+\[\]\s*$', "return []"),
            (r'return\s+\{\}\s*$', "return {}"),
            (r'return\s+""\s*$', 'return ""'),
            (r"return\s+''\s*$", "return ''"),
        ]
        
        for pattern, description in simple_returns:
            if re.search(pattern, full_code, re.MULTILINE):
                if not self._is_valid_simple_return(node, full_code):
                    issues.append(FunctionIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=node.lineno,
                        function_name=node.name,
                        class_name=self._get_class_name(node),
                        issue_type="Простой возврат",
                        description=f"Функция содержит {description}",
                        code_snippet=self._get_context(lines, node.lineno, 3),
                        full_function_code=full_code
                    ))
                break
        
        return issues

    def _check_not_implemented(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], full_code: str) -> List[FunctionIssue]:
        return []
        """Проверить NotImplementedError."""
        issues: List[FunctionIssue] = []
        
        if 'NotImplementedError' in full_code or 'raise NotImplementedError' in full_code:
            issues.append(FunctionIssue(
                file_path=str(file_path.relative_to(self.project_root)),
                line_number=node.lineno,
                function_name=node.name,
                class_name=self._get_class_name(node),
                issue_type="NotImplementedError",
                description="Функция содержит NotImplementedError",
                code_snippet=self._get_context(lines, node.lineno, 3),
                full_function_code=full_code
            ))
        
        return issues

    def _check_placeholder_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], full_code: str) -> List[FunctionIssue]:
        return []
        """Проверить комментарии-заглушки."""
        issues: List[FunctionIssue] = []
        
        placeholder_patterns = [
            r'#\s*placeholder',
            r'#\s*заглушка',
            r'#\s*stub',
            r'#\s*dummy',
            r'#\s*temp',
            r'#\s*временно',
            r'#\s*TODO',
            r'#\s*FIXME',
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, full_code, re.IGNORECASE):
                issues.append(FunctionIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Комментарий-заглушка",
                    description="Функция содержит комментарий-заглушку",
                    code_snippet=self._get_context(lines, node.lineno, 3),
                    full_function_code=full_code
                ))
                break
        
        return issues

    def _check_todo_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], full_code: str) -> List[FunctionIssue]:
        return []
        """Проверить TODO комментарии."""
        issues: List[FunctionIssue] = []
        
        todo_patterns = [
            r'#\s*TODO\s*:',
            r'#\s*TODO\s*-',
            r'#\s*FIXME\s*:',
            r'#\s*FIXME\s*-',
            r'#\s*HACK\s*:',
            r'#\s*HACK\s*-',
        ]
        
        for pattern in todo_patterns:
            if re.search(pattern, full_code, re.IGNORECASE):
                issues.append(FunctionIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="TODO комментарий",
                    description="Функция содержит TODO/FIXME/HACK комментарий",
                    code_snippet=self._get_context(lines, node.lineno, 3),
                    full_function_code=full_code
                ))
                break
        
        return issues

    def _is_abstract_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверить, является ли метод абстрактным."""
        if not hasattr(node, 'decorator_list'):
            return False
        
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Name) and 
                decorator.id == 'abstractmethod'):
                return True
        
        if node.name.startswith('_'):
            return True
        
        return False

    def _is_valid_simple_return(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], full_code: str) -> bool:
        """Проверить, является ли простой возврат валидным."""
        if node.name.startswith('get_') or node.name.startswith('set_'):
            return True
        
        if node.name == '__init__':
            return True
        
        if self._is_abstract_method(node):
            return True
        
        if len(full_code.split('\n')) > 5:
            return True
        
        return False

    def _get_class_name(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Получить имя класса для функции."""
        parent = getattr(node, 'parent', None)
        while parent:
            if isinstance(parent, ast.ClassDef):
                return parent.name

    def _get_context(self, lines: List[str], line_num: int, context_size: int) -> str:
        """Получить контекст вокруг строки."""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return '\n'.join(lines[start:end])

    def analyze_application_layer(self) -> List[FunctionIssue]:
        """Анализировать весь application слой."""
        all_issues: List[FunctionIssue] = []
        
        if not self.application_dir.exists():
            print(f"Директория {self.application_dir} не найдена")
            return all_issues
        
        python_files = self.find_python_files(self.application_dir)
        print(f"Найдено {len(python_files)} Python файлов в application слое")
        
        for file_path in python_files:
            print(f"Анализирую: {file_path.relative_to(self.project_root)}")
            issues = self.analyze_file(file_path)
            all_issues.extend(issues)
        
        return all_issues

    def generate_detailed_report(self, issues: List[FunctionIssue]) -> str:
        """Сгенерировать детальный отчет."""
        if not issues:
            return "✅ Не найдено проблемных функций в application слое"
        
        report = []
        report.append("# Детальный отчет по проблемным функциям в Application слое")
        report.append(f"## Общая статистика")
        report.append(f"- Всего найдено проблем: {len(issues)}")
        
        # Группировка по типам
        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        report.append("### Распределение по типам:")
        for issue_type, count in sorted(type_counts.items()):
            report.append(f"- {issue_type}: {count}")
        
        report.append("\n## 📋 Детальный список проблем:")
        
        # Группировка по файлам
        files_issues: Dict[str, List[FunctionIssue]] = {}
        for issue in issues:
            if issue.file_path not in files_issues:
                files_issues[issue.file_path] = []
            files_issues[issue.file_path].append(issue)
        
        for file_path, file_issues in sorted(files_issues.items()):
            report.append(f"\n### 📁 {file_path}")
            report.append(f"Найдено проблем: {len(file_issues)}")
            
            for issue in sorted(file_issues, key=lambda x: x.line_number):
                report.append(f"\n#### 🔍 Строка {issue.line_number}: {issue.function_name}")
                if issue.class_name:
                    report.append(f"**Класс:** `{issue.class_name}`")
                report.append(f"**Тип проблемы:** {issue.issue_type}")
                report.append(f"**Описание:** {issue.description}")
                report.append("**Контекст:**")
                report.append("```python")
                report.append(issue.code_snippet)
                report.append("```")
                report.append("**Полный код функции:**")
                report.append("```python")
                report.append(issue.full_function_code)
                report.append("```")
                report.append("---")
        
        return '\n'.join(report)


def main() -> None:
    """Основная функция."""
    analyzer = DetailedAnalyzer()
    
    print("🔍 Детальный анализ application слоя...")
    
    issues = analyzer.analyze_application_layer()
    
    report = analyzer.generate_detailed_report(issues)
    
    # Сохраняем отчет в файл
    report_file = "detailed_application_analysis.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Детальный отчет сохранен в файл: {report_file}")
    print(f"📈 Найдено проблем: {len(issues)}")
    
    if issues:
        print("\n🚨 Найдены проблемы:")
        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        for issue_type, count in sorted(type_counts.items()):
            print(f"  - {issue_type}: {count}")
    else:
        print("\n✅ Проблем не найдено!")


if __name__ == "__main__":
    main()