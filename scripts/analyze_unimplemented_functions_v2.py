#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный скрипт для анализа нереализованных и упрощенных функций в application слое.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class UnimplementedFunction:
    """Информация о нереализованной функции."""
    file_path: str
    line_number: int
    function_name: str
    class_name: Optional[str]
    issue_type: str
    description: str
    code_snippet: str
    severity: str  # high, medium, low


class ApplicationLayerAnalyzerV2:
    """Улучшенный анализатор application слоя."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.application_dir = self.project_root / "application"
        
        # Исключения - файлы, которые не нужно анализировать
        self.exclude_patterns = [
            r"__init__\.py$",
            r"__pycache__",
            r"\.pyc$",
            r"test_.*\.py$",
            r"_test\.py$"
        ]
        
        # Исключения для строк - не анализировать
        self.line_exclusions = [
            r"^#.*$",  # Комментарии
            r"^from.*import.*$",  # Импорты
            r"^import.*$",  # Импорты
            r"^class.*:$",  # Определения классов
            r"^def.*:$",  # Определения функций
            r"^async def.*:$",  # Определения async функций
            r"^@.*$",  # Декораторы
            r"^return.*$",  # Возвраты (анализируем отдельно)
            r"^pass$",  # Pass (анализируем отдельно)
            r"^raise.*$",  # Raise (анализируем отдельно)
        ]

    def find_python_files(self, directory: Path) -> List[Path]:
        """Найти все Python файлы в директории."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Исключаем служебные директории
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Проверяем исключения
                    if not any(re.search(pattern, str(file_path)) for pattern in self.exclude_patterns):
                        python_files.append(file_path)
        return python_files

    def analyze_file(self, file_path: Path) -> List[UnimplementedFunction]:
        """Анализировать один файл."""
        issues: List[UnimplementedFunction] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return issues

        # Анализ AST
        try:
            tree = ast.parse(content)
            issues.extend(self._analyze_ast(tree, file_path, lines))
        except SyntaxError:
            print(f"Ошибка парсинга AST в файле {file_path}")

        return issues

    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[UnimplementedFunction]:
        """Анализировать AST для поиска проблемных функций."""
        issues: List[UnimplementedFunction] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
            elif isinstance(node, ast.AsyncFunctionDef):
                issues.extend(self._analyze_function(node, file_path, lines))
        
        return issues

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str]) -> List[UnimplementedFunction]:
        """Анализировать отдельную функцию."""
        issues: List[UnimplementedFunction] = []
        
        # Получаем тело функции
        body_lines = []
        for child in node.body:
            if hasattr(child, 'lineno'):
                body_lines.append(lines[child.lineno-1])
        
        body_text = '\n'.join(body_lines)
        
        # Проверяем различные паттерны
        issues.extend(self._check_empty_implementation(node, file_path, lines))
        issues.extend(self._check_default_returns(node, file_path, lines, body_text))
        issues.extend(self._check_not_implemented(node, file_path, lines))
        issues.extend(self._check_simplified_implementation(node, file_path, lines, body_text))
        issues.extend(self._check_todo_comments(node, file_path, lines, body_text))
        
        return issues

    def _check_empty_implementation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str]) -> List[UnimplementedFunction]:
        """Проверить пустые реализации."""
        issues: List[UnimplementedFunction] = []
        
        # Функция содержит только pass
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Pass)):
            
            # Проверяем, не является ли это абстрактным методом
            if not self._is_abstract_method(node):
                issues.append(UnimplementedFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Пустая реализация",
                    description="Функция содержит только pass",
                    code_snippet=self._get_context(lines, node.lineno, 5),
                    severity="high"
                ))
        
        # Функция пустая (нет тела)
        elif len(node.body) == 0:
            issues.append(UnimplementedFunction(
                file_path=str(file_path.relative_to(self.project_root)),
                line_number=node.lineno,
                function_name=node.name,
                class_name=self._get_class_name(node),
                issue_type="Пустая реализация",
                description="Функция не имеет тела",
                code_snippet=self._get_context(lines, node.lineno, 3),
                severity="high"
            ))
        
        return issues

    def _check_default_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], body_text: str) -> List[UnimplementedFunction]:
        """Проверить возвраты по умолчанию."""
        issues: List[UnimplementedFunction] = []
        
        # Паттерны возвратов по умолчанию
        default_patterns = [
            (r'return\s+None\s*$', "return None"),
            (r'return\s+0\s*$', "return 0"),
            (r'return\s+False\s*$', "return False"),
            (r'return\s+True\s*$', "return True"),
            (r'return\s+\[\]\s*$', "return []"),
            (r'return\s+\{\}\s*$', "return {}"),
            (r'return\s+""\s*$', 'return ""'),
            (r"return\s+''\s*$", "return ''"),
            (r'return\s+0\.0\s*$', "return 0.0"),
        ]
        
        for pattern, description in default_patterns:
            if re.search(pattern, body_text, re.MULTILINE):
                # Проверяем, не является ли это валидным случаем
                if not self._is_valid_default_return(node, body_text):
                    issues.append(UnimplementedFunction(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=node.lineno,
                        function_name=node.name,
                        class_name=self._get_class_name(node),
                        issue_type="Возврат по умолчанию",
                        description=f"Функция возвращает {description}",
                        code_snippet=self._get_context(lines, node.lineno, 5),
                        severity="medium"
                    ))
                    break
        
        return issues

    def _check_not_implemented(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str]) -> List[UnimplementedFunction]:
        """Проверить NotImplementedError."""
        issues: List[UnimplementedFunction] = []
        
        for child in ast.walk(node):
            if (isinstance(child, ast.Raise) and 
                isinstance(child.exc, ast.Name) and 
                child.exc.id == 'NotImplementedError'):
                
                issues.append(UnimplementedFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=child.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Не реализовано",
                    description="Функция вызывает NotImplementedError",
                    code_snippet=self._get_context(lines, child.lineno, 3),
                    severity="high"
                ))
                break
        
        return issues

    def _check_simplified_implementation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], body_text: str) -> List[UnimplementedFunction]:
        """Проверить упрощенные реализации."""
        issues: List[UnimplementedFunction] = []
        
        # Паттерны упрощенных реализаций
        simplified_patterns = [
            (r'#.*простая.*реализация', "Простая реализация"),
            (r'#.*базовая.*реализация', "Базовая реализация"),
            (r'#.*временная.*реализация', "Временная реализация"),
            (r'#.*заглушка', "Заглушка"),
            (r'#.*placeholder', "Placeholder"),
            (r'#.*stub', "Stub"),
            (r'#.*dummy', "Dummy"),
            (r'#.*fake', "Fake"),
            (r'#.*mock', "Mock"),
        ]
        
        for pattern, description in simplified_patterns:
            if re.search(pattern, body_text, re.IGNORECASE):
                issues.append(UnimplementedFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="Упрощенная реализация",
                    description=f"Функция содержит комментарий: {description}",
                    code_snippet=self._get_context(lines, node.lineno, 5),
                    severity="medium"
                ))
                break
        
        return issues

    def _check_todo_comments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path, lines: List[str], body_text: str) -> List[UnimplementedFunction]:
        """Проверить TODO комментарии."""
        issues: List[UnimplementedFunction] = []
        
        todo_patterns = [
            (r'#.*TODO.*implement', "TODO - требуется реализация"),
            (r'#.*FIXME.*implement', "FIXME - требуется исправление"),
            (r'#.*HACK', "HACK - временное решение"),
        ]
        
        for pattern, description in todo_patterns:
            if re.search(pattern, body_text, re.IGNORECASE):
                issues.append(UnimplementedFunction(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    function_name=node.name,
                    class_name=self._get_class_name(node),
                    issue_type="TODO/FIXME",
                    description=description,
                    code_snippet=self._get_context(lines, node.lineno, 5),
                    severity="low"
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
            elif (isinstance(decorator, ast.Attribute) and 
                  decorator.attr == 'abstractmethod'):
                return True
        
        if node.name.startswith('_'):
            return True
        
        return False

    def _is_valid_default_return(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], body_text: str) -> bool:
        """Проверить, является ли возврат по умолчанию валидным."""
        if node.name.startswith('get_') or node.name.startswith('set_'):
            return True
        
        if node.name == '__init__':
            return True
        
        if self._is_abstract_method(node):
            return True
        
        if len(body_text.split('\n')) > 5:
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

    def _get_context(self, lines: List[str], line_num: int, context_size: int) -> str:
        """Получить контекст вокруг строки."""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return '\n'.join(lines[start:end])

    def analyze_application_layer(self) -> List[UnimplementedFunction]:
        """Анализировать весь application слой."""
        all_issues: List[UnimplementedFunction] = []
        
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

    def generate_report(self, issues: List[UnimplementedFunction]) -> str:
        """Сгенерировать отчет."""
        if not issues:
            return "✅ Не найдено проблемных функций в application слое"
        
        report = []
        report.append("# Отчет: Нереализованные функции в Application слое")
        report.append(f"## Общая статистика")
        report.append(f"- Всего найдено проблем: {len(issues)}")
        
        # Группировка по типам
        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        report.append("### Распределение по типам:")
        for issue_type, count in sorted(type_counts.items()):
            report.append(f"- {issue_type}: {count}")
        
        report.append("### Распределение по серьезности:")
        for severity, count in sorted(severity_counts.items()):
            report.append(f"- {severity}: {count}")
        
        report.append("\n## 📋 Список проблемных функций:")
        report.append("")
        report.append("| Файл | Строка | Функция | Класс | Тип проблемы | Серьезность | Описание |")
        report.append("|------|--------|---------|-------|--------------|--------------|----------|")
        
        for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
            class_name = issue.class_name or "-"
            report.append(f"| {issue.file_path} | {issue.line_number} | {issue.function_name} | {class_name} | {issue.issue_type} | {issue.severity} | {issue.description} |")
        
        report.append("")
        report.append("## 🔍 Детализация по файлам:")
        
        # Группировка по файлам
        files_issues: Dict[str, List[UnimplementedFunction]] = {}
        for issue in issues:
            if issue.file_path not in files_issues:
                files_issues[issue.file_path] = []
            files_issues[issue.file_path].append(issue)
        
        for file_path, file_issues in sorted(files_issues.items()):
            report.append(f"\n### 📁 {file_path}")
            report.append(f"**Найдено проблем:** {len(file_issues)}")
            report.append("")
            
            for issue in sorted(file_issues, key=lambda x: x.line_number):
                class_info = f" (класс: {issue.class_name})" if issue.class_name else ""
                report.append(f"- **Строка {issue.line_number}:** `{issue.function_name}`{class_info}")
                report.append(f"  - Тип: {issue.issue_type}")
                report.append(f"  - Серьезность: {issue.severity}")
                report.append(f"  - Описание: {issue.description}")
                report.append("")
        
        return '\n'.join(report)


def main() -> None:
    """Основная функция."""
    analyzer = ApplicationLayerAnalyzerV2()
    
    print("🔍 Улучшенный анализ application слоя...")
    
    issues = analyzer.analyze_application_layer()
    
    report = analyzer.generate_report(issues)
    
    # Сохраняем отчет в файл
    report_file = "application_unimplemented_functions_v2_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Отчет сохранен в файл: {report_file}")
    print(f"📈 Найдено проблем: {len(issues)}")
    
    if issues:
        print("\n🚨 Найдены проблемы:")
        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        print("По типам:")
        for issue_type, count in sorted(type_counts.items()):
            print(f"  - {issue_type}: {count}")
        
        print("По серьезности:")
        for severity, count in sorted(severity_counts.items()):
            print(f"  - {severity}: {count}")
    else:
        print("\n✅ Проблем не найдено!")


if __name__ == "__main__":
    main() 