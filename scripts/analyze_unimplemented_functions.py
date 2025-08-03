#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа нереализованных и упрощенных функций в application слое.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
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


class ApplicationLayerAnalyzer:
    """Анализатор application слоя для поиска нереализованного функционала."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.application_dir = self.project_root / "application"
        
        # Триггерные слова для поиска нереализованного функционала
        self.trigger_words = {
            "простая", "простое", "простые", "упрощено", "упрощенная", 
            "упрощенные", "упрощенный", "здесь должна", "здесь будет", 
            "здесь можно", "здесь", "базовая", "временно", "заглушка", 
            "заглушку", "todo", "pass", "notimplemented", "not implemented",
            "raise notimplementederror", "return none", "return 0", 
            "return false", "return true", "return []", "return {}",
            "return ''", "return 0.0", "placeholder", "stub", "mock",
            "dummy", "fake", "temporary", "basic", "simple", "minimal",
            "filler", "empty", "void", "null", "undefined", "default",
            "example", "sample", "test", "debug", "fixme", "hack",
            "workaround", "temporary solution", "quick fix", "band-aid"
        }
        
        # Паттерны для поиска абстрактных методов
        self.abstract_patterns = [
            r"@abstractmethod",
            r"raise NotImplementedError",
            r"pass\s*#.*abstract",
            r"pass\s*#.*implement",
            r"#.*abstract.*method",
            r"#.*implement.*later"
        ]
        
        # Паттерны для поиска упрощенных реализаций
        self.simplified_patterns = [
            r"return\s+(None|0|False|True|\[\]|\{\}|''|0\.0)\s*#.*(простая|базовая|временно)",
            r"pass\s*#.*(заглушка|временно|простая)",
            r"#.*(простая|базовая|временно|заглушка).*реализация",
            r"#.*(todo|fixme).*implement",
            r"#.*(temporary|basic|simple).*implementation"
        ]

    def find_python_files(self, directory: Path) -> List[Path]:
        """Найти все Python файлы в директории."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Исключаем __pycache__, .git, venv и другие служебные директории
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def analyze_file(self, file_path: Path) -> List[UnimplementedFunction]:
        """Анализировать один файл на предмет нереализованного функционала."""
        issues: List[UnimplementedFunction] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return issues

        # Анализ по строкам
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Поиск триггерных слов
            for trigger in self.trigger_words:
                if trigger in line_lower:
                    # Получаем контекст (несколько строк до и после)
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 3)
                    context = '\n'.join(lines[context_start-1:context_end])
                    
                    # Определяем тип проблемы
                    issue_type = self._determine_issue_type(line_lower, trigger)
                    
                    # Пытаемся найти имя функции
                    function_name, class_name = self._extract_function_info(lines, line_num)
                    
                    issues.append(UnimplementedFunction(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        function_name=function_name or "Unknown",
                        class_name=class_name,
                        issue_type=issue_type,
                        description=f"Найдено триггерное слово: '{trigger}'",
                        code_snippet=context
                    ))
                    break

        # Анализ AST для поиска абстрактных методов
        try:
            tree = ast.parse(content)
            issues.extend(self._analyze_ast(tree, file_path, lines))
        except SyntaxError:
            print(f"Ошибка парсинга AST в файле {file_path}")

        return issues

    def _determine_issue_type(self, line: str, trigger: str) -> str:
        """Определить тип проблемы на основе триггерного слова."""
        if trigger in ['todo', 'fixme', 'hack']:
            return "TODO/FIXME"
        elif trigger in ['pass', 'notimplemented', 'not implemented']:
            return "Не реализовано"
        elif trigger in ['простая', 'базовая', 'временно', 'заглушка']:
            return "Упрощенная реализация"
        elif trigger in ['placeholder', 'stub', 'dummy', 'fake']:
            return "Заглушка"
        elif trigger in ['return none', 'return 0', 'return false', 'return true']:
            return "Возврат по умолчанию"
        else:
            return "Подозрительная реализация"

    def _extract_function_info(self, lines: List[str], target_line: int) -> Tuple[Optional[str], Optional[str]]:
        """Извлечь информацию о функции и классе."""
        function_name = None
        class_name = None
        
        # Ищем определение функции выше текущей строки
        for i in range(target_line - 1, max(0, target_line - 20), -1):
            line = lines[i-1].strip()
            
            # Поиск определения класса
            if line.startswith('class ') and ':' in line:
                match = re.match(r'class\s+(\w+)', line)
                if match:
                    class_name = match.group(1)
            
            # Поиск определения функции
            if line.startswith('def ') and ':' in line:
                match = re.match(r'def\s+(\w+)', line)
                if match:
                    function_name = match.group(1)
                    break
                    
            # Поиск async функции
            elif line.startswith('async def ') and ':' in line:
                match = re.match(r'async\s+def\s+(\w+)', line)
                if match:
                    function_name = match.group(1)
                    break
        
        return function_name, class_name

    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[UnimplementedFunction]:
        """Анализировать AST для поиска абстрактных методов."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Проверяем тело функции на упрощенные реализации
                body_lines = [lines[child.lineno-1] for child in node.body if hasattr(child, 'lineno')]
                body_text = '\n'.join(body_lines).lower()
                
                # Поиск pass, return None, raise NotImplementedError
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Pass)):
                    issues.append(UnimplementedFunction(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=node.lineno,
                        function_name=node.name,
                        class_name=self._get_class_name(node),
                        issue_type="Пустая реализация",
                        description="Функция содержит только pass",
                        code_snippet=self._get_context(lines, node.lineno, 3)
                    ))
                
                elif any(pattern in body_text for pattern in ['return none', 'return 0', 'return false', 'return true']):
                    issues.append(UnimplementedFunction(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=node.lineno,
                        function_name=node.name,
                        class_name=self._get_class_name(node),
                        issue_type="Возврат по умолчанию",
                        description="Функция возвращает значение по умолчанию",
                        code_snippet=self._get_context(lines, node.lineno, 5)
                    ))
                
                # Поиск raise NotImplementedError
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
                            code_snippet=self._get_context(lines, child.lineno, 3)
                        ))
                        break

        return issues

    def _get_class_name(self, node: ast.FunctionDef) -> Optional[str]:
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
        """Сгенерировать отчет по найденным проблемам."""
        if not issues:
            return "✅ Не найдено нереализованных функций в application слое"
        
        report = []
        report.append("# Отчет по нереализованным функциям в Application слое")
        report.append(f"## Общая статистика")
        report.append(f"- Всего найдено проблем: {len(issues)}")
        
        # Группировка по типам
        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        report.append("### Распределение по типам:")
        for issue_type, count in sorted(type_counts.items()):
            report.append(f"- {issue_type}: {count}")
        
        report.append("\n## Детальный список проблем:")
        
        # Группировка по файлам
        files_issues: Dict[str, List[UnimplementedFunction]] = {}
        for issue in issues:
            if issue.file_path not in files_issues:
                files_issues[issue.file_path] = []
            files_issues[issue.file_path].append(issue)
        
        for file_path, file_issues in sorted(files_issues.items()):
            report.append(f"\n### 📁 {file_path}")
            report.append(f"Найдено проблем: {len(file_issues)}")
            
            for issue in sorted(file_issues, key=lambda x: x.line_number):
                report.append(f"\n#### Строка {issue.line_number}: {issue.function_name}")
                if issue.class_name:
                    report.append(f"**Класс:** {issue.class_name}")
                report.append(f"**Тип:** {issue.issue_type}")
                report.append(f"**Описание:** {issue.description}")
                report.append("**Код:**")
                report.append("```python")
                report.append(issue.code_snippet)
                report.append("```")
        
        return '\n'.join(report)


def main() -> None:
    """Основная функция."""
    analyzer = ApplicationLayerAnalyzer()
    
    print("🔍 Анализ application слоя на предмет нереализованного функционала...")
    
    issues = analyzer.analyze_application_layer()
    
    report = analyzer.generate_report(issues)
    
    # Сохраняем отчет в файл
    report_file = "application_unimplemented_functions_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Отчет сохранен в файл: {report_file}")
    print(f"📈 Найдено проблем: {len(issues)}")
    
    # Выводим краткую сводку
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