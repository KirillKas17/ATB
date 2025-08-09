#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор domain слоя.
Использует базовый класс для устранения дублирования кода.
"""

import ast
from pathlib import Path
from typing import List, Union

from shared.abstractions.base_analyzer import BaseAnalyzer, ProblemFunction


class DomainLayerAnalyzer(BaseAnalyzer):
    """Анализатор domain слоя."""
    
    def __init__(self):
        super().__init__("Domain")
    
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализ файла domain слоя."""
        issues = []
        
        tree = self._parse_file(file_path)
        if not tree:
            return issues
        
        # Устанавливаем родительские ссылки для навигации по AST
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        # Читаем содержимое файла для проверки TODO комментариев
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception:
            file_content = ""
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.stats["total_functions"] += 1
                
                # Получаем код функции для анализа комментариев
                function_code = ""
                try:
                    lines = file_content.split('\n')
                    function_code = '\n'.join(lines[node.lineno-1:node.end_lineno])
                except (IndexError, AttributeError):
                    pass
                
                # Проверяем различные проблемы
                issues.extend(self._check_todo_comments(node, file_path, function_code))
                issues.extend(self._check_simple_returns(node, file_path))
                issues.extend(self._check_unused_parameters(node, file_path))
                issues.extend(self._check_domain_specific_issues(node, file_path))
        
        return issues
    
    def _check_domain_specific_issues(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверка специфичных для domain слоя проблем."""
        issues = []
        
        # Проверяем, что domain функции не содержат инфраструктурную логику
        infrastructure_keywords = [
            'database', 'http', 'api', 'network', 'file', 'socket', 'thread',
            'process', 'memory', 'cache', 'redis', 'postgres', 'mysql'
        ]
        
        function_name_lower = node.name.lower()
        for keyword in infrastructure_keywords:
            if keyword in function_name_lower:
                issues.append(ProblemFunction(
                    file_path=file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    issue_type="Infrastructure Logic in Domain",
                    description=f"Функция содержит инфраструктурный термин '{keyword}' в domain слое",
                    severity="high"
                ))
        
        # Проверяем, что domain функции содержат бизнес-логику
        if not self._is_business_logic_function(node):
            issues.append(ProblemFunction(
                file_path=file_path,
                function_name=node.name,
                line_number=node.lineno,
                issue_type="Non-Business Logic Function",
                description="Функция в domain слое должна содержать бизнес-логику",
                severity="medium"
            ))
        
        return issues
    
    def _is_business_logic_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверяет, содержит ли функция бизнес-логику."""
        business_logic_patterns = [
            'calculate_', 'validate_', 'process_', 'analyze_', 'evaluate_',
            'determine_', 'compute_', 'estimate_', 'predict_', 'forecast_',
            'optimize_', 'balance_', 'allocate_', 'distribute_', 'aggregate_'
        ]
        
        function_name_lower = node.name.lower()
        return any(pattern in function_name_lower for pattern in business_logic_patterns)

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

    def analyze_domain_layer(self) -> List[ProblemFunction]:
        """Анализировать весь domain слой."""
        all_issues: List[ProblemFunction] = []
        
        if not self.domain_dir.exists():
            print(f"Директория {self.domain_dir} не найдена")
            return all_issues
        
        python_files = self.find_python_files(self.domain_dir)
        print(f"Найдено {len(python_files)} Python файлов в domain слое")
        
        for file_path in python_files:
            print(f"Анализирую: {file_path.relative_to(self.project_root)}")
            issues = self.analyze_file(file_path)
            all_issues.extend(issues)
        
        return all_issues

    def generate_report(self, issues: List[ProblemFunction]) -> str:
        """Сгенерировать отчет."""
        if not issues:
            return "[OK] Не найдено проблемных функций в domain слое"
        
        report = []
        report.append("# Отчет: Нереализованные функции в Domain слое")
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
    analyzer = DomainLayerAnalyzer()
    
    from shared.smart_logger import smart_print
    smart_print("Анализ domain слоя...", module="domain_analyzer")
    
    issues = analyzer.analyze_domain_layer()
    
    report = analyzer.generate_report(issues)
    
    # Сохраняем отчет в файл
    report_file = "domain_issues_report.md"
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
