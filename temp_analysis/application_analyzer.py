#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор application слоя.
Использует базовый класс для устранения дублирования кода.
"""

import ast
from pathlib import Path
from typing import List, Union

from shared.abstractions.base_analyzer import BaseAnalyzer, ProblemFunction


class ApplicationLayerAnalyzer(BaseAnalyzer):
    """Анализатор application слоя."""
    
    def __init__(self):
        super().__init__("Application")
    
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализ файла application слоя."""
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
                issues.extend(self._check_application_specific_issues(node, file_path))
        
        return issues
    
    def _check_application_specific_issues(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверка специфичных для application слоя проблем."""
        issues = []
        
        # Проверяем, что application функции являются use cases или сервисами
        if not self._is_application_function(node):
            issues.append(ProblemFunction(
                file_path=file_path,
                function_name=node.name,
                line_number=node.lineno,
                issue_type="Non-Application Function",
                description="Функция в application слое должна быть use case или сервисом",
                severity="medium"
            ))
        
        return issues
    
    def _is_application_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверяет, является ли функция application функцией."""
        application_patterns = [
            'execute_', 'process_', 'handle_', 'manage_', 'coordinate_',
            'orchestrate_', 'mediate_', 'facilitate_', 'arrange_', 'organize_'
        ]
        
        function_name_lower = node.name.lower()
        return any(pattern in function_name_lower for pattern in application_patterns) 