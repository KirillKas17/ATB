#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор shared слоя.
Использует базовый класс для устранения дублирования кода.
"""

import ast
from pathlib import Path
from typing import List, Union

from shared.abstractions.base_analyzer import BaseAnalyzer, ProblemFunction


class SharedLayerAnalyzer(BaseAnalyzer):
    """Анализатор shared слоя."""
    
    def __init__(self):
        super().__init__("Shared")
    
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализ файла shared слоя."""
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
                issues.extend(self._check_shared_specific_issues(node, file_path))
        
        return issues
    
    def _check_shared_specific_issues(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверка специфичных для shared слоя проблем."""
        issues = []
        
        # Проверяем, что функции shared слоя не содержат бизнес-логику
        business_logic_keywords = [
            'order', 'trade', 'position', 'strategy', 'market', 'portfolio',
            'risk', 'profit', 'loss', 'buy', 'sell', 'execute'
        ]
        
        function_name_lower = node.name.lower()
        for keyword in business_logic_keywords:
            if keyword in function_name_lower:
                issues.append(ProblemFunction(
                    file_path=file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    issue_type="Business Logic in Shared",
                    description=f"Функция содержит бизнес-термин '{keyword}' в shared слое",
                    severity="high"
                ))
        
        # Проверяем, что функции shared слоя являются утилитарными
        if not self._is_utility_function(node):
            issues.append(ProblemFunction(
                file_path=file_path,
                function_name=node.name,
                line_number=node.lineno,
                issue_type="Non-Utility Function",
                description="Функция в shared слое должна быть утилитарной",
                severity="medium"
            ))
        
        return issues
    
    def _is_utility_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверяет, является ли функция утилитарной."""
        utility_patterns = [
            'get_', 'set_', 'is_', 'has_', 'validate_', 'format_', 'parse_',
            'convert_', 'transform_', 'calculate_', 'compute_', 'generate_',
            'create_', 'build_', 'make_', 'prepare_', 'process_', 'handle_'
        ]
        
        function_name_lower = node.name.lower()
        return any(pattern in function_name_lower for pattern in utility_patterns) 