#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор infrastructure слоя.
Использует базовый класс для устранения дублирования кода.
"""

import ast
from pathlib import Path
from typing import List, Union

from shared.abstractions.base_analyzer import BaseAnalyzer, ProblemFunction


class InfrastructureLayerAnalyzer(BaseAnalyzer):
    """Анализатор infrastructure слоя."""
    
    def __init__(self):
        super().__init__("Infrastructure")
    
    def analyze_file(self, file_path: Path) -> List[ProblemFunction]:
        """Анализ файла infrastructure слоя."""
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
                issues.extend(self._check_infrastructure_specific_issues(node, file_path))
        
        return issues
    
    def _check_infrastructure_specific_issues(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: Path) -> List[ProblemFunction]:
        """Проверка специфичных для infrastructure слоя проблем."""
        issues = []
        
        # Проверяем, что infrastructure функции не содержат бизнес-логику
        business_logic_keywords = [
            'calculate_profit', 'validate_order', 'process_trade', 'analyze_market',
            'evaluate_risk', 'determine_position', 'compute_returns'
        ]
        
        function_name_lower = node.name.lower()
        for keyword in business_logic_keywords:
            if keyword in function_name_lower:
                issues.append(ProblemFunction(
                    file_path=file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    issue_type="Business Logic in Infrastructure",
                    description=f"Функция содержит бизнес-логику '{keyword}' в infrastructure слое",
                    severity="high"
                ))
        
        # Проверяем, что infrastructure функции являются техническими
        if not self._is_infrastructure_function(node):
            issues.append(ProblemFunction(
                file_path=file_path,
                function_name=node.name,
                line_number=node.lineno,
                issue_type="Non-Infrastructure Function",
                description="Функция в infrastructure слое должна быть технической",
                severity="medium"
            ))
        
        return issues
    
    def _is_infrastructure_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Проверяет, является ли функция infrastructure функцией."""
        infrastructure_patterns = [
            'connect_', 'send_', 'receive_', 'store_', 'load_', 'save_',
            'fetch_', 'retrieve_', 'persist_', 'serialize_', 'deserialize_',
            'encode_', 'decode_', 'encrypt_', 'decrypt_', 'compress_', 'decompress_'
        ]
        
        function_name_lower = node.name.lower()
        return any(pattern in function_name_lower for pattern in infrastructure_patterns) 