#!/usr/bin/env python3
"""
Скрипт для автоматического исправления ошибок mypy.
Исправляет основные типы ошибок:
1. Отсутствующие аннотации типов (no-untyped-def)
2. Отсутствующие импорты
3. Неправильные аргументы функций
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Any

def fix_test_file_annotations(file_path: str) -> None:
    """Исправляет аннотации типов в тестовом файле."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Добавляем необходимые импорты
    if 'from typing import' not in content:
        content = content.replace(
            'import pytest',
            'import pytest\nfrom typing import Any, Dict, List, Optional, Union, AsyncGenerator'
        )
    
    # Исправляем функции без аннотаций типов
    # Паттерн для функций без аннотаций
    patterns = [
        # async def test_*(
        (r'async def (test_\w+)\(([^)]*)\):', r'async def \1(\2) -> None:'),
        # def test_*(
        (r'def (test_\w+)\(([^)]*)\):', r'def \1(\2) -> None:'),
        # @pytest.fixture async def
        (r'async def (\w+)\(([^)]*)\):', r'async def \1(\2) -> Any:'),
        # @pytest.fixture def
        (r'def (\w+)\(([^)]*)\):', r'def \1(\2) -> Any:'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Исправляем AgentContext() без аргументов
    content = re.sub(
        r'AgentContext\(\)',
        'AgentContext(symbol="BTC/USDT")',
        content
    )
    
    # Добавляем импорты для часто используемых типов
    if 'import numpy as np' not in content and 'np.' in content:
        content = content.replace(
            'import pytest',
            'import pytest\nimport numpy as np'
        )
    
    if 'import pandas as pd' not in content and 'pd.' in content:
        content = content.replace(
            'import pytest',
            'import pytest\nimport pandas as pd'
        )
    
    # Исправляем импорты Mock
    if 'from unittest.mock import Mock' not in content and 'Mock(' in content:
        content = content.replace(
            'import pytest',
            'import pytest\nfrom unittest.mock import Mock, patch'
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_main_files() -> None:
    """Исправляет основные файлы проекта."""
    files_to_fix = [
        'main.py',
        'application/di_container_refactored.py',
        'infrastructure/core/system_integration.py',
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Добавляем аннотации типов для функций
            content = re.sub(
                r'def (\w+)\(([^)]*)\):',
                r'def \1(\2) -> None:',
                content
            )
            
            # Исправляем Task без типовых параметров
            content = re.sub(
                r'Task\[',
                r'Task[Any, ',
                content
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

def main() -> None:
    """Основная функция."""
    # Исправляем тестовые файлы
    test_files = glob.glob('tests/**/*.py', recursive=True)
    for test_file in test_files:
        print(f"Исправляю {test_file}")
        fix_test_file_annotations(test_file)
    
    # Исправляем основные файлы
    fix_main_files()
    
    print("Исправления завершены!")

if __name__ == "__main__":
    main() 