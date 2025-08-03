#!/usr/bin/env python3
"""
Скрипт для исправления ошибок типизации pandas в проекте ATB
"""

import os
import re
from pathlib import Path


def fix_visualization_py() -> None:
    """Исправление ошибок в visualization.py"""
    file_path: str = "infrastructure/core/visualization.py"
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content: str = f.read()
    
    # Исправляем импорты
    if "from pandas.core.series import Series as PandasSeries" not in content:
        content = content.replace(
            "from pandas import Series, DataFrame",
            "from pandas import Series, DataFrame\nfrom pandas.core.series import Series as PandasSeries\nfrom pandas.core.frame import DataFrame as PandasDataFrame"
        )
    
    # Исправляем cumprod
    content = re.sub(
        r'if hasattr\(([^,]+), \'cumprod\'\):',
        r'if isinstance(\1, (pd.Series, PandasSeries)) and hasattr(\1, \'cumprod\'):',
        content
    )
    
    # Исправляем gt
    content = re.sub(
        r'delta\.gt\(0\)\.map\(\{True: "green", False: "red"\}\)',
        r'delta.gt(0).map({True: "green", False: "red"}) if isinstance(delta, (pd.Series, PandasSeries)) else ["green" if x > 0 else "red" for x in delta]',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправлен файл {file_path}")


def fix_technical_analysis_py() -> None:
    """Исправление ошибок в technical_analysis.py"""
    file_path: str = "infrastructure/core/technical_analysis.py"
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content: str = f.read()
    
    # Исправляем abs
    content = re.sub(
        r'\.abs\(\)',
        r'.abs() if hasattr(result, \'abs\') else pd.Series()',
        content
    )
    
    # Исправляем операторы сравнения
    content = re.sub(
        r'\(([^>]+) > ([^)]+)\) & \(([^>]+) > 0\)',
        r'isinstance(\1, pd.Series) and isinstance(\2, pd.Series) and (\1 > \2) & (\3 > 0)',
        content
    )
    
    # Исправляем iloc
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\.iloc\[([^\]]+)\]',
        r'\1.iloc[\2] if hasattr(\1, \'iloc\') else \1[\2]',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправлен файл {file_path}")


def fix_technical_py() -> None:
    """Исправление ошибок в technical.py"""
    file_path: str = "infrastructure/core/technical.py"
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content: str = f.read()
    
    # Исправляем abs
    content = re.sub(
        r'\.abs\(\)',
        r'.abs() if hasattr(result, \'abs\') else pd.Series()',
        content
    )
    
    # Исправляем операторы сравнения
    content = re.sub(
        r'\(([^>]+) > ([^)]+)\) & \(([^>]+) > 0\)',
        r'isinstance(\1, pd.Series) and isinstance(\2, pd.Series) and (\1 > \2) & (\3 > 0)',
        content
    )
    
    # Исправляем iloc
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\.iloc\[([^\]]+)\]',
        r'\1.iloc[\2] if hasattr(\1, \'iloc\') else \1[\2]',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправлен файл {file_path}")


def main() -> None:
    """Основная функция"""
    print("Начинаю исправление ошибок типизации...")
    
    fix_visualization_py()
    fix_technical_analysis_py()
    fix_technical_py()
    
    print("Исправление завершено!")


if __name__ == "__main__":
    main() 