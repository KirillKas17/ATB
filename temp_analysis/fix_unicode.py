#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для исправления Unicode проблем в анализаторах.
"""

import re
from pathlib import Path

def fix_unicode_in_file(file_path: str) -> None:
    """Исправить Unicode символы в файле."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем эмодзи на обычный текст
    replacements = {
        r'🔍': '[Анализ]',
        r'📊': '[Отчет]',
        r'📈': '[Статистика]',
        r'🚨': '[Проблемы]',
        r'📋': '[Список]',
        r'✅': '[OK]',
        r'❌': '[Ошибка]',
        r'⚠️': '[Внимание]',
        r'🎯': '[Цель]',
        r'🎉': '[Успех]',
        r'🔴': '[Критично]',
        r'🟡': '[Средне]',
        r'🟠': '[Средне]',
        r'📁': '[Файл]',
        r'📝': '[Документ]',
        r'🔧': '[Настройка]',
        r'🧹': '[Очистка]',
        r'📞': '[Поддержка]',
        r'🚀': '[Запуск]',
    }
    
    for emoji, replacement in replacements.items():
        content = re.sub(emoji, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправлен файл: {file_path}")

def main() -> None:
    """Основная функция."""
    files_to_fix = [
        'application_analyzer.py',
        'domain_analyzer.py',
        'infrastructure_analyzer.py',
        'shared_analyzer.py',
        'run_all_analyzers.py'
    ]
    
    for file_name in files_to_fix:
        if Path(file_name).exists():
            fix_unicode_in_file(file_name)
        else:
            print(f"Файл не найден: {file_name}")

if __name__ == "__main__":
    main() 