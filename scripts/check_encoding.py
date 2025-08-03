#!/usr/bin/env python3
"""
Простой скрипт для проверки и исправления проблем с кодировкой.
"""

import os
import sys
from pathlib import Path

def fix_file(file_path):
    """Исправляет проблемы с кодировкой в файле."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Удаляем проблемные байты
        content = content.replace(b'\x00', b'')
        content = content.replace(b'\xff', b'')
        
        # Удаляем BOM если есть
        if content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]
        
        # Перекодируем в UTF-8
        try:
            decoded = content.decode('utf-8', errors='ignore')
        except:
            decoded = content.decode('latin-1', errors='ignore')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded)
        
        print(f"✓ Исправлен: {file_path}")
        return True
    except Exception as e:
        print(f"✗ Ошибка в {file_path}: {e}")
        return False

def check_and_fix():
    """Проверяет и исправляет все Python файлы."""
    fixed_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Пропускаем системные папки
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Проверяем на проблемы
                    if b'\x00' in content or b'\xff' in content:
                        print(f"Найдена проблема в: {file_path}")
                        if fix_file(file_path):
                            fixed_count += 1
                        else:
                            error_count += 1
                            
                except Exception as e:
                    print(f"Ошибка проверки {file_path}: {e}")
                    error_count += 1
    
    print(f"\nРезультат: исправлено {fixed_count} файлов, ошибок {error_count}")

if __name__ == '__main__':
    check_and_fix() 