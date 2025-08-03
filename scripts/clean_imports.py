#!/usr/bin/env python3
"""
Скрипт для удаления неиспользуемых импортов в проекте ATB.
Использует ast для анализа кода и безопасного удаления импортов.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Set, List, Tuple, Optional
from collections import defaultdict
import argparse


class ImportCleaner:
    """Класс для очистки неиспользуемых импортов"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.stats = {
            'files_processed': 0,
            'imports_removed': 0,
            'files_modified': 0,
            'errors': 0
        }
        
    def find_python_files(self, directory: Path) -> List[Path]:
        """Найти все Python файлы в директории"""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Пропускаем виртуальные окружения и кэш
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', '.git']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def parse_imports(self, source: str) -> Tuple[Set[str], Set[str], List[ast.Import], List[ast.ImportFrom]]:
        """Парсинг импортов из исходного кода"""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set(), set(), [], []
        
        imported_names = set()
        imported_modules = set()
        import_nodes = []
        import_from_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_nodes.append(node)
                for alias in node.names:
                    imported_names.add(alias.name)
                    imported_modules.add(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                import_from_nodes.append(node)
                if node.module:
                    imported_modules.add(node.module)
                for alias in node.names:
                    imported_names.add(alias.name)
        
        return imported_names, imported_modules, import_nodes, import_from_nodes
    
    def find_used_names(self, source: str) -> Set[str]:
        """Найти все используемые имена в коде"""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Добавляем только базовое имя атрибута
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        return used_names
    
    def clean_imports_in_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """Очистить неиспользуемые импорты в файле"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    source = f.read()
            except UnicodeDecodeError:
                print(f"Ошибка кодировки в файле: {file_path}")
                self.stats['errors'] += 1
                return False
        
        # Парсим импорты и используемые имена
        imported_names, imported_modules, import_nodes, import_from_nodes = self.parse_imports(source)
        used_names = self.find_used_names(source)
        
        # Находим неиспользуемые импорты
        unused_imports = imported_names - used_names
        
        if not unused_imports:
            return False
        
        # Удаляем неиспользуемые импорты
        lines = source.split('\n')
        modified = False
        
        # Удаляем строки с неиспользуемыми импортами
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Проверяем обычные импорты
            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                # Простая проверка на неиспользуемые импорты
                for unused in unused_imports:
                    if f'import {unused}' in line or f'from {unused}' in line:
                        if dry_run:
                            print(f"  Удалить: {line_stripped}")
                        else:
                            lines[i] = ''
                            modified = True
                        self.stats['imports_removed'] += 1
                        break
        
        if modified and not dry_run:
            # Удаляем пустые строки
            lines = [line for line in lines if line.strip()]
            
            # Записываем обратно
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.stats['files_modified'] += 1
            return True
        
        return False
    
    def clean_directory(self, directory: Optional[str] = None, dry_run: bool = True) -> None:
        """Очистить импорты во всей директории"""
        target_dir = Path(directory) if directory else self.root_dir
        
        print(f"Очистка импортов в: {target_dir}")
        print(f"Режим: {'Проверка' if dry_run else 'Изменение'}")
        print("-" * 50)
        
        python_files = self.find_python_files(target_dir)
        
        for file_path in python_files:
            print(f"Обработка: {file_path}")
            try:
                modified = self.clean_imports_in_file(file_path, dry_run)
                self.stats['files_processed'] += 1
                
                if modified and not dry_run:
                    print(f"  ✓ Изменен")
                elif not dry_run:
                    print(f"  - Без изменений")
                    
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                self.stats['errors'] += 1
        
        self.print_stats()
    
    def print_stats(self) -> None:
        """Вывести статистику"""
        print("\n" + "=" * 50)
        print("СТАТИСТИКА:")
        print(f"Обработано файлов: {self.stats['files_processed']}")
        print(f"Удалено импортов: {self.stats['imports_removed']}")
        print(f"Изменено файлов: {self.stats['files_modified']}")
        print(f"Ошибок: {self.stats['errors']}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Очистка неиспользуемых импортов в Python файлах')
    parser.add_argument('--directory', '-d', default='.', help='Директория для обработки')
    parser.add_argument('--apply', '-a', action='store_true', help='Применить изменения (по умолчанию только проверка)')
    parser.add_argument('--file', '-f', help='Обработать только один файл')
    
    args = parser.parse_args()
    
    cleaner = ImportCleaner(args.directory)
    
    if args.file:
        # Обработать один файл
        file_path = Path(args.file)
        if file_path.exists():
            print(f"Обработка файла: {file_path}")
            cleaner.clean_imports_in_file(file_path, dry_run=not args.apply)
            cleaner.print_stats()
        else:
            print(f"Файл не найден: {file_path}")
            sys.exit(1)
    else:
        # Обработать директорию
        cleaner.clean_directory(args.directory, dry_run=not args.apply)


if __name__ == '__main__':
    main() 