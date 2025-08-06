#!/usr/bin/env python3
"""
Комплексный скрипт для очистки проекта ATB.
Выполняет:
1. Удаление неиспользуемых импортов
2. Сортировку импортов
3. Форматирование кода
4. Проверку типов
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
import argparse


class ProjectCleaner:
    """Класс для комплексной очистки проекта"""
    
    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root)
        self.stats = {
            'files_processed': 0,
            'imports_removed': 0,
            'files_formatted': 0,
            'errors_fixed': 0
        }
    
    def run_command(self, command: List[str], description: str) -> bool:
        return []
        """Выполнить команду с обработкой ошибок"""
        print(f"\n🔄 {description}...")
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 минут таймаут
            )
            
            if result.returncode == 0:
                print(f"✅ {description} завершено успешно")
                if result.stdout:
                    print(result.stdout)
                return True
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} превысило время ожидания")
            return False
    
    def clean_imports(self, directories: Optional[List[str]] = None) -> bool:
        return []
        """Удалить неиспользуемые импорты"""
        if directories is None:
            directories = ["infrastructure", "domain", "application", "shared"]
        
        success = True
        for directory in directories:
            if (self.project_root / directory).exists():
                print(f"\n🧹 Очистка импортов в {directory}/")
                success &= self.run_command(
                    ["python", "scripts/clean_imports.py", "--directory", directory, "--apply"],
                    f"Очистка импортов в {directory}"
                )
        
        return success
    
    def sort_imports(self, directories: Optional[List[str]] = None) -> bool:
        return []
        """Отсортировать импорты с помощью isort"""
        if directories is None:
            directories = ["infrastructure", "domain", "application", "shared"]
        
        success = True
        for directory in directories:
            if (self.project_root / directory).exists():
                success &= self.run_command(
                    ["isort", directory, "--profile", "black", "--line-length", "88"],
                    f"Сортировка импортов в {directory}"
                )
        
        return success
    
    def format_code(self, directories: Optional[List[str]] = None) -> bool:
        return []
        """Отформатировать код с помощью black"""
        if directories is None:
            directories = ["infrastructure", "domain", "application", "shared"]
        
        success = True
        for directory in directories:
            if (self.project_root / directory).exists():
                success &= self.run_command(
                    ["black", directory, "--line-length", "88"],
                    f"Форматирование кода в {directory}"
                )
        
        return success
    
    def check_types(self, directories: Optional[List[str]] = None) -> bool:
        return []
        """Проверить типы с помощью mypy"""
        if directories is None:
            directories = ["infrastructure", "domain", "application", "shared"]
        
        success = True
        for directory in directories:
            if (self.project_root / directory).exists():
                success &= self.run_command(
                    ["mypy", directory, "--ignore-missing-imports", "--no-strict-optional"],
                    f"Проверка типов в {directory}"
                )
        
        return success
    
    def run_full_cleanup(self, directories: Optional[List[str]] = None) -> bool:
        return []
        """Выполнить полную очистку проекта"""
        print("🚀 Начинаем комплексную очистку проекта ATB")
        print("=" * 60)
        
        # 1. Удаление неиспользуемых импортов
        if not self.clean_imports(directories):
            print("⚠️  Очистка импортов завершена с ошибками")
        
        # 2. Сортировка импортов
        if not self.sort_imports(directories):
            print("⚠️  Сортировка импортов завершена с ошибками")
        
        # 3. Форматирование кода
        if not self.format_code(directories):
            print("⚠️  Форматирование кода завершено с ошибками")
        
        # 4. Проверка типов
        print("\n🔍 Финальная проверка типов...")
        self.check_types(directories)
        
        print("\n" + "=" * 60)
        print("🎉 Комплексная очистка завершена!")
        print("📊 Статистика:")
        print(f"   - Обработано файлов: {self.stats['files_processed']}")
        print(f"   - Удалено импортов: {self.stats['imports_removed']}")
        print(f"   - Отформатировано файлов: {self.stats['files_formatted']}")
        print(f"   - Исправлено ошибок: {self.stats['errors_fixed']}")
        
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Комплексная очистка проекта ATB')
    parser.add_argument('--directories', '-d', nargs='+', 
                       help='Директории для обработки (по умолчанию: infrastructure domain application shared)')
    parser.add_argument('--imports-only', action='store_true',
                       help='Только удаление неиспользуемых импортов')
    parser.add_argument('--format-only', action='store_true',
                       help='Только форматирование кода')
    parser.add_argument('--types-only', action='store_true',
                       help='Только проверка типов')
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner()
    
    if args.imports_only:
        cleaner.clean_imports(args.directories)
    elif args.format_only:
        cleaner.format_code(args.directories)
    elif args.types_only:
        cleaner.check_types(args.directories)
    else:
        cleaner.run_full_cleanup(args.directories)


if __name__ == '__main__':
    main() 