#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска тестов с анализом покрытия.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json


class TestCoverageRunner:
    """Класс для запуска тестов с анализом покрытия."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_dir = project_root / "coverage_reports"
        self.coverage_dir.mkdir(exist_ok=True)

    def run_tests(self, test_type: str = "all", parallel: bool = False, verbose: bool = False) -> bool:
        """Запуск тестов."""
        print(f"🚀 Запуск тестов типа: {test_type}")
        
        cmd = ["python", "-m", "pytest"]
        
        if test_type != "all":
            cmd.extend(["-m", test_type])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--cov=.",
            "--cov-report=html:coverage_reports/html",
            "--cov-report=json:coverage_reports/coverage.json",
            "--cov-report=term-missing",
            "--cov-fail-under=90",
            "tests/"
        ])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Все тесты прошли успешно!")
                return True
            else:
                print("❌ Тесты завершились с ошибками:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Ошибка запуска тестов: {e}")
            return False

    def analyze_coverage(self) -> Dict[str, Any]:
        """Анализ покрытия кода."""
        print("📊 Анализ покрытия кода...")
        
        coverage_file = self.coverage_dir / "coverage.json"
        if not coverage_file.exists():
            print("❌ Файл покрытия не найден")
            return {}
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            return self._process_coverage_data(coverage_data)
            
        except Exception as e:
            print(f"❌ Ошибка анализа покрытия: {e}")
            return {}

    def _process_coverage_data(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка данных покрытия."""
        total_lines = coverage_data.get("totals", {}).get("num_statements", 0)
        covered_lines = coverage_data.get("totals", {}).get("covered_lines", 0)
        missing_lines = coverage_data.get("totals", {}).get("missing_lines", 0)
        
        coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Анализ по модулям
        modules_analysis = {}
        for file_path, file_data in coverage_data.get("files", {}).items():
            if file_path.startswith("tests/") or file_path.startswith("venv/"):
                continue
                
            file_lines = file_data.get("summary", {}).get("num_statements", 0)
            file_covered = file_data.get("summary", {}).get("covered_lines", 0)
            file_missing = file_data.get("summary", {}).get("missing_lines", 0)
            
            if file_lines > 0:
                file_coverage = (file_covered / file_lines * 100)
                modules_analysis[file_path] = {
                    "coverage": round(file_coverage, 2),
                    "lines": file_lines,
                    "covered": file_covered,
                    "missing": file_missing
                }
        
        return {
            "total_coverage": round(coverage_percentage, 2),
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "missing_lines": missing_lines,
            "modules": modules_analysis
        }

    def generate_coverage_report(self, analysis: Dict[str, Any]) -> None:
        """Генерация отчета о покрытии."""
        print("📋 Генерация отчета о покрытии...")
        
        report_file = self.coverage_dir / "coverage_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📊 Отчет о покрытии тестами\n\n")
            f.write(f"**Общее покрытие**: {analysis.get('total_coverage', 0)}%\n\n")
            f.write(f"**Всего строк кода**: {analysis.get('total_lines', 0)}\n")
            f.write(f"**Покрыто строк**: {analysis.get('covered_lines', 0)}\n")
            f.write(f"**Непокрыто строк**: {analysis.get('missing_lines', 0)}\n\n")
            
            f.write("## 📁 Покрытие по модулям\n\n")
            f.write("| Модуль | Покрытие | Строк | Покрыто | Непокрыто |\n")
            f.write("|--------|----------|-------|---------|-----------|\n")
            
            modules = analysis.get("modules", {})
            for module_path, module_data in sorted(modules.items()):
                coverage = module_data.get("coverage", 0)
                lines = module_data.get("lines", 0)
                covered = module_data.get("covered", 0)
                missing = module_data.get("missing", 0)
                
                status = "🟢" if coverage >= 90 else "🟡" if coverage >= 70 else "🔴"
                
                f.write(f"| {module_path} | {status} {coverage}% | {lines} | {covered} | {missing} |\n")
        
        print(f"✅ Отчет сохранен в: {report_file}")

    def find_uncovered_files(self) -> List[str]:
        """Поиск непокрытых файлов."""
        print("🔍 Поиск непокрытых файлов...")
        
        uncovered_files = []
        
        # Поиск Python файлов в проекте
        for root, dirs, files in os.walk(self.project_root):
            # Пропускаем служебные директории
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', 'tests', 'coverage_reports']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.project_root)
                    
                    # Проверяем, есть ли тест для этого файла
                    test_file = self.project_root / "tests" / f"test_{file}"
                    if not test_file.exists():
                        uncovered_files.append(str(relative_path))
        
        return uncovered_files

    def generate_test_templates(self, uncovered_files: List[str]) -> None:
        """Генерация шаблонов тестов для непокрытых файлов."""
        print("📝 Генерация шаблонов тестов...")
        
        templates_dir = self.project_root / "test_templates"
        templates_dir.mkdir(exist_ok=True)
        
        for file_path in uncovered_files[:10]:  # Ограничиваем первыми 10 файлами
            template_content = self._generate_test_template(file_path)
            
            template_file = templates_dir / f"test_{Path(file_path).stem}.py"
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
        
        print(f"✅ Шаблоны тестов сохранены в: {templates_dir}")

    def _generate_test_template(self, file_path: str) -> str:
        """Генерация шаблона теста для файла."""
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        template = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit тесты для {module_name}.
"""

import pytest
from unittest.mock import Mock, AsyncMock

# Импорты для тестирования
# from {module_name} import ClassName


class Test{Path(file_path).stem.title()}:
    """Тесты для {Path(file_path).stem}."""

    def test_initialization(self):
        """Тест инициализации."""
        # Arrange & Act
        # obj = ClassName()
        
        # Assert
        # assert obj is not None
        pass

    def test_method_name(self):
        """Тест метода method_name."""
        # Arrange
        # obj = ClassName()
        
        # Act
        # result = obj.method_name()
        
        # Assert
        # assert result == expected_value
        pass

    @pytest.mark.asyncio
    async def test_async_method_name(self):
        """Тест асинхронного метода method_name."""
        # Arrange
        # obj = ClassName()
        
        # Act
        # result = await obj.async_method_name()
        
        # Assert
        # assert result == expected_value
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        return template

    def run_coverage_analysis(self) -> None:
        """Полный анализ покрытия."""
        print("🔍 Запуск полного анализа покрытия...")
        
        # Поиск непокрытых файлов
        uncovered_files = self.find_uncovered_files()
        
        print(f"📁 Найдено {len(uncovered_files)} непокрытых файлов:")
        for file_path in uncovered_files[:5]:  # Показываем первые 5
            print(f"  - {file_path}")
        
        if len(uncovered_files) > 5:
            print(f"  ... и еще {len(uncovered_files) - 5} файлов")
        
        # Генерация шаблонов тестов
        self.generate_test_templates(uncovered_files)
        
        # Анализ покрытия
        analysis = self.analyze_coverage()
        if analysis:
            self.generate_coverage_report(analysis)
            
            # Проверка целевого покрытия
            total_coverage = analysis.get('total_coverage', 0)
            if total_coverage >= 90:
                print(f"🎉 Целевое покрытие достигнуто: {total_coverage}%")
            else:
                print(f"⚠️ Целевое покрытие не достигнуто: {total_coverage}% (цель: 90%)")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Запуск тестов с анализом покрытия")
    parser.add_argument("--test-type", choices=["all", "unit", "integration", "e2e"], 
                       default="all", help="Тип тестов для запуска")
    parser.add_argument("--parallel", action="store_true", help="Запуск тестов параллельно")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument("--analyze-only", action="store_true", help="Только анализ покрытия")
    
    args = parser.parse_args()
    
    # Определение корневой директории проекта
    project_root = Path(__file__).parent.parent
    
    runner = TestCoverageRunner(project_root)
    
    if args.analyze_only:
        runner.run_coverage_analysis()
    else:
        # Запуск тестов
        success = runner.run_tests(args.test_type, args.parallel, args.verbose)
        
        if success:
            # Анализ покрытия
            runner.run_coverage_analysis()
        else:
            print("❌ Анализ покрытия пропущен из-за ошибок в тестах")


if __name__ == "__main__":
    main() 