#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска всех анализаторов слоев.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_analyzer(script_name: str, layer_name: str) -> bool:
    """Запустить анализатор для конкретного слоя."""
    print(f"\n{'='*60}")
    print(f"[Анализ] Анализ {layer_name} слоя...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"[Ошибка] Ошибка при анализе {layer_name} слоя:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[Ошибка] Ошибка запуска {script_name}: {e}")
        return False


def generate_summary_report():
    """Сгенерировать сводный отчет."""
    print(f"\n{'='*60}")
    print("[Отчет] ГЕНЕРАЦИЯ СВОДНОГО ОТЧЕТА")
    print(f"{'='*60}")
    
    summary = []
    summary.append("# Сводный отчет: Нереализованные функции по всем слоям")
    summary.append(f"## Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Список отчетов
    reports = [
        ("application_issues_report.md", "Application"),
        ("domain_issues_report.md", "Domain"),
        ("infrastructure_issues_report.md", "Infrastructure"),
        ("shared_issues_report.md", "Shared"),
    ]
    
    total_issues = 0
    
    for report_file, layer_name in reports:
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Извлекаем количество проблем
            if "Всего найдено проблем:" in content:
                for line in content.split('\n'):
                    if "Всего найдено проблем:" in line:
                        count = int(line.split(':')[1].strip())
                        total_issues += count
                        summary.append(f"## {layer_name} слой")
                        summary.append(f"- Найдено проблем: **{count}**")
                        summary.append(f"- [Подробный отчет]({report_file})")
                        summary.append("")
                        break
            else:
                summary.append(f"## {layer_name} слой")
                summary.append("- Найдено проблем: **0** [OK]")
                summary.append(f"- [Подробный отчет]({report_file})")
                summary.append("")
                
        except FileNotFoundError:
            summary.append(f"## {layer_name} слой")
            summary.append("- Отчет не найден [Ошибка]")
            summary.append("")
    
    summary.append("## [Статистика] Общая статистика")
    summary.append(f"- **Всего проблем по всем слоям:** {total_issues}")
    
    if total_issues == 0:
        summary.append("- [Успех] **Все слои проанализированы успешно!**")
    elif total_issues < 10:
        summary.append("- [Средне] **Небольшое количество проблем**")
    elif total_issues < 50:
        summary.append("- [Средне] **Среднее количество проблем**")
    else:
        summary.append("- [Критично] **Критическое количество проблем**")
    
    summary.append("")
    summary.append("## [Цель] Рекомендации")
    summary.append("1. **Высокий приоритет:** Исправить критические заглушки")
    summary.append("2. **Средний приоритет:** Реализовать TODO/FIXME")
    summary.append("3. **Низкий приоритет:** Улучшить простые возвраты")
    summary.append("4. **Тестирование:** Написать тесты для реализованных функций")
    summary.append("5. **Документация:** Обновить документацию")
    
    # Сохраняем сводный отчет
    with open("summary_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print("[OK] Сводный отчет сохранен в файл: summary_report.md")
    print(f"[Отчет] Общее количество проблем: {total_issues}")


def main():
    """Основная функция."""
    print("[Запуск] ЗАПУСК АНАЛИЗА ВСЕХ СЛОЕВ")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Список анализаторов
    analyzers = [
        ("application_analyzer.py", "Application"),
        ("domain_analyzer.py", "Domain"),
        ("infrastructure_analyzer.py", "Infrastructure"),
        ("shared_analyzer.py", "Shared"),
    ]
    
    success_count = 0
    
    for script_name, layer_name in analyzers:
        if run_analyzer(script_name, layer_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("[Отчет] РЕЗУЛЬТАТЫ АНАЛИЗА")
    print(f"{'='*60}")
    print(f"[OK] Успешно проанализировано слоев: {success_count}/{len(analyzers)}")
    
    if success_count == len(analyzers):
        print("[Успех] Все слои проанализированы успешно!")
    else:
        print("[Внимание] Некоторые слои не удалось проанализировать")
    
    # Генерируем сводный отчет
    generate_summary_report()
    
    print(f"\n{'='*60}")
    print("[Файл] СОЗДАННЫЕ ФАЙЛЫ:")
    print(f"{'='*60}")
    
    report_files = [
        "application_issues_report.md",
        "domain_issues_report.md", 
        "infrastructure_issues_report.md",
        "shared_issues_report.md",
        "summary_report.md"
    ]
    
    for report_file in report_files:
        if Path(report_file).exists():
            print(f"[OK] {report_file}")
        else:
            print(f"[Ошибка] {report_file} (не создан)")
    
    print(f"\n[Цель] Анализ завершен! Проверьте созданные отчеты.")


if __name__ == "__main__":
    main() 