# Руководство по очистке проекта ATB

## Обзор

Этот документ описывает инструменты и процедуры для автоматической очистки проекта от неиспользуемых импортов и улучшения качества кода.

## Установленные инструменты

### 1. autoflake
- **Назначение**: Удаление неиспользуемых импортов и переменных
- **Установка**: `pip install autoflake`
- **Использование**: `autoflake --remove-all-unused-imports --in-place --recursive directory/`

### 2. unimport
- **Назначение**: Современная альтернатива autoflake
- **Установка**: `pip install unimport`
- **Использование**: `unimport --check file.py`

### 3. isort
- **Назначение**: Сортировка и организация импортов
- **Установка**: `pip install isort`
- **Использование**: `isort directory/ --profile black`

### 4. black
- **Назначение**: Форматирование кода
- **Установка**: `pip install black`
- **Использование**: `black directory/ --line-length 88`

## Автоматические скрипты

### 1. clean_imports.py
Основной скрипт для удаления неиспользуемых импортов.

```bash
# Очистка всей инфраструктуры
python scripts/clean_imports.py --directory infrastructure --apply

# Очистка конкретного файла
python scripts/clean_imports.py --file path/to/file.py --apply

# Только проверка (без изменений)
python scripts/clean_imports.py --directory infrastructure
```

### 2. clean_project.py
Комплексный скрипт для полной очистки проекта.

```bash
# Полная очистка
python scripts/clean_project.py

# Только импорты
python scripts/clean_project.py --imports-only

# Только форматирование
python scripts/clean_project.py --format-only

# Только проверка типов
python scripts/clean_project.py --types-only
```

### 3. clean_project.bat (Windows)
Интерактивный bat-файл для Windows.

```cmd
clean_project.bat
```

## Результаты очистки

После выполнения очистки инфраструктурного слоя:

- **Удалено импортов**: 850
- **Изменено файлов**: 316
- **Обработано файлов**: 432
- **Ошибок mypy**: уменьшено с тысяч до 336

## Рекомендации по использованию

### 1. Регулярная очистка
Рекомендуется выполнять очистку импортов:
- После каждого крупного рефакторинга
- Перед коммитом изменений
- При добавлении новых зависимостей

### 2. Проверка перед коммитом
```bash
# Быстрая проверка
python scripts/clean_project.py --types-only

# Полная очистка
python scripts/clean_project.py
```

### 3. Интеграция с CI/CD
Добавьте в pipeline:
```yaml
- name: Clean imports
  run: python scripts/clean_imports.py --directory . --apply

- name: Format code
  run: python scripts/clean_project.py --format-only

- name: Type check
  run: python scripts/clean_project.py --types-only
```

## Устранение проблем

### 1. Ошибки кодировки
Если возникают ошибки кодировки:
```python
# В clean_imports.py добавлена поддержка cp1251
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
except UnicodeDecodeError:
    with open(file_path, 'r', encoding='cp1251') as f:
        source = f.read()
```

### 2. Синтаксические ошибки
Если autoflake/unimport не работают:
- Используйте наш кастомный скрипт `clean_imports.py`
- Проверьте синтаксис файлов перед очисткой

### 3. Конфликты с mypy
После очистки импортов:
- Проверьте типизацию
- Добавьте недостающие type stubs
- Исправьте оставшиеся ошибки типизации

## Мониторинг качества

### Метрики до очистки:
- Ошибок mypy: 1000+
- Неиспользуемых импортов: 850+
- Файлов с проблемами: 400+

### Метрики после очистки:
- Ошибок mypy: 336
- Неиспользуемых импортов: 0
- Файлов с проблемами: 38

## Заключение

Автоматическая очистка импортов значительно улучшает качество кода и уменьшает количество ошибок mypy. Рекомендуется интегрировать эти инструменты в рабочий процесс разработки. 