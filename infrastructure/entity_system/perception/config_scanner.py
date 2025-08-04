"""Сканер конфигурационных файлов."""

import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger
from shared.numpy_utils import np

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False
    toml = None


@dataclass
class ConfigFile:
    path: Path
    format: str
    content: Any
    size: int
    last_modified: float
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigScanner:
    """Промышленный сканер конфигураций проекта с поддержкой множественных форматов."""

    def __init__(self) -> None:
        self.supported_formats = {
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
        }
        self.config_files: List[ConfigFile] = []
        self.scan_results: Dict[str, Any] = {}

    async def scan_configs(self) -> Dict[str, Any]:
        """Промышленное сканирование конфигураций с анализом и валидацией."""
        logger.info("Начало промышленного сканирования конфигов")

        try:
            # Поиск конфигурационных файлов
            config_files = await self._find_config_files()

            # Анализ каждого файла
            for file_path in config_files:
                config_file = await self._analyze_config_file(file_path)
                self.config_files.append(config_file)

            # Анализ структуры конфигураций
            structure_analysis = await self._analyze_config_structure()

            # Детекция проблем и конфликтов
            problems = await self._detect_config_problems()

            # Генерация отчёта
            self.scan_results = {
                "total_files": len(self.config_files),
                "valid_files": len([f for f in self.config_files if f.is_valid]),
                "invalid_files": len([f for f in self.config_files if not f.is_valid]),
                "formats": self._get_format_statistics(),
                "structure": structure_analysis,
                "problems": problems,
                "files": [self._serialize_config_file(f) for f in self.config_files],
                "recommendations": await self._generate_recommendations(),
            }

            logger.info(
                f"Сканирование завершено: {len(self.config_files)} файлов, "
                f"{len([f for f in self.config_files if f.is_valid])} валидных"
            )

            return self.scan_results

        except Exception as e:
            logger.error(f"Ошибка сканирования конфигураций: {e}")
            return {"error": str(e)}

    async def _find_config_files(self) -> List[Path]:
        """Поиск конфигурационных файлов в проекте."""
        config_files: List[Path] = []
        project_root = Path(".")

        # Поиск по расширениям
        for format_ext in self.supported_formats:
            config_files.extend(project_root.rglob(f"*{format_ext}"))

        # Поиск по именам файлов
        config_names = {"config", "settings", "conf", "ini", "cfg"}
        for name in config_names:
            config_files.extend(project_root.rglob(f"{name}.*"))
            config_files.extend(project_root.rglob(f".{name}.*"))

        # Удаление дубликатов и исключение системных файлов
        unique_files = list(set(config_files))
        filtered_files = [
            f
            for f in unique_files
            if not any(part.startswith(".") for part in f.parts)
            and not any(
                part in {"venv", "__pycache__", "node_modules"} for part in f.parts
            )
        ]

        return filtered_files

    async def _analyze_config_file(self, file_path: Path) -> ConfigFile:
        """Промышленный анализ конфигурационного файла."""
        try:
            content = file_path.read_text(encoding="utf-8")
            format_type = self._detect_format(file_path)

            # Парсинг контента
            parsed_content, is_valid, errors, warnings = (
                await self._parse_config_content(content, format_type)
            )

            return ConfigFile(
                path=file_path,
                format=format_type,
                content=parsed_content,
                size=len(content),
                last_modified=file_path.stat().st_mtime,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Ошибка анализа файла {file_path}: {e}")
            return ConfigFile(
                path=file_path,
                format="unknown",
                content=None,
                size=0,
                last_modified=0,
                is_valid=False,
                errors=[str(e)],
                warnings=[],
            )

    def _detect_format(self, file_path: Path) -> str:
        """Определение формата конфигурационного файла."""
        suffix = file_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        elif suffix == ".json":
            return "json"
        elif suffix == ".toml":
            return "toml"
        elif suffix in {".ini", ".cfg", ".conf"}:
            return "ini"
        else:
            return "unknown"

    async def _parse_config_content(self, content: str, format_type: str) -> tuple:
        """Промышленный парсинг конфигурационного контента."""
        parsed_content = None
        is_valid = True
        errors: List[str] = []
        warnings: List[str] = []

        try:
            if format_type == "yaml":
                parsed_content = yaml.safe_load(content)
                if parsed_content is None:
                    warnings.append("Пустой YAML файл")

            elif format_type == "json":
                parsed_content = json.loads(content)

            elif format_type == "toml":
                if TOML_AVAILABLE:
                    parsed_content = toml.loads(content)
                else:
                    errors.append("Библиотека toml не установлена")
                    is_valid = False

            elif format_type == "ini":
                config = configparser.ConfigParser()
                config.read_string(content)
                parsed_content = {
                    section: dict(config[section]) for section in config.sections()
                }

            else:
                errors.append(f"Неподдерживаемый формат: {format_type}")
                is_valid = False

        except yaml.YAMLError as e:
            errors.append(f"Ошибка парсинга YAML: {e}")
            is_valid = False
        except json.JSONDecodeError as e:
            errors.append(f"Ошибка парсинга JSON: {e}")
            is_valid = False
        except toml.TomlDecodeError as e:
            if TOML_AVAILABLE:
                errors.append(f"Ошибка парсинга TOML: {e}")
            else:
                errors.append("Библиотека toml не установлена")
            is_valid = False
        except configparser.Error as e:
            errors.append(f"Ошибка парсинга INI: {e}")
            is_valid = False
        except Exception as e:
            errors.append(f"Неожиданная ошибка парсинга: {e}")
            is_valid = False

        # Дополнительные проверки
        if parsed_content is not None:
            validation_errors, validation_warnings = self._validate_config_content(
                parsed_content, format_type
            )
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
            if validation_errors:
                is_valid = False

        return parsed_content, is_valid, errors, warnings

    def _validate_config_content(self, content: Any, format_type: str) -> tuple:
        """Валидация конфигурационного контента."""
        errors: List[str] = []
        warnings: List[str] = []

        if isinstance(content, dict):
            # Проверка на пустые секции
            for key, value in content.items():
                if value is None or value == "":
                    warnings.append(f"Пустое значение для ключа: {key}")

            # Проверка на дублирующиеся ключи (для JSON)
            if format_type == "json":
                # JSON не поддерживает дублирующиеся ключи, но можно проверить структуру
                pass

        elif isinstance(content, list):
            if len(content) == 0:
                warnings.append("Пустой список конфигураций")

        return errors, warnings

    async def _analyze_config_structure(self) -> Dict[str, Any]:
        """Анализ структуры конфигураций."""
        structure: Dict[str, Any] = {
            "common_keys": {},
            "nested_levels": {},
            "value_types": {},
            "file_relationships": {},
        }

        # Анализ общих ключей
        all_keys = set()
        for config_file in self.config_files:
            if config_file.content and isinstance(config_file.content, dict):
                all_keys.update(config_file.content.keys())

        structure["common_keys"] = {
            "total_unique": len(all_keys),
            "keys": list(all_keys),
        }

        # Анализ уровней вложенности
        max_nesting = 0
        for config_file in self.config_files:
            if config_file.content:
                nesting = self._calculate_nesting_level(config_file.content)
                max_nesting = max(max_nesting, nesting)

        structure["nested_levels"] = {
            "max_depth": max_nesting,
            "average_depth": max_nesting,  # Упрощённо
        }

        return structure

    def _calculate_nesting_level(self, obj: Any, current_level: int = 0) -> int:
        """Расчёт уровня вложенности конфигурации."""
        if isinstance(obj, dict):
            if not obj:
                return current_level
            return max(
                self._calculate_nesting_level(value, current_level + 1)
                for value in obj.values()
            )
        elif isinstance(obj, list):
            if not obj:
                return current_level
            return max(
                self._calculate_nesting_level(item, current_level + 1) for item in obj
            )
        else:
            return current_level

    async def _detect_config_problems(self) -> List[Dict[str, Any]]:
        """Детекция проблем в конфигурациях."""
        problems: List[Dict[str, Any]] = []

        # Проверка на невалидные файлы
        invalid_files: List[Any] = [f for f in self.config_files if not f.is_valid]
        if invalid_files:
            problems.append(
                {
                    "type": "invalid_files",
                    "severity": "high",
                    "description": f"Найдено {len(invalid_files)} невалидных конфигурационных файлов",
                    "files": [str(f.path) for f in invalid_files],
                }
            )

        # Проверка на дублирующиеся конфигурации
        duplicate_configs: List[Dict[str, Any]] = self._find_duplicate_configs()
        if duplicate_configs:
            problems.append(
                {
                    "type": "duplicate_configs",
                    "severity": "medium",
                    "description": f"Найдено {len(duplicate_configs)} дублирующихся конфигураций",
                    "details": duplicate_configs,
                }
            )

        # Проверка на устаревшие форматы
        outdated_formats: List[Any] = [f for f in self.config_files if f.format == "ini"]
        if outdated_formats:
            problems.append(
                {
                    "type": "outdated_formats",
                    "severity": "low",
                    "description": f"Найдено {len(outdated_formats)} файлов в устаревшем формате INI",
                    "files": [str(f.path) for f in outdated_formats],
                }
            )

        return problems

    def _find_duplicate_configs(self) -> List[Dict[str, Any]]:
        """Поиск дублирующихся конфигураций."""
        duplicates: List[Dict[str, Any]] = []
        config_contents: Dict[int, Any] = {}

        for config_file in self.config_files:
            if config_file.content:
                content_hash = hash(str(config_file.content))
                if content_hash in config_contents:
                    duplicates.append(
                        {
                            "original": str(config_contents[content_hash]),
                            "duplicate": str(config_file.path),
                        }
                    )
                else:
                    config_contents[content_hash] = config_file.path

        return duplicates

    def _get_format_statistics(self) -> Dict[str, int]:
        """Статистика по форматам файлов."""
        stats: Dict[str, int] = {}
        for config_file in self.config_files:
            format_type = config_file.format
            stats[format_type] = stats.get(format_type, 0) + 1
        return stats

    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Генерация рекомендаций по улучшению конфигураций."""
        recommendations: List[Dict[str, Any]] = []

        # Рекомендации по форматам
        if any(f.format == "ini" for f in self.config_files):
            recommendations.append(
                {
                    "type": "format_migration",
                    "priority": "medium",
                    "description": "Рассмотрите миграцию с INI на YAML или TOML для лучшей читаемости",
                    "action": "migrate_to_yaml",
                }
            )

        # Рекомендации по структуре
        if any(
            f.content and self._calculate_nesting_level(f.content) > 5
            for f in self.config_files
        ):
            recommendations.append(
                {
                    "type": "structure_simplification",
                    "priority": "low",
                    "description": "Упростите структуру конфигураций для лучшей читаемости",
                    "action": "flatten_structure",
                }
            )

        return recommendations

    def _serialize_config_file(self, config_file: ConfigFile) -> Dict[str, Any]:
        """Сериализация конфигурационного файла для отчёта."""
        return {
            "path": str(config_file.path),
            "format": config_file.format,
            "size": config_file.size,
            "last_modified": config_file.last_modified,
            "is_valid": config_file.is_valid,
            "errors": config_file.errors,
            "warnings": config_file.warnings,
        }
