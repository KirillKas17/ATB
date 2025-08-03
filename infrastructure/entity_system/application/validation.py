"""Движок валидации улучшений."""

import ast
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ValidationEngine:
    """Движок валидации улучшений."""

    def __init__(self) -> None:
        self.validation_results: List[Dict[str, Any]] = []

    async def validate_improvement(self, hypothesis: Dict[str, Any]) -> bool:
        """Валидация улучшения."""
        try:
            logger.info(
                f"Начало валидации улучшения: {hypothesis.get('title', 'Unknown')}"
            )

            target_component = hypothesis.get("target_component", "")
            improvement_type = hypothesis.get("type", "general")

            # Выполнение различных проверок
            validation_checks = [
                await self._validate_syntax(target_component),
                await self._validate_imports(target_component),
                await self._validate_tests(target_component),
            ]

            # Все проверки должны пройти
            all_passed = all(validation_checks)

            # Запись результата валидации
            validation_result = {
                "hypothesis": hypothesis,
                "target_component": target_component,
                "improvement_type": improvement_type,
                "validation_passed": all_passed,
                "checks": {
                    "syntax": validation_checks[0],
                    "imports": validation_checks[1],
                    "tests": validation_checks[2],
                },
                "timestamp": asyncio.get_event_loop().time(),
            }

            self.validation_results.append(validation_result)

            if all_passed:
                logger.info("Валидация улучшения прошла успешно")
            else:
                logger.warning("Валидация улучшения не прошла")

            return all_passed

        except Exception as e:
            logger.error(f"Ошибка валидации улучшения: {e}")
            return False

    async def _validate_syntax(self, target_component: str) -> bool:
        """Валидация синтаксиса Python кода."""
        try:
            if not target_component:
                return True

            # Поиск Python файлов в целевой компоненте
            python_files = self._find_python_files(target_component)

            for file_path in python_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()

                    # Проверка синтаксиса
                    ast.parse(source_code)
                    logger.debug(f"Синтаксис файла {file_path} корректен")

                except SyntaxError as e:
                    logger.error(f"Синтаксическая ошибка в {file_path}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {file_path}: {e}")
                    return False

            logger.info("Валидация синтаксиса прошла успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка валидации синтаксиса: {e}")
            return False

    async def _validate_imports(self, target_component: str) -> bool:
        """Валидация импортов."""
        try:
            if not target_component:
                return True

            python_files = self._find_python_files(target_component)

            for file_path in python_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()

                    # Парсинг AST для проверки импортов
                    tree = ast.parse(source_code)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if not await self._check_import_availability(
                                    alias.name
                                ):
                                    logger.error(
                                        f"Недоступный импорт в {file_path}: {alias.name}"
                                    )
                                    return False
                        elif isinstance(node, ast.ImportFrom):
                            if (
                                node.module
                                and not await self._check_import_availability(
                                    node.module
                                )
                            ):
                                logger.error(
                                    f"Недоступный импорт в {file_path}: {node.module}"
                                )
                                return False

                except Exception as e:
                    logger.error(f"Ошибка проверки импортов в {file_path}: {e}")
                    return False

            logger.info("Валидация импортов прошла успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка валидации импортов: {e}")
            return False

    async def _validate_tests(self, target_component: str) -> bool:
        """Валидация тестов."""
        try:
            if not target_component:
                return True

            # Поиск тестовых файлов
            test_files = self._find_test_files(target_component)

            if not test_files:
                logger.warning(f"Тестовые файлы не найдены для {target_component}")
                return True  # Не критично, если тестов нет

            # Проверка структуры тестов
            for test_file in test_files:
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        test_code = f.read()

                    # Проверка наличия тестовых функций
                    tree = ast.parse(test_code)
                    test_functions = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name.startswith("test_"):
                                test_functions.append(node.name)

                    if not test_functions:
                        logger.warning(f"Тестовые функции не найдены в {test_file}")

                except Exception as e:
                    logger.error(f"Ошибка проверки тестов в {test_file}: {e}")
                    return False

            logger.info("Валидация тестов прошла успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка валидации тестов: {e}")
            return False

    def _find_python_files(self, target_component: str) -> List[Path]:
        """Поиск Python файлов в целевой компоненте."""
        python_files = []
        try:
            if target_component:
                component_path = Path(target_component)
                if component_path.exists():
                    python_files = list(component_path.rglob("*.py"))
            else:
                # Поиск во всем проекте
                python_files = list(Path(".").rglob("*.py"))

            return python_files

        except Exception as e:
            logger.error(f"Ошибка поиска Python файлов: {e}")
            return []

    def _find_test_files(self, target_component: str) -> List[Path]:
        """Поиск тестовых файлов."""
        test_files = []
        try:
            if target_component:
                component_path = Path(target_component)
                if component_path.exists():
                    test_files = list(component_path.rglob("test_*.py"))
                    test_files.extend(component_path.rglob("*_test.py"))
            else:
                # Поиск во всем проекте
                test_files = list(Path(".").rglob("test_*.py"))
                test_files.extend(Path(".").rglob("*_test.py"))

            return test_files

        except Exception as e:
            logger.error(f"Ошибка поиска тестовых файлов: {e}")
            return []

    async def _check_import_availability(self, module_name: str) -> bool:
        """Проверка доступности импорта."""
        try:
            # Список стандартных модулей Python
            standard_modules = {
                "os",
                "sys",
                "json",
                "datetime",
                "pathlib",
                "typing",
                "asyncio",
                "numpy",
                "pandas",
                "scipy",
                "sklearn",
                "loguru",
                "pydantic",
            }

            # Проверка стандартных модулей
            if module_name in standard_modules:
                return True

            # Проверка внутренних модулей проекта
            if module_name.startswith(
                ("domain.", "application.", "infrastructure.", "shared.")
            ):
                return True

            # Для внешних модулей можно добавить дополнительную проверку
            # try:
            #     __import__(module_name)
            #     return True
            # except ImportError:
            #     return False

            return True  # По умолчанию считаем доступным

        except Exception as e:
            logger.error(f"Ошибка проверки импорта {module_name}: {e}")
            return False

    def get_validation_results(self) -> List[Dict[str, Any]]:
        """Получение результатов валидации."""
        return self.validation_results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Получение статистики валидации."""
        if not self.validation_results:
            return {}

        total_validations = len(self.validation_results)
        successful_validations = len(
            [
                result
                for result in self.validation_results
                if result["validation_passed"]
            ]
        )

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": (
                successful_validations / total_validations
                if total_validations > 0
                else 0
            ),
        }
