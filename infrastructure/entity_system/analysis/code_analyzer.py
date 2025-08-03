"""Анализатор кода."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class CodeIssue:
    type: str
    severity: str
    file: str
    line: Optional[int]
    description: str
    suggestion: str
    priority: float
    details: Optional[Dict[str, Any]] = None
    detected_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.detected_at is None:
            self.detected_at = datetime.now()


class CodeAnalyzer:
    """Промышленный анализатор кода с глубоким анализом."""

    def __init__(self) -> None:
        self.issue_patterns: Dict[str, List[str]] = {
            "performance": [
                r"for.*in.*range\(len\(",
                r"\.append\(.*\)",
                r"list\(.*\)",
                r"dict\(.*\)",
                r"set\(.*\)",
                r"\.join\(\[.*\]\)",
                r"sum\(\[.*\]\)",
                r"any\(\[.*\]\)",
                r"all\(\[.*\]\)",
                r"filter\(.*\)",
                r"map\(.*\)",
                r"reduce\(.*\)",
            ],
            "security": [
                r"eval\(",
                r"exec\(",
                r"__import__\(",
                r"open\(",
                r"subprocess\.",
                r"os\.system\(",
                r"pickle\.loads\(",
                r"yaml\.load\(",
                r"json\.loads\(",
                r"input\(",
                r"raw_input\(",
            ],
            "maintainability": [
                r"magic_number",
                r"hardcoded",
                r"TODO",
                r"FIXME",
                r"HACK",
                r"XXX",
                r"BUG",
                r"NOTE",
                r"WARNING",
                r"DEPRECATED",
            ],
            "complexity": [
                r"if.*if.*if",
                r"for.*for.*for",
                r"while.*while",
                r"try.*except.*except",
                r"lambda.*lambda",
                r"list.*comprehension.*list.*comprehension",
            ],
            "code_smells": [
                r"def.*def.*def.*def.*def",  # Слишком много вложенных функций
                r"class.*class.*class.*class.*class",  # Слишком много вложенных классов
                r"import.*\*",  # Wildcard imports
                r"from.*import.*\*",  # Wildcard imports
                r"global.*global",  # Множественные global
                r"nonlocal.*nonlocal",  # Множественные nonlocal
            ],
            "best_practices": [
                r"print\(",
                r"assert.*,",
                r"except:",
                r"except Exception:",
                r"finally:",
                r"pass",
                r"return None",
                r"return True",
                r"return False",
            ],
        }
        self.todo_patterns = [
            r"#\s*TODO[:\s]*(.+)",
            r"#\s*FIXME[:\s]*(.+)",
            r"#\s*HACK[:\s]*(.+)",
            r"#\s*XXX[:\s]*(.+)",
            r"#\s*BUG[:\s]*(.+)",
            r"#\s*NOTE[:\s]*(.+)",
            r"#\s*WARNING[:\s]*(.+)",
            r"#\s*DEPRECATED[:\s]*(.+)",
            r"\"\"\".*TODO.*\"\"\"",
            r"\"\"\".*FIXME.*\"\"\"",
            r"\"\"\".*HACK.*\"\"\"",
        ]

    async def analyze_code(
        self, code_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Промышленный анализ кода и выявление проблем."""
        logger.info("Начало промышленного анализа кода")
        issues: List[CodeIssue] = []
        # Анализ файлов
        for file_path, metrics in code_structure.get("complexity_metrics", {}).items():
            file_issues = await self._analyze_file(file_path, metrics)
            issues.extend(file_issues)
        # Анализ архитектурных проблем
        architecture_issues = await self._analyze_architecture(
            code_structure.get("architecture", {})
        )
        issues.extend(architecture_issues)
        # Анализ TODO и комментариев
        todo_issues = await self._analyze_todos_and_comments(code_structure)
        issues.extend(todo_issues)
        # Анализ безопасности
        security_issues = await self._analyze_security(code_structure)
        issues.extend(security_issues)
        # Анализ производительности
        performance_issues = await self._analyze_performance(code_structure)
        issues.extend(performance_issues)
        # Анализ code smells
        smell_issues = await self._analyze_code_smells(code_structure)
        issues.extend(smell_issues)
        # Сортировка по приоритету
        issues.sort(key=lambda x: x.priority, reverse=True)
        # Преобразование в словари для совместимости
        issue_dicts = [
            {
                "type": issue.type,
                "severity": issue.severity,
                "file": issue.file,
                "line": issue.line,
                "description": issue.description,
                "suggestion": issue.suggestion,
                "priority": issue.priority,
                "details": issue.details,
                "detected_at": (
                    issue.detected_at.isoformat() if issue.detected_at else None
                ),
            }
            for issue in issues
        ]
        logger.info(f"Промышленный анализ кода завершен, найдено {len(issues)} проблем")
        return issue_dicts

    async def _analyze_file(
        self, file_path: str, metrics: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Промышленный анализ отдельного файла."""
        issues: List[CodeIssue] = []
        # Анализ сложности
        if "complexity" in metrics:
            complexity = metrics["complexity"]
            if complexity.get("cyclomatic", 0) > 10:
                issues.append(
                    CodeIssue(
                        type="complexity",
                        severity="high",
                        file=file_path,
                        line=None,
                        description=(
                            f"Высокая цикломатическая сложность: {complexity['cyclomatic']}"
                        ),
                        suggestion=(
                            "Разбить функцию на более мелкие части, "
                            "использовать стратегии и паттерны"
                        ),
                        priority=0.9,
                        details={
                            "cyclomatic_complexity": complexity["cyclomatic"],
                            "threshold": 10,
                            "impact": "Высокая сложность затрудняет тестирование и поддержку",
                        },
                    )
                )
            if complexity.get("cognitive", 0) > 15:
                issues.append(
                    CodeIssue(
                        type="complexity",
                        severity="medium",
                        file=file_path,
                        line=None,
                        description=f"Высокая когнитивная сложность: {complexity['cognitive']}",
                        suggestion="Упростить логику функции, разбить на подфункции",
                        priority=0.7,
                        details={
                            "cognitive_complexity": complexity["cognitive"],
                            "threshold": 15,
                            "impact": "Сложная логика затрудняет понимание кода",
                        },
                    )
                )
            if complexity.get("nesting", 0) > 5:
                issues.append(
                    CodeIssue(
                        type="complexity",
                        severity="medium",
                        file=file_path,
                        line=None,
                        description=f"Глубокая вложенность: {complexity['nesting']} уровней",
                        suggestion="Уменьшить вложенность, использовать ранний возврат",
                        priority=0.6,
                        details={
                            "nesting_depth": complexity["nesting"],
                            "threshold": 5,
                            "impact": "Глубокая вложенность затрудняет чтение кода",
                        },
                    )
                )
        # Анализ качества
        if "quality" in metrics:
            quality = metrics["quality"]
            if quality.get("score", 100) < 70:
                issues.append(
                    CodeIssue(
                        type="quality",
                        severity="medium",
                        file=file_path,
                        line=None,
                        description=f"Низкое качество кода: {quality['score']}/100",
                        suggestion="Исправить выявленные проблемы качества, добавить документацию",
                        priority=0.8,
                        details={
                            "quality_score": quality["score"],
                            "threshold": 70,
                            "issues": quality.get("issues", []),
                            "suggestions": quality.get("suggestions", []),
                        },
                    )
                )
        # Анализ размера
        if metrics.get("lines", 0) > 500:
            issues.append(
                CodeIssue(
                    type="size",
                    severity="medium",
                    file=file_path,
                    line=None,
                    description=f"Большой файл: {metrics['lines']} строк",
                    suggestion="Разбить файл на модули, выделить отдельные классы/функции",
                    priority=0.6,
                    details={
                        "line_count": metrics["lines"],
                        "threshold": 500,
                        "impact": "Большие файлы затрудняют навигацию и понимание",
                    },
                )
            )
        # Анализ функций
        if metrics.get("functions", 0) > 20:
            issues.append(
                CodeIssue(
                    type="size",
                    severity="low",
                    file=file_path,
                    line=None,
                    description=f"Много функций в файле: {metrics['functions']}",
                    suggestion="Рассмотреть разделение на модули",
                    priority=0.4,
                    details={"function_count": metrics["functions"], "threshold": 20},
                )
            )
        return issues

    async def _analyze_architecture(
        self, architecture: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Промышленный анализ архитектурных проблем."""
        issues: List[CodeIssue] = []
        # Анализ зависимостей
        dependencies = architecture.get("dependencies", {})
        if "graph" in dependencies:
            for module, deps_info in dependencies["graph"].items():
                import_count = deps_info.get("import_count", 0)
                if import_count > 20:
                    issues.append(
                        CodeIssue(
                            type="architecture",
                            severity="medium",
                            file=module,
                            line=None,
                            description=f"Много зависимостей: {import_count} импортов",
                            suggestion="Уменьшить количество импортов, использовать интерфейсы",
                            priority=0.7,
                            details={
                                "import_count": import_count,
                                "threshold": 20,
                                "impact": "Высокая связанность затрудняет тестирование",
                            },
                        )
                    )
        # Анализ циклических зависимостей
        if "circular" in dependencies and dependencies["circular"]:
            for cycle in dependencies["circular"]:
                issues.append(
                    CodeIssue(
                        type="architecture",
                        severity="high",
                        file=", ".join(cycle),
                        line=None,
                        description=f"Циклическая зависимость: {' -> '.join(cycle)}",
                        suggestion="Разорвать цикл, использовать инверсию зависимостей",
                        priority=0.9,
                        details={
                            "cycle": cycle,
                            "impact": "Циклические зависимости нарушают принципы SOLID",
                        },
                    )
                )
        # Анализ модулей
        modules = architecture.get("modules", {})
        large_modules = [
            name
            for name, info in modules.items()
            if info.get("metrics", {}).get("lines", 0) > 1000
        ]
        if large_modules:
            issues.append(
                CodeIssue(
                    type="architecture",
                    severity="medium",
                    file=", ".join(large_modules),
                    line=None,
                    description=f"Большие модули: {len(large_modules)} модулей > 1000 строк",
                    suggestion="Разбить модули на более мелкие компоненты",
                    priority=0.7,
                    details={
                        "large_modules": large_modules,
                        "threshold": 1000,
                        "impact": "Большие модули затрудняют понимание и тестирование",
                    },
                )
            )
        return issues

    async def _analyze_todos_and_comments(
        self, code_structure: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Анализ TODO, FIXME и других комментариев."""
        issues: List[CodeIssue] = []
        # Анализ файлов на предмет TODO комментариев
        for file_path in code_structure.get("files", []):
            if file_path.endswith(".py"):
                try:
                    file_issues = await self._scan_file_for_todos(file_path)
                    issues.extend(file_issues)
                except Exception as e:
                    logger.warning(f"Ошибка анализа TODO в файле {file_path}: {e}")
        return issues

    async def _scan_file_for_todos(self, file_path: str) -> List[CodeIssue]:
        """Сканирование файла на предмет TODO комментариев."""
        issues: List[CodeIssue] = []
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return issues
            with open(file_path_obj, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                for pattern in self.todo_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        todo_type = self._extract_todo_type(line)
                        description = (
                            match.group(1).strip()
                            if match.groups()
                            else "TODO комментарий"
                        )
                        issues.append(
                            CodeIssue(
                                type="todo",
                                severity=self._get_todo_severity(todo_type),
                                file=file_path,
                                line=line_num,
                                description=f"{todo_type}: {description}",
                                suggestion=self._get_todo_suggestion(todo_type),
                                priority=self._get_todo_priority(todo_type),
                                details={
                                    "todo_type": todo_type,
                                    "line_content": line.strip(),
                                    "impact": "Незавершённая работа может привести к проблемам",
                                },
                            )
                        )
                        break  # Один TODO на строку
        except Exception as e:
            logger.error(f"Ошибка сканирования файла {file_path}: {e}")
        return issues

    def _extract_todo_type(self, line: str) -> str:
        """Извлечение типа TODO из строки."""
        line_upper = line.upper()
        if "FIXME" in line_upper:
            return "FIXME"
        elif "HACK" in line_upper:
            return "HACK"
        elif "BUG" in line_upper:
            return "BUG"
        elif "XXX" in line_upper:
            return "XXX"
        elif "WARNING" in line_upper:
            return "WARNING"
        elif "DEPRECATED" in line_upper:
            return "DEPRECATED"
        elif "NOTE" in line_upper:
            return "NOTE"
        else:
            return "TODO"

    def _get_todo_severity(self, todo_type: str) -> str:
        """Получение серьёзности TODO."""
        severity_map = {
            "FIXME": "high",
            "BUG": "high",
            "HACK": "medium",
            "WARNING": "medium",
            "DEPRECATED": "medium",
            "XXX": "low",
            "NOTE": "low",
            "TODO": "low",
        }
        return severity_map.get(todo_type, "low")

    def _get_todo_priority(self, todo_type: str) -> float:
        """Получение приоритета TODO."""
        priority_map = {
            "FIXME": 0.9,
            "BUG": 0.9,
            "HACK": 0.7,
            "WARNING": 0.6,
            "DEPRECATED": 0.5,
            "XXX": 0.4,
            "NOTE": 0.3,
            "TODO": 0.4,
        }
        return priority_map.get(todo_type, 0.4)

    def _get_todo_suggestion(self, todo_type: str) -> str:
        """Получение предложения для TODO."""
        suggestion_map = {
            "FIXME": "Исправить выявленную проблему",
            "BUG": "Исправить баг",
            "HACK": "Заменить временное решение на правильное",
            "WARNING": "Обратить внимание на предупреждение",
            "DEPRECATED": "Заменить устаревший код на современный",
            "XXX": "Завершить незавершённую работу",
            "NOTE": "Рассмотреть примечание",
            "TODO": "Завершить запланированную работу",
        }
        return suggestion_map.get(todo_type, "Завершить работу")

    async def _analyze_security(
        self, code_structure: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Анализ проблем безопасности."""
        issues: List[CodeIssue] = []
        # Анализ файлов на предмет проблем безопасности
        for file_path in code_structure.get("files", []):
            if file_path.endswith(".py"):
                try:
                    security_issues = await self._scan_file_for_security(file_path)
                    issues.extend(security_issues)
                except Exception as e:
                    logger.warning(
                        f"Ошибка анализа безопасности в файле {file_path}: {e}"
                    )
        return issues

    async def _scan_file_for_security(self, file_path: str) -> List[CodeIssue]:
        """Сканирование файла на предмет проблем безопасности."""
        issues: List[CodeIssue] = []
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return issues
            with open(file_path_obj, "r", encoding="utf-8") as f:
                content = f.read()
            # Поиск опасных паттернов
            for pattern in self.issue_patterns["security"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    issues.append(
                        CodeIssue(
                            type="security",
                            severity="high",
                            file=file_path,
                            line=line_num,
                            description=f"Потенциальная проблема безопасности: {match.group()}",
                            suggestion="Заменить на безопасную альтернативу",
                            priority=0.9,
                            details={
                                "pattern": pattern,
                                "matched_text": match.group(),
                                "impact": "Может привести к уязвимостям безопасности",
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"Ошибка сканирования безопасности в файле {file_path}: {e}")
        return issues

    async def _analyze_performance(
        self, code_structure: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Анализ проблем производительности."""
        issues: List[CodeIssue] = []
        # Анализ файлов на предмет проблем производительности
        for file_path in code_structure.get("files", []):
            if file_path.endswith(".py"):
                try:
                    performance_issues = await self._scan_file_for_performance(
                        file_path
                    )
                    issues.extend(performance_issues)
                except Exception as e:
                    logger.warning(
                        f"Ошибка анализа производительности в файле {file_path}: {e}"
                    )
        return issues

    async def _scan_file_for_performance(self, file_path: str) -> List[CodeIssue]:
        """Сканирование файла на предмет проблем производительности."""
        issues: List[CodeIssue] = []
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return issues
            with open(file_path_obj, "r", encoding="utf-8") as f:
                content = f.read()
            # Поиск неэффективных паттернов
            for pattern in self.issue_patterns["performance"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    issues.append(
                        CodeIssue(
                            type="performance",
                            severity="medium",
                            file=file_path,
                            line=line_num,
                            description=f"Потенциальная проблема производительности: {match.group()}",
                            suggestion="Оптимизировать для лучшей производительности",
                            priority=0.6,
                            details={
                                "pattern": pattern,
                                "matched_text": match.group(),
                                "impact": "Может снизить производительность",
                            },
                        )
                    )
        except Exception as e:
            logger.error(
                f"Ошибка сканирования производительности в файле {file_path}: {e}"
            )
        return issues

    async def _analyze_code_smells(
        self, code_structure: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Анализ code smells."""
        issues: List[CodeIssue] = []
        # Анализ файлов на предмет code smells
        for file_path in code_structure.get("files", []):
            if file_path.endswith(".py"):
                try:
                    smell_issues = await self._scan_file_for_smells(file_path)
                    issues.extend(smell_issues)
                except Exception as e:
                    logger.warning(
                        f"Ошибка анализа code smells в файле {file_path}: {e}"
                    )
        return issues

    async def _scan_file_for_smells(self, file_path: str) -> List[CodeIssue]:
        """Сканирование файла на предмет code smells."""
        issues: List[CodeIssue] = []
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return issues
            with open(file_path_obj, "r", encoding="utf-8") as f:
                content = f.read()
            # Поиск code smells
            for pattern in self.issue_patterns["code_smells"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    issues.append(
                        CodeIssue(
                            type="code_smell",
                            severity="medium",
                            file=file_path,
                            line=line_num,
                            description=f"Code smell: {match.group()}",
                            suggestion="Рефакторинг для улучшения читаемости и поддерживаемости",
                            priority=0.5,
                            details={
                                "pattern": pattern,
                                "matched_text": match.group(),
                                "impact": "Снижает читаемость и поддерживаемость кода",
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"Ошибка сканирования code smells в файле {file_path}: {e}")
        return issues

    def get_analysis_summary(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Получение сводки анализа."""
        if not issues:
            return {"total_issues": 0}
        summary: Dict[str, Any] = {
            "total_issues": len(issues),
            "by_type": {},
            "by_severity": {},
            "by_priority": {"high": 0, "medium": 0, "low": 0},
        }
        by_type: Dict[str, int] = summary["by_type"]
        by_severity: Dict[str, int] = summary["by_severity"]
        by_priority: Dict[str, int] = summary["by_priority"]
        
        for issue in issues:
            # Подсчёт по типу
            issue_type = issue.type
            by_type[issue_type] = by_type.get(issue_type, 0) + 1
            # Подсчёт по серьёзности
            severity = issue.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1
            # Подсчёт по приоритету
            priority = issue.priority
            if priority >= 0.8:
                by_priority["high"] += 1
            elif priority >= 0.5:
                by_priority["medium"] += 1
            else:
                by_priority["low"] += 1
        return summary
