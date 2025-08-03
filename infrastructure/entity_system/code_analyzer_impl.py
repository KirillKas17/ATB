from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from domain.types.entity_system_types import (
    AnalysisResult,
    BaseCodeAnalyzer,
    CodeStructure,
    Hypothesis,
)


class CodeAnalyzerImpl(BaseCodeAnalyzer):
    """
    Продвинутый анализатор кода:
    - Анализирует качество, производительность, поддерживаемость, сложность
    - Генерирует гипотезы по улучшению
    - Валидирует гипотезы на основе метрик
    """

    async def analyze_code(self, code_structure: CodeStructure) -> AnalysisResult:
        logger.info(f"Анализ кода: {code_structure['file_path']}")
        quality_score = self._calculate_quality_score(code_structure)
        performance_score = self._calculate_performance_score(code_structure)
        maintainability_score = self._calculate_maintainability_score(code_structure)
        complexity_score = self._calculate_complexity_score(code_structure)
        suggestions = self._generate_suggestions(code_structure)
        issues = self._identify_issues(code_structure)
        return AnalysisResult(
            file_path=code_structure["file_path"],
            quality_score=quality_score,
            performance_score=performance_score,
            maintainability_score=maintainability_score,
            complexity_score=complexity_score,
            suggestions=suggestions,
            issues=issues,
            timestamp=datetime.now(),
        )

    async def generate_hypotheses(
        self, analysis_results: List[AnalysisResult]
    ) -> List[Hypothesis]:
        logger.info("Генерация гипотез по результатам анализа кода")
        hypotheses = []
        for result in analysis_results:
            if result["quality_score"] < 0.7:
                hypotheses.append(
                    Hypothesis(
                        id=f"hypo_{result['file_path']}",
                        description="Улучшить качество кода (рефакторинг, тесты)",
                        expected_improvement=0.2,
                        confidence=0.9,
                        implementation_cost=0.3,
                        risk_level="medium",
                        category="maintainability",
                        created_at=datetime.now(),
                        status="pending",
                    )
                )
            if result["complexity_score"] > 0.7:
                hypotheses.append(
                    Hypothesis(
                        id=f"hypo_complex_{result['file_path']}",
                        description="Снизить сложность (разделить функции, декомпозиция)",
                        expected_improvement=0.15,
                        confidence=0.85,
                        implementation_cost=0.4,
                        risk_level="low",
                        category="architecture",
                        created_at=datetime.now(),
                        status="pending",
                    )
                )
        return hypotheses

    async def validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        logger.info(f"Валидация гипотезы: {hypothesis['id']}")
        # Пример: простая валидация по confidence и expected_improvement
        return (
            hypothesis["confidence"] > 0.7 and hypothesis["expected_improvement"] > 0.1
        )

    def _calculate_quality_score(self, code_structure: CodeStructure) -> float:
        # Чем меньше длинных строк и ошибок, тем выше качество
        long_lines: int = code_structure["quality_metrics"].get("long_lines", 0)
        total_lines: int = code_structure["quality_metrics"].get("total_lines", 1)
        return max(0.0, 1.0 - long_lines / total_lines)

    def _calculate_performance_score(self, code_structure: CodeStructure) -> float:
        # Пример: если много циклов — ниже производительность
        cyclomatic: int = code_structure["complexity_metrics"].get("cyclomatic", 0)
        return max(0.0, 1.0 - 0.05 * cyclomatic)

    def _calculate_maintainability_score(self, code_structure: CodeStructure) -> float:
        # Чем меньше классов и функций — тем проще поддерживать
        n_classes: int = len(code_structure["classes"])
        n_functions: int = len(code_structure["functions"])
        return max(0.0, 1.0 - 0.02 * (n_classes + n_functions))

    def _calculate_complexity_score(self, code_structure: CodeStructure) -> float:
        cyclomatic: int = code_structure["complexity_metrics"].get("cyclomatic", 0)
        return min(1.0, cyclomatic / 10.0)

    def _generate_suggestions(
        self, code_structure: CodeStructure
    ) -> List[Dict[str, Any]]:
        suggestions = []
        if code_structure["quality_metrics"].get("long_lines", 0) > 0:
            suggestions.append({"action": "Разбить длинные строки", "priority": "high"})
        if code_structure["complexity_metrics"].get("cyclomatic", 0) > 5:
            suggestions.append({"action": "Упростить логику", "priority": "medium"})
        return suggestions

    def _identify_issues(self, code_structure: CodeStructure) -> List[Dict[str, Any]]:
        issues = []
        if code_structure["quality_metrics"].get("long_lines", 0) > 10:
            issues.append({"type": "style", "message": "Слишком много длинных строк"})
        if code_structure["complexity_metrics"].get("cyclomatic", 0) > 10:
            issues.append(
                {"type": "complexity", "message": "Высокая цикломатическая сложность"}
            )
        return issues
