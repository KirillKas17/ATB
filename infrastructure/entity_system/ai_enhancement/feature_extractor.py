from typing import Any, Dict, List

from shared.numpy_utils import np

from domain.types.entity_system_types import CodeStructure


class CodeFeatureExtractor:
    """Экстрактор признаков кода для ML моделей."""

    def __init__(self) -> None:
        self.feature_names: List[str] = [
            "lines_of_code",
            "functions",
            "classes",
            "imports",
            "complexity_cyclomatic",
            "complexity_cognitive",
            "nesting_max",
            "comments_ratio",
            "docstrings_ratio",
            "variable_count",
            "function_length_avg",
            "class_length_avg",
            "import_diversity",
            "error_handling_ratio",
            "type_hints_ratio",
            "async_functions_ratio",
            "decorators_count",
            "inheritance_depth",
            "coupling_score",
            "cohesion_score",
            "maintainability_index",
        ]

    def extract_features(self, code_structure: CodeStructure) -> np.ndarray:
        features: List[List[float]] = []
        for file_path, metrics in code_structure.get("complexity_metrics", {}).items():
            file_features = self._extract_file_features(metrics)
            features.append(file_features)
        return np.array(features) if features else np.array([])

    def _extract_file_features(self, metrics: Dict[str, Any]) -> List[float]:
        features: List[float] = []
        features.append(float(metrics.get("lines", 0)))
        features.append(float(metrics.get("functions", 0)))
        features.append(float(metrics.get("classes", 0)))
        features.append(float(metrics.get("imports", 0)))
        complexity = metrics.get("complexity", {})
        features.append(float(complexity.get("cyclomatic", 1)))
        features.append(float(complexity.get("cognitive", 0)))
        features.append(float(complexity.get("nesting", 0)))
        quality = metrics.get("quality", {})
        features.append(float(quality.get("comments_ratio", 0)))
        features.append(float(quality.get("docstrings_ratio", 0)))
        features.extend(
            [
                float(metrics.get("variable_count", 0)),
                float(metrics.get("function_length_avg", 0)),
                float(metrics.get("class_length_avg", 0)),
                float(metrics.get("import_diversity", 0)),
                float(metrics.get("error_handling_ratio", 0)),
                float(metrics.get("type_hints_ratio", 0)),
                float(metrics.get("async_functions_ratio", 0)),
                float(metrics.get("decorators_count", 0)),
                float(metrics.get("inheritance_depth", 0)),
                float(metrics.get("coupling_score", 0)),
                float(metrics.get("cohesion_score", 0)),
                float(metrics.get("maintainability_index", 100)),
            ]
        )
        return features
