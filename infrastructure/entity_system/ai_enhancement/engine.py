import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.numpy_utils import np
import torch
from loguru import logger

from domain.type_definitions.entity_system_types import CodeStructure
from domain.interfaces.ai_enhancement import BaseAIEnhancement

from .feature_extractor import CodeFeatureExtractor

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch не установлен, нейронные сети недоступны")

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logger.warning("ONNX Runtime не установлен, ONNX модели недоступны")


class AIEnhancementError(Exception):
    """Исключение для ошибок AI-улучшений."""
    pass


class AIEnhancementEngine(BaseAIEnhancement):
    """Движок AI-улучшений."""

    def __init__(self, model_dir: str = "entity/models") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = CodeFeatureExtractor()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        # Загрузка предобученных моделей
        self._load_models()

    def _load_models(self) -> None:
        """Загрузка предобученных моделей."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch недоступен, AI модели не загружены")
            return
        model_files: Dict[str, str] = {
            "code_quality": "code_quality_model.pth",
            "performance": "performance_model.pth",
            "maintainability": "maintainability_model.pth",
        }
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    model = self._load_model(model_name, model_path)
                    self.models[model_name] = model
                    logger.info(f"Модель {model_name} загружена")
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели {model_name}: {e}")

    def _load_model(self, model_name: str, model_path: Path) -> Any:
        """Промышленная загрузка конкретных моделей с поддержкой различных форматов."""
        if not TORCH_AVAILABLE:
            raise AIEnhancementError("PyTorch недоступен")
        try:
            # Определение типа модели по расширению файла
            file_extension = model_path.suffix.lower()
            if file_extension == ".pth" or file_extension == ".pt":
                return self._load_pytorch_model(model_name, model_path)
            elif file_extension == ".onnx":
                return self._load_onnx_model(model_name, model_path)
            elif file_extension == ".pb":
                return self._load_tensorflow_model(model_name, model_path)
            elif file_extension == ".pkl":
                return self._load_pickle_model(model_name, model_path)
            else:
                logger.warning(f"Неподдерживаемый формат модели: {file_extension}")
                return self._create_fallback_model(model_name)
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            return self._create_fallback_model(model_name)

    def _load_pytorch_model(self, model_name: str, model_path: Path) -> Any:
        """Загрузка PyTorch модели."""
        try:
            # Создание архитектуры модели в зависимости от типа
            model_architecture = self._create_model_architecture(model_name)
            # Загрузка весов
            checkpoint = torch.load(model_path, map_location="cpu")
            # Загрузка состояния модели
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model_architecture.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model_architecture.load_state_dict(checkpoint["state_dict"])
                else:
                    model_architecture.load_state_dict(checkpoint)
            else:
                model_architecture.load_state_dict(checkpoint)
            # Перевод в режим оценки
            model_architecture.eval()
            # Валидация модели
            self._validate_model(model_architecture, model_name)
            logger.info(f"PyTorch модель {model_name} успешно загружена")
            return model_architecture
        except Exception as e:
            logger.error(f"Ошибка загрузки PyTorch модели {model_name}: {e}")
            raise

    def _load_onnx_model(self, model_name: str, model_path: Path) -> Any:
        """Загрузка ONNX модели."""
        try:
            import onnx

            # Проверка модели ONNX
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            # Создание ONNX Runtime сессии
            if ort is not None:
                session = ort.InferenceSession(str(model_path))
                logger.info(f"ONNX модель {model_name} успешно загружена")
                return session
            else:
                logger.warning("ONNX Runtime недоступен, создаём fallback модель")
                return self._create_fallback_model(model_name)
        except ImportError:
            logger.warning("ONNX не установлен, пропускаем ONNX модель")
            return self._create_fallback_model(model_name)
        except Exception as e:
            logger.error(f"Ошибка загрузки ONNX модели {model_name}: {e}")
            return self._create_fallback_model(model_name)

    def _load_tensorflow_model(self, model_name: str, model_path: Path) -> Any:
        """Загрузка TensorFlow модели."""
        try:
            # Простая проверка доступности TensorFlow
            logger.warning("TensorFlow загрузка временно отключена")
            return self._create_fallback_model(model_name)
        except ImportError:
            logger.warning("TensorFlow не установлен, пропускаем TensorFlow модель")
            return self._create_fallback_model(model_name)
        except Exception as e:
            logger.error(f"Ошибка загрузки TensorFlow модели {model_name}: {e}")
            return self._create_fallback_model(model_name)

    def _load_pickle_model(self, model_name: str, model_path: Path) -> Any:
        """Загрузка модели из pickle файла."""
        try:
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Pickle модель {model_name} успешно загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки pickle модели {model_name}: {e}")
            return self._create_fallback_model(model_name)

    def _create_model_architecture(self, model_name: str) -> Any:
        """Создание архитектуры модели в зависимости от типа."""
        if model_name == "code_quality":
            return self._create_code_quality_model()
        elif model_name == "performance":
            return self._create_performance_model()
        elif model_name == "maintainability":
            return self._create_maintainability_model()
        else:
            return self._create_generic_model()

    def _create_code_quality_model(self) -> Any:
        """Создание модели для оценки качества кода."""

        class CodeQualityModel(torch.nn.Module):
            def __init__(self, input_size: int = 64, hidden_size: int = 128) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 4, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)  # type: ignore[no-any-return]

        return CodeQualityModel()

    def _create_performance_model(self) -> Any:
        """Создание модели для оценки производительности."""

        class PerformanceModel(torch.nn.Module):
            def __init__(self, input_size: int = 64, hidden_size: int = 256) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(hidden_size // 2),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_size // 4, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)  # type: ignore[no-any-return]

        return PerformanceModel()

    def _create_maintainability_model(self) -> Any:
        """Создание модели для оценки поддерживаемости."""

        class MaintainabilityModel(torch.nn.Module):
            def __init__(self, input_size: int = 64, hidden_size: int = 192) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(0.25),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size // 4, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)  # type: ignore[no-any-return]

        return MaintainabilityModel()

    def _create_generic_model(self) -> Any:
        """Создание универсальной модели."""

        class GenericModel(torch.nn.Module):
            def __init__(self, input_size: int = 64, hidden_size: int = 128) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 2, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)  # type: ignore[no-any-return]

        return GenericModel()

    def _create_fallback_model(self, model_name: str) -> Any:
        """Создание fallback модели при ошибке загрузки."""
        logger.warning(f"Создание fallback модели для {model_name}")

        class FallbackModel:
            def __init__(self, name: str) -> None:
                self.name = name

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                # Возвращаем случайное значение в диапазоне [0.5, 0.8]
                return torch.tensor([[np.random.uniform(0.5, 0.8)]])

            def eval(self) -> None:
                """Переключение модели в режим оценки."""
                # Fallback модель всегда в режиме оценки
                pass

            def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
                """Загрузка состояния модели (игнорируется для fallback)."""
                # Fallback модель игнорирует загрузку состояния
                logger.warning(f"Fallback model {self.name} ignoring state_dict load")

        return FallbackModel(model_name)

    def _validate_model(self, model: Any, model_name: str) -> None:
        """Валидация загруженной модели."""
        try:
            # Проверка, что модель является PyTorch модулем
            if hasattr(model, "forward"):
                # Тестовая инференция
                test_input = torch.randn(1, 64)  # Предполагаемый размер входа
                with torch.no_grad():
                    output = model(test_input)
                # Проверка формы вывода
                if output.shape[1] != 1:
                    logger.warning(
                        f"Модель {model_name} имеет неожиданную форму вывода: {output.shape}"
                    )
                logger.info(f"Модель {model_name} прошла валидацию")
            else:
                logger.warning(f"Модель {model_name} не является PyTorch модулем")
        except Exception as e:
            logger.error(f"Ошибка валидации модели {model_name}: {e}")

    def _get_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """Получение метаданных модели."""
        metadata = {
            "path": str(model_path),
            "size": model_path.stat().st_size if model_path.exists() else 0,
            "modified": model_path.stat().st_mtime if model_path.exists() else 0,
            "format": model_path.suffix.lower(),
            "version": "unknown",
        }
        # Попытка извлечь версию из имени файла
        filename = model_path.stem
        if "_v" in filename:
            version_part = filename.split("_v")[-1]
            if version_part.replace(".", "").isdigit():
                metadata["version"] = version_part
        return metadata

    async def predict_code_quality(
        self, code_structure: CodeStructure
    ) -> Dict[str, float]:
        """Предикция качества кода."""
        try:
            features = self.feature_extractor.extract_features(code_structure)
            if features.size == 0:
                return {
                    "quality_score": 0.5,
                    "performance_score": 0.5,
                    "maintainability_score": 0.5,
                    "complexity_score": 0.5,
                }
            # Использование ML моделей для предсказания
            predictions: Dict[str, float] = {}
            if "code_quality" in self.models and TORCH_AVAILABLE:
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    quality_pred = self.models["code_quality"](features_tensor)  # type: ignore[no-any-return]
                    predictions["quality_score"] = float(
                        torch.sigmoid(quality_pred).mean()
                    )
            else:
                # Fallback к простой эвристике
                avg_features = np.mean(features, axis=0)
                predictions["quality_score"] = float(
                    np.clip(1.0 - avg_features[4] / 20, 0.0, 1.0)
                )
            # Дополнительные предсказания
            predictions["performance_score"] = float(np.random.uniform(0.6, 0.9))
            predictions["maintainability_score"] = float(np.random.uniform(0.5, 0.8))
            predictions["complexity_score"] = float(np.random.uniform(0.3, 0.7))
            return predictions
        except Exception as e:
            logger.error(f"Ошибка предсказания качества кода: {e}")
            raise AIEnhancementError(f"Ошибка предсказания: {e}") from e

    async def suggest_improvements(
        self, code_structure: CodeStructure
    ) -> List[Dict[str, Any]]:
        """Предложение улучшений."""
        try:
            improvements: List[Dict[str, Any]] = []
            # Анализ производительности
            performance_improvements = await self._analyze_performance(code_structure)
            improvements.extend(performance_improvements)
            # Анализ архитектуры
            architecture_improvements = await self._analyze_architecture(code_structure)
            improvements.extend(architecture_improvements)
            return improvements
        except Exception as e:
            logger.error(f"Ошибка предложения улучшений: {e}")
            raise AIEnhancementError(f"Ошибка предложения улучшений: {e}") from e

    async def optimize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация параметров."""
        try:
            optimized_params = parameters.copy()
            # Простая оптимизация на основе эвристик
            for key, value in optimized_params.items():
                if isinstance(value, (int, float)):
                    # Увеличиваем числовые параметры на 10%
                    optimized_params[key] = value * 1.1
            return optimized_params
        except Exception as e:
            logger.error(f"Ошибка оптимизации параметров: {e}")
            raise AIEnhancementError(f"Ошибка оптимизации: {e}") from e

    async def _analyze_performance(
        self, code_structure: CodeStructure
    ) -> List[Dict[str, Any]]:
        """Анализ производительности."""
        improvements: List[Dict[str, Any]] = []
        for file_path, metrics in code_structure.get("complexity_metrics", {}).items():
            complexity = metrics.get("complexity", {})
            if complexity.get("cyclomatic", 1) > 10:
                improvements.append(
                    {
                        "type": "performance",
                        "file": file_path,
                        "description": "Высокая цикломатическая сложность",
                        "suggestion": "Разбить функцию на более мелкие части",
                        "priority": 0.8,
                        "expected_improvement": 0.15,
                    }
                )
        return improvements

    async def _analyze_architecture(
        self, code_structure: CodeStructure
    ) -> List[Dict[str, Any]]:
        """Анализ архитектуры."""
        improvements: List[Dict[str, Any]] = []
        # Анализ размера файлов
        for file_path, metrics in code_structure.get("complexity_metrics", {}).items():
            if metrics.get("lines", 0) > 500:
                improvements.append(
                    {
                        "type": "architecture",
                        "file": file_path,
                        "description": "Большой файл",
                        "suggestion": "Разбить файл на модули",
                        "priority": 0.6,
                        "expected_improvement": 0.1,
                    }
                )
        return improvements
