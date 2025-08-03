"""
Доменная сущность параметров стратегии.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Mapping
from uuid import UUID, uuid4

# Унифицированный тип для параметров, поддерживающий вложенные структуры
ParameterValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ParameterDict = Dict[str, ParameterValue]

# Тип для совместимости с Mapping (ковариантный)
ParameterMapping = Mapping[str, ParameterValue]


@dataclass
class StrategyParameters:
    """Параметры стратегии"""

    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    parameters: ParameterDict = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: ParameterDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Пост-инициализация с валидацией"""
        if not isinstance(self.parameters, dict):
            self.parameters = ParameterDict()  # type: ignore[unreachable]
        if not isinstance(self.metadata, dict):
            self.metadata = ParameterDict()  # type: ignore[unreachable]

    def set_parameter(self, key: str, value: ParameterValue) -> None:
        """Установить параметр"""
        self.parameters[key] = value
        self.updated_at = datetime.now()

    def get_parameter(
        self, key: str, default: Optional[ParameterValue] = None
    ) -> Optional[ParameterValue]:
        """Получить параметр"""
        return self.parameters.get(key, default)

    def remove_parameter(self, key: str) -> bool:
        """Удалить параметр"""
        if key in self.parameters:
            del self.parameters[key]
            self.updated_at = datetime.now()
            return True
        return False

    def has_parameter(self, key: str) -> bool:
        """Проверить наличие параметра"""
        return key in self.parameters

    def get_all_parameters(self) -> ParameterDict:
        """Получить все параметры"""
        return self.parameters.copy()

    def update_parameters(self, new_parameters: ParameterMapping) -> None:
        """Обновить параметры"""
        self.parameters.update(new_parameters)
        self.updated_at = datetime.now()

    def clear_parameters(self) -> None:
        """Очистить все параметры"""
        self.parameters.clear()
        self.updated_at = datetime.now()

    def validate_parameters(self) -> List[str]:
        """Валидация параметров"""
        errors = []

        # Проверка обязательных параметров
        required_params = ["stop_loss", "take_profit", "position_size"]
        for param in required_params:
            if param not in self.parameters:
                errors.append(f"Missing required parameter: {param}")

        # Проверка типов и значений
        for key, value in self.parameters.items():
            if key == "stop_loss":
                if not isinstance(value, (int, float, Decimal, str)):
                    errors.append(f"stop_loss must be numeric, got {type(value)}")
                else:
                    try:
                        val = float(value)
                        if val <= 0 or val > 1:
                            errors.append("stop_loss must be between 0 and 1")
                    except (ValueError, TypeError):
                        errors.append("stop_loss must be a valid number")

            elif key == "take_profit":
                if not isinstance(value, (int, float, Decimal, str)):
                    errors.append(f"take_profit must be numeric, got {type(value)}")
                else:
                    try:
                        val = float(value)
                        if val <= 0 or val > 10:
                            errors.append("take_profit must be between 0 and 10")
                    except (ValueError, TypeError):
                        errors.append("take_profit must be a valid number")

            elif key == "position_size":
                if not isinstance(value, (int, float, Decimal, str)):
                    errors.append(f"position_size must be numeric, got {type(value)}")
                else:
                    try:
                        val = float(value)
                        if val <= 0 or val > 1:
                            errors.append("position_size must be between 0 and 1")
                    except (ValueError, TypeError):
                        errors.append("position_size must be a valid number")

            elif key == "confidence_threshold":
                if not isinstance(value, (int, float, Decimal, str)):
                    errors.append(
                        f"confidence_threshold must be numeric, got {type(value)}"
                    )
                else:
                    try:
                        val = float(value)
                        if val < 0 or val > 1:
                            errors.append(
                                "confidence_threshold must be between 0 and 1"
                            )
                    except (ValueError, TypeError):
                        errors.append("confidence_threshold must be a valid number")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "parameters": dict(self.parameters),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyParameters":
        """Создание из словаря"""
        id_value = data.get("id", "")
        strategy_id_value = data.get("strategy_id", "")
        parameters_value = data.get("parameters", {})
        is_active = bool(data.get("is_active", True))
        created_at_value = data.get("created_at", "")
        updated_at_value = data.get("updated_at", "")
        metadata_value = data.get("metadata", {})

        try:
            id_uuid = UUID(str(id_value)) if id_value else uuid4()
        except ValueError:
            id_uuid = uuid4()

        try:
            strategy_id = UUID(str(strategy_id_value)) if strategy_id_value else uuid4()
        except ValueError:
            strategy_id = uuid4()

        if not isinstance(parameters_value, dict):
            parameters_value = {}
        else:
            parameters_value = dict(parameters_value)

        try:
            created_at = (
                datetime.fromisoformat(str(created_at_value))
                if created_at_value
                else datetime.now()
            )
        except ValueError:
            created_at = datetime.now()

        try:
            updated_at = (
                datetime.fromisoformat(str(updated_at_value))
                if updated_at_value
                else datetime.now()
            )
        except ValueError:
            updated_at = datetime.now()

        if not isinstance(metadata_value, dict):
            metadata_value = {}
        else:
            metadata_value = dict(metadata_value)

        return cls(
            id=id_uuid,
            strategy_id=strategy_id,
            parameters=parameters_value,
            is_active=is_active,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata_value,
        )

    def __str__(self) -> str:
        """Строковое представление параметров"""
        return f"StrategyParameters(strategy_id={self.strategy_id}, params_count={len(self.parameters)})"

    def __repr__(self) -> str:
        """Представление для отладки"""
        return f"StrategyParameters(id={self.id}, strategy_id={self.strategy_id}, parameters={self.parameters})"
