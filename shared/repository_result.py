"""
Результат операции репозитория.
"""

from typing import Any, Optional


class RepositoryResult:
    """Результат операции репозитория."""
    
    def __init__(self, success: bool, data: Any = None, error_message: str = "") -> None:
        self.success = success
        self.data = data
        self.error_message = error_message
    
    def is_success(self) -> bool:
        """Проверка успешности операции."""
        return self.success
    
    def get_data(self) -> Any:
        """Получение данных результата."""
        return self.data
    
    def get_error_message(self) -> str:
        """Получение сообщения об ошибке."""
        return self.error_message
    
    def __str__(self) -> str:
        if self.success:
            return f"RepositoryResult(success=True, data={self.data})"
        else:
            return f"RepositoryResult(success=False, error='{self.error_message}')"
    
    def __repr__(self) -> str:
        return self.__str__()