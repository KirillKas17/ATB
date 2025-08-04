from typing import Dict, Any


class Dashboard:
    """Дашборд для отображения данных торговой системы."""
    
    def __init__(self) -> None:
        """Инициализация дашборда."""
        self.data: Dict[str, Any] = {}

    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Обновление данных дашборда."""
        self.data.update(new_data)

    def get_data(self) -> Dict[str, Any]:
        """Получение всех данных дашборда."""
        return self.data
