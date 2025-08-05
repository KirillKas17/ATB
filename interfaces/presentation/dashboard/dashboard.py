from typing import Any, Dict


class Dashboard:
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    def update_data(self, new_data: Dict[str, Any]) -> None:
        self.data.update(new_data)

    def get_data(self) -> Dict[str, Any]:
        return self.data
