import json
import os
from typing import Any, Dict, Optional


class ConfigLoader:
    """Класс для загрузки конфигурации"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.json")
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Загрузка конфигурации из файла"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
            print(f"Warning: Config file {self.config_path} not found")
        except json.JSONDecodeError:
            self.config = {}
            print(f"Error: Invalid JSON in config file {self.config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Получение значения из конфигурации"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Установка значения в конфигурации"""
        self.config[key] = value

    def save(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def reload(self) -> None:
        """Перезагрузка конфигурации из файла"""
        self.load_config()
