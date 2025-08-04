"""
Простая обёртка для SessionProfile без проблем с Pydantic
"""

from typing import Dict, List, Optional
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SessionProfile:
    """Простая обёртка для работы с сессионными профилями."""
    
    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self.profiles_count = 4  # Asian, London, NY, Sydney
        self._is_initialized = True
        
        logger.info(f"SessionProfile создан для символа {symbol} с {self.profiles_count} профилями")
    
    def get_profile(self, session_type: str) -> Dict[str, str]:
        """Получить профиль сессии."""
        return {
            "session_type": session_type,
            "symbol": self.symbol,
            "status": "active"
        }
    
    def get_all_profiles(self) -> Dict[str, dict]:
        """Получить все профили."""
        return {
            "asian": self.get_profile("asian"),
            "london": self.get_profile("london"),
            "ny": self.get_profile("ny"),
            "sydney": self.get_profile("sydney")
        }
    
    def get_current_session_profile(self) -> Dict[str, str]:
        """Получить профиль текущей сессии."""
        return self.get_profile("london")  # Default