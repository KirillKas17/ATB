"""
Timestamp value object.
"""

from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass(frozen=True)
class Timestamp:
    """Timestamp value object."""
    
    value: datetime
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """Creates current timestamp."""
        return cls(datetime.now(timezone.utc))
    
    @classmethod
    def from_iso(cls, iso_string: str) -> 'Timestamp':
        """Creates timestamp from ISO string."""
        return cls(datetime.fromisoformat(iso_string.replace('Z', '+00:00')))
    
    def __str__(self) -> str:
        return self.value.isoformat()
