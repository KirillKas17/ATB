"""
Агент социальных сетей - основной файл.
Импортирует функциональность из модульной архитектуры.
"""

from .social_media.agent_social_media import SocialMediaAgent
from .social_media.types import (
    SocialPlatform,
    SocialPost,
    SocialSentimentResult,
    SocialMediaConfig,
)
from .social_media.providers import (
    ISocialMediaProvider,
    RedditProvider,
    TelegramProvider,
    DiscordProvider,
)

__all__ = [
    "SocialMediaAgent",
    "SocialPlatform",
    "SocialPost",
    "SocialSentimentResult",
    "SocialMediaConfig",
    "ISocialMediaProvider",
    "RedditProvider",
    "TelegramProvider",
    "DiscordProvider",
]
