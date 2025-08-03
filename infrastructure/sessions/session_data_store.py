"""Хранилище данных торговых сессий (устарело)."""


class SessionDataStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "SessionDataStore устарел. Используйте SessionRepository и новые сервисы сессий."
        )
