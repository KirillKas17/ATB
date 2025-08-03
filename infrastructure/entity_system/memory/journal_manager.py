import asyncio
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


@dataclass
class JournalEntry:
    id: int
    timestamp: datetime
    event_type: str
    message: str
    data: Optional[Dict[str, Any]]
    severity: str
    source: str
    session_id: Optional[str]


class JournalManager:
    """Промышленный менеджер журнала событий системы с персистентностью и индексацией."""

    def __init__(self, db_path: str = "logs/journal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Инициализация базы данных для журнала."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS journal_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data TEXT,
                        severity TEXT NOT NULL,
                        source TEXT NOT NULL,
                        session_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Создание индексов для быстрого поиска
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON journal_entries(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON journal_entries(event_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_severity ON journal_entries(severity)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_id ON journal_entries(session_id)"
                )

                conn.commit()
                logger.info(f"База данных журнала инициализирована: {self.db_path}")

        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных журнала: {e}")

    async def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        source: str = "system",
        session_id: Optional[str] = None,
    ) -> None:
        """Промышленное логирование события с персистентностью и индексацией."""
        try:
            entry = JournalEntry(
                id=0,  # Будет установлено базой данных
                timestamp=datetime.now(),
                event_type=event_type,
                message=message,
                data=data,
                severity=severity,
                source=source,
                session_id=session_id,
            )

            # Асинхронная запись в базу данных
            await self._write_entry_to_db(entry)

            # Логирование в консоль для отладки
            logger.info(f"Событие записано в журнал: {event_type} - {message}")

        except Exception as e:
            logger.error(f"Ошибка логирования события: {e}")

    async def _write_entry_to_db(self, entry: JournalEntry) -> None:
        """Асинхронная запись записи в базу данных."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_entry_sync, entry)

    def _write_entry_sync(self, entry: JournalEntry) -> None:
        """Синхронная запись записи в базу данных."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO journal_entries
                        (timestamp, event_type, message, data, severity, source, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entry.timestamp.isoformat(),
                            entry.event_type,
                            entry.message,
                            json.dumps(entry.data) if entry.data else None,
                            entry.severity,
                            entry.source,
                            entry.session_id,
                        ),
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"Ошибка записи в базу данных: {e}")

    async def get_journal_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Промышленное получение записей журнала с фильтрацией и пагинацией."""
        try:
            loop = asyncio.get_event_loop()
            entries = await loop.run_in_executor(
                None, self._get_entries_sync, limit, offset, start_date, end_date
            )

            # Преобразование в словари для JSON сериализации
            result = []
            for entry in entries:
                entry_dict = asdict(entry)
                entry_dict["timestamp"] = entry.timestamp.isoformat()
                # entry.data уже имеет правильный тип Optional[Dict[str, Any]]
                result.append(entry_dict)

            logger.info(f"Получено {len(result)} записей журнала")
            return result

        except Exception as e:
            logger.error(f"Ошибка получения записей журнала: {e}")
            return []

    def _get_entries_sync(
        self,
        limit: int,
        offset: int,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[JournalEntry]:
        """Синхронное получение записей из базы данных."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    query = "SELECT * FROM journal_entries WHERE 1=1"
                    params = []

                    if start_date:
                        query += " AND timestamp >= ?"
                        params.append(start_date.isoformat())

                    if end_date:
                        query += " AND timestamp <= ?"
                        params.append(end_date.isoformat())

                    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                    params.extend([str(limit), str(offset)])

                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()

                    entries = []
                    for row in rows:
                        entry = JournalEntry(
                            id=row["id"],
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            event_type=row["event_type"],
                            message=row["message"],
                            data=json.loads(row["data"]) if row["data"] else None,
                            severity=row["severity"],
                            source=row["source"],
                            session_id=row["session_id"],
                        )
                        entries.append(entry)

                    return entries

        except Exception as e:
            logger.error(f"Ошибка синхронного получения записей: {e}")
            return []

    async def get_journal_entries_by_type(
        self, event_type: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Промышленная фильтрация записей журнала по типу события."""
        try:
            loop = asyncio.get_event_loop()
            entries = await loop.run_in_executor(
                None, self._get_entries_by_type_sync, event_type, limit, offset
            )

            # Преобразование в словари
            result = []
            for entry in entries:
                entry_dict = asdict(entry)
                entry_dict["timestamp"] = entry.timestamp.isoformat()
                # entry.data уже имеет правильный тип Optional[Dict[str, Any]]
                result.append(entry_dict)

            logger.info(f"Получено {len(result)} записей типа {event_type}")
            return result

        except Exception as e:
            logger.error(f"Ошибка получения записей по типу {event_type}: {e}")
            return []

    def _get_entries_by_type_sync(
        self, event_type: str, limit: int, offset: int
    ) -> List[JournalEntry]:
        """Синхронное получение записей по типу события."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    cursor = conn.execute(
                        """
                        SELECT * FROM journal_entries
                        WHERE event_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ? OFFSET ?
                    """,
                        (event_type, limit, offset),
                    )

                    rows = cursor.fetchall()

                    entries = []
                    for row in rows:
                        entry = JournalEntry(
                            id=row["id"],
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            event_type=row["event_type"],
                            message=row["message"],
                            data=json.loads(row["data"]) if row["data"] else None,
                            severity=row["severity"],
                            source=row["source"],
                            session_id=row["session_id"],
                        )
                        entries.append(entry)

                    return entries

        except Exception as e:
            logger.error(f"Ошибка синхронного получения записей по типу: {e}")
            return []

    async def clear_journal(self, older_than_days: Optional[int] = None) -> None:
        """Промышленная очистка журнала с поддержкой ротации по времени."""
        try:
            loop = asyncio.get_event_loop()
            deleted_count = await loop.run_in_executor(
                None, self._clear_journal_sync, older_than_days
            )

            logger.info(f"Очистка журнала завершена: удалено {deleted_count} записей")

        except Exception as e:
            logger.error(f"Ошибка очистки журнала: {e}")

    def _clear_journal_sync(self, older_than_days: Optional[int] = None) -> int:
        """Синхронная очистка журнала."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    if older_than_days:
                        cutoff_date = datetime.now() - timedelta(days=older_than_days)
                        cursor = conn.execute(
                            """
                            DELETE FROM journal_entries
                            WHERE timestamp < ?
                        """,
                            (cutoff_date.isoformat(),),
                        )
                    else:
                        cursor = conn.execute("DELETE FROM journal_entries")

                    deleted_count = cursor.rowcount
                    conn.commit()

                    return deleted_count

        except Exception as e:
            logger.error(f"Ошибка синхронной очистки журнала: {e}")
            return 0

    async def get_journal_statistics(self) -> Dict[str, Any]:
        """Промышленная статистика журнала с детальным анализом."""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(None, self._get_statistics_sync)

            logger.info("Статистика журнала получена")
            return stats

        except Exception as e:
            logger.error(f"Ошибка получения статистики журнала: {e}")
            return {}

    def _get_statistics_sync(self) -> Dict[str, Any]:
        """Синхронное получение статистики журнала."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    # Общая статистика
                    total_entries = conn.execute(
                        "SELECT COUNT(*) as count FROM journal_entries"
                    ).fetchone()["count"]

                    # Статистика по типам событий
                    event_types = conn.execute(
                        """
                        SELECT event_type, COUNT(*) as count
                        FROM journal_entries
                        GROUP BY event_type
                        ORDER BY count DESC
                    """
                    ).fetchall()

                    # Статистика по уровням важности
                    severities = conn.execute(
                        """
                        SELECT severity, COUNT(*) as count
                        FROM journal_entries
                        GROUP BY severity
                        ORDER BY count DESC
                    """
                    ).fetchall()

                    # Статистика по источникам
                    sources = conn.execute(
                        """
                        SELECT source, COUNT(*) as count
                        FROM journal_entries
                        GROUP BY source
                        ORDER BY count DESC
                    """
                    ).fetchall()

                    # Статистика по времени
                    time_stats = conn.execute(
                        """
                        SELECT
                            MIN(timestamp) as earliest,
                            MAX(timestamp) as latest,
                            COUNT(*) as total
                        FROM journal_entries
                    """
                    ).fetchone()

                    # Статистика по сессиям
                    session_stats = conn.execute(
                        """
                        SELECT
                            COUNT(DISTINCT session_id) as unique_sessions,
                            COUNT(*) as total_entries
                        FROM journal_entries
                        WHERE session_id IS NOT NULL
                    """
                    ).fetchone()

                    return {
                        "total_entries": total_entries,
                        "event_types": [dict(row) for row in event_types],
                        "severities": [dict(row) for row in severities],
                        "sources": [dict(row) for row in sources],
                        "time_range": {
                            "earliest": time_stats["earliest"],
                            "latest": time_stats["latest"],
                            "total": time_stats["total"],
                        },
                        "sessions": {
                            "unique_sessions": session_stats["unique_sessions"],
                            "total_entries": session_stats["total_entries"],
                        },
                        "database_size": (
                            self.db_path.stat().st_size if self.db_path.exists() else 0
                        ),
                    }

        except Exception as e:
            logger.error(f"Ошибка синхронного получения статистики: {e}")
            return {}

    async def export_journal(
        self, format: str = "json", file_path: Optional[str] = None
    ) -> str:
        """Экспорт журнала в различные форматы."""
        try:
            entries = await self.get_journal_entries(
                limit=10000
            )  # Большой лимит для экспорта

            if not file_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"logs/journal_export_{timestamp}.{format}"

            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                import csv

                with open(export_path, "w", newline="", encoding="utf-8") as f:
                    if entries:
                        writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                        writer.writeheader()
                        writer.writerows(entries)

            logger.info(f"Журнал экспортирован в {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Ошибка экспорта журнала: {e}")
            return ""

    async def search_journal(
        self, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Поиск по журналу с поддержкой полнотекстового поиска."""
        try:
            loop = asyncio.get_event_loop()
            entries = await loop.run_in_executor(
                None, self._search_journal_sync, query, limit
            )

            # Преобразование в словари
            result = []
            for entry in entries:
                entry_dict = asdict(entry)
                entry_dict["timestamp"] = entry.timestamp.isoformat()
                # entry.data уже имеет правильный тип Optional[Dict[str, Any]]
                result.append(entry_dict)

            logger.info(
                f"Поиск завершён: найдено {len(result)} записей для запроса '{query}'"
            )
            return result

        except Exception as e:
            logger.error(f"Ошибка поиска по журналу: {e}")
            return []

    def _search_journal_sync(self, query: str, limit: int) -> List[JournalEntry]:
        """Синхронный поиск по журналу."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    cursor = conn.execute(
                        """
                        SELECT * FROM journal_entries
                        WHERE message LIKE ? OR event_type LIKE ? OR source LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (f"%{query}%", f"%{query}%", f"%{query}%", limit),
                    )

                    rows = cursor.fetchall()

                    entries = []
                    for row in rows:
                        entry = JournalEntry(
                            id=row["id"],
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            event_type=row["event_type"],
                            message=row["message"],
                            data=json.loads(row["data"]) if row["data"] else None,
                            severity=row["severity"],
                            source=row["source"],
                            session_id=row["session_id"],
                        )
                        entries.append(entry)

                    return entries

        except Exception as e:
            logger.error(f"Ошибка синхронного поиска: {e}")
            return []
