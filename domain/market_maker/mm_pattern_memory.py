"""
Модуль для работы с памятью паттернов маркет-мейкера.
"""

import asyncio
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from shared.numpy_utils import np

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternMemory,
    PatternOutcome,
    PatternResult,
)
from domain.types.market_maker_types import PatternMemoryDict


class IPatternMemoryRepository(Protocol):
    """Интерфейс репозитория памяти паттернов"""

    async def save_pattern(self, symbol: str, pattern: MarketMakerPattern) -> bool: ...
    async def update_pattern_result(
        self, symbol: str, pattern_id: str, result: PatternResult
    ) -> bool: ...
    async def get_patterns_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> List[PatternMemory]: ...
    async def get_successful_patterns(
        self, symbol: str, min_accuracy: float = 0.7
    ) -> List[PatternMemory]: ...
    async def find_similar_patterns(
        self, symbol: str, features: Dict[str, Any], similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]: ...
    async def get_success_map(self, symbol: str) -> Dict[str, float]: ...
    async def update_success_map(
        self, symbol: str, pattern_type: str, success_rate: float
    ) -> bool: ...
    async def save_behavior_history(
        self, symbol: str, behavior_data: Dict[str, Any]
    ) -> bool: ...
    async def get_behavior_history(
        self, symbol: str, days: int = 30
    ) -> List[Dict[str, Any]]: ...
    async def cleanup_old_data(self, symbol: str, days: int = 30) -> int: ...
    async def get_storage_statistics(self) -> Dict[str, Any]: ...
@dataclass
class MatchedPattern:
    pattern_memory: PatternMemory
    similarity_score: float
    confidence_boost: float
    signal_strength: float
    metadata: Dict[str, Any]


class PatternMemoryRepository:
    """Реализация репозитория памяти паттернов маркет-мейкера"""

    def __init__(self, base_path: str = "market_profiles"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.pattern_cache: Dict[str, List[PatternMemory]] = {}
        self.success_map_cache: Dict[str, Dict[str, float]] = {}
        self.behavior_history_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.storage_stats = {
            "total_patterns": 0,
            "total_symbols": 0,
            "total_successful_patterns": 0,
            "last_cleanup": datetime.now(),
        }
        self._create_directory_structure()
        self._init_databases()

    def _create_directory_structure(self) -> None:
        try:
            profiles_dir = self.base_path / "market_profiles"
            profiles_dir.mkdir(exist_ok=True)
        except Exception as e:
            print(f"Error creating directory structure: {e}")

    def _init_databases(self) -> None:
        try:
            main_db_path = self.base_path / "mm_patterns_metadata.db"
            with sqlite3.connect(main_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS symbols (
                        symbol TEXT PRIMARY KEY,
                        first_pattern_date TEXT,
                        last_pattern_date TEXT,
                        total_patterns INTEGER DEFAULT 0,
                        successful_patterns INTEGER DEFAULT 0,
                        avg_accuracy REAL DEFAULT 0.0,
                        avg_return REAL DEFAULT 0.0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pattern_types (
                        symbol TEXT,
                        pattern_type TEXT,
                        count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        avg_return REAL DEFAULT 0.0,
                        last_seen TEXT,
                        PRIMARY KEY (symbol, pattern_type)
                    )
                """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_symbols_last_date ON symbols(last_pattern_date)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_pattern_types_symbol ON pattern_types(symbol)"
                )
        except Exception as e:
            print(f"Error initializing databases: {e}")

    def get_symbol_directory(self, symbol: str) -> Path:
        symbol_dir = self.base_path / "market_profiles" / symbol / "mm_patterns"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir

    async def save_pattern(self, symbol: str, pattern: MarketMakerPattern) -> bool:
        try:
            symbol_dir = self.get_symbol_directory(symbol)
            pattern_file = symbol_dir / "pattern_memory.jsonl"
            pattern_data = pattern.to_dict()
            pattern_id = self._generate_pattern_id(symbol, pattern)
            pattern_data["pattern_id"] = pattern_id
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._write_pattern_to_file, pattern_file, pattern_data
            )
            if symbol in self.pattern_cache:
                del self.pattern_cache[symbol]
            return True
        except Exception as e:
            print(f"Error saving pattern for {symbol}: {e}")
            return False

    def _write_pattern_to_file(
        self, pattern_file: Path, pattern_data: Dict[str, Any]
    ) -> None:
        with open(pattern_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pattern_data) + "\n")

    def _read_patterns_from_file(self, pattern_file: Path) -> List[Dict[str, Any]]:
        patterns = []
        if pattern_file.exists():
            with open(pattern_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        patterns.append(json.loads(line))
        return patterns

    def _generate_pattern_id(self, symbol: str, pattern: MarketMakerPattern) -> str:
        pattern_str = (
            f"{symbol}_{pattern.pattern_type.value}_{pattern.timestamp.isoformat()}"
        )
        return hashlib.md5(pattern_str.encode()).hexdigest()

    async def get_patterns_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> List[PatternMemory]:
        try:
            if symbol in self.pattern_cache:
                return self.pattern_cache[symbol][:limit]
            symbol_dir = self.get_symbol_directory(symbol)
            pattern_file = symbol_dir / "pattern_memory.jsonl"
            loop = asyncio.get_event_loop()
            patterns_data = await loop.run_in_executor(
                self.executor, self._read_patterns_from_file, pattern_file
            )
            patterns = []
            for pattern_data in patterns_data[:limit]:
                pattern_memory = self._data_to_pattern_memory(pattern_data)
                if pattern_memory:
                    patterns.append(pattern_memory)
            self.pattern_cache[symbol] = patterns
            return patterns
        except Exception as e:
            print(f"Error getting patterns for {symbol}: {e}")
            return []

    async def get_successful_patterns(
        self, symbol: str, min_accuracy: float = 0.7
    ) -> List[PatternMemory]:
        try:
            patterns = await self.get_patterns_by_symbol(symbol)
            successful_patterns = [p for p in patterns if p.accuracy >= min_accuracy]
            successful_patterns.sort(key=lambda x: x.accuracy, reverse=True)
            return successful_patterns
        except Exception as e:
            print(f"Error getting successful patterns for {symbol}: {e}")
            return []

    async def find_similar_patterns(
        self, symbol: str, features: Dict[str, Any], similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        try:
            patterns = await self.get_patterns_by_symbol(symbol)
            similar_patterns = []
            for pattern in patterns:
                pattern_features = self._extract_pattern_features(pattern)
                similarity = self._calculate_similarity(features, pattern_features)
                if similarity >= similarity_threshold:
                    matched_pattern = {
                        "pattern_memory": pattern,
                        "similarity_score": similarity,
                        "confidence_boost": self._calculate_confidence_boost(
                            similarity, pattern
                        ),
                        "signal_strength": self._calculate_signal_strength(pattern),
                        "metadata": {
                            "pattern_type": (
                                pattern.pattern.pattern_type.value
                                if hasattr(pattern.pattern.pattern_type, "value")
                                else pattern.pattern.pattern_type
                            ),
                            "timestamp": pattern.pattern.timestamp.isoformat(),
                            "accuracy": pattern.accuracy,
                            "avg_return": pattern.avg_return,
                        },
                    }
                    similar_patterns.append(matched_pattern)
            similar_patterns.sort(
                key=lambda x: float(str(x["similarity_score"])), reverse=True
            )
            return similar_patterns
        except Exception as e:
            print(f"Error finding similar patterns for {symbol}: {e}")
            return []

    def _data_to_pattern_memory(
        self, pattern_data: Dict[str, Any]
    ) -> Optional[PatternMemory]:
        try:
            pattern_memory_data: PatternMemoryDict = {
                "pattern": pattern_data["pattern"],
                "result": pattern_data.get("result"),
                "accuracy": pattern_data.get("accuracy", 0.0),
                "avg_return": pattern_data.get("avg_return", 0.0),
                "success_count": pattern_data.get("success_count", 0),
                "total_count": pattern_data.get("total_count", 0),
                "last_seen": pattern_data.get("last_seen"),
            }
            return PatternMemory.from_dict(pattern_memory_data)
        except Exception as e:
            print(f"Error converting data to PatternMemory: {e}")
            return None

    def _extract_pattern_features(self, pattern: PatternMemory) -> Dict[str, Any]:
        f = pattern.pattern.features
        return {
            "book_pressure": float(f.book_pressure),
            "volume_delta": float(f.volume_delta),
            "price_reaction": float(f.price_reaction),
            "spread_change": float(f.spread_change),
            "order_imbalance": float(f.order_imbalance),
            "liquidity_depth": float(f.liquidity_depth),
            "time_duration": int(f.time_duration),
            "volume_concentration": float(f.volume_concentration),
            "price_volatility": float(f.price_volatility),
        }

    def _calculate_similarity(
        self, features1: Dict[str, Any], features2: Dict[str, Any]
    ) -> float:
        try:
            numeric_features = [
                "book_pressure",
                "volume_delta",
                "price_reaction",
                "spread_change",
                "order_imbalance",
                "liquidity_depth",
                "volume_concentration",
                "price_volatility",
            ]
            total_distance = 0.0
            feature_count = 0
            for feature in numeric_features:
                if feature in features1 and feature in features2:
                    val1 = float(features1[feature])
                    val2 = float(features2[feature])
                    distance = (val1 - val2) ** 2
                    total_distance += distance
                    feature_count += 1
            if feature_count == 0:
                return 0.0
            avg_distance = total_distance / feature_count
            similarity = max(0.0, 1.0 - avg_distance)
            return min(1.0, similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def _calculate_confidence_boost(
        self, similarity: float, pattern: PatternMemory
    ) -> float:
        base_boost = similarity * 0.5
        accuracy_boost = pattern.accuracy * 0.3
        success_boost = min(0.2, pattern.success_count * 0.01)
        return min(1.0, base_boost + accuracy_boost + success_boost)

    def _calculate_signal_strength(self, pattern: PatternMemory) -> float:
        base_strength = abs(pattern.avg_return)
        accuracy_modifier = pattern.accuracy
        count_modifier = min(1.0, pattern.success_count / 10.0)
        return base_strength * accuracy_modifier * count_modifier

    async def update_pattern_result(
        self, symbol: str, pattern_id: str, result: PatternResult
    ) -> bool:
        try:
            symbol_dir = self.get_symbol_directory(symbol)
            pattern_file = symbol_dir / "pattern_memory.jsonl"
            loop = asyncio.get_event_loop()
            patterns = await loop.run_in_executor(
                self.executor, self._read_patterns_from_file, pattern_file
            )
            pattern_found = False
            for i, pattern_data in enumerate(patterns):
                if pattern_data.get("pattern_id") == pattern_id:
                    pattern_data["result"] = result.to_dict()
                    if "statistics" not in pattern_data:
                        pattern_data["statistics"] = {
                            "accuracy": 0.0,
                            "avg_return": 0.0,
                            "success_count": 0,
                            "total_count": 0,
                        }
                    stats = pattern_data["statistics"]
                    stats["total_count"] += 1
                    if result.outcome == PatternOutcome.SUCCESS:
                        stats["success_count"] += 1
                    stats["accuracy"] = stats["success_count"] / stats["total_count"]
                    stats["avg_return"] = (
                        stats["avg_return"] * (stats["total_count"] - 1)
                        + result.price_change_15min
                    ) / stats["total_count"]
                    pattern_found = True
                    break
            if pattern_found:
                await loop.run_in_executor(
                    self.executor,
                    self._write_patterns_to_file,
                    pattern_file,
                    patterns,
                )
                if symbol in self.pattern_cache:
                    del self.pattern_cache[symbol]
                return True
            return False
        except Exception as e:
            print(f"Error updating pattern result for {symbol}: {e}")
            return False

    def _write_patterns_to_file(
        self, pattern_file: Path, patterns: List[Dict[str, Any]]
    ) -> None:
        with open(pattern_file, "w", encoding="utf-8") as f:
            for pattern in patterns:
                f.write(json.dumps(pattern) + "\n")

    async def get_success_map(self, symbol: str) -> Dict[str, float]:
        try:
            if symbol in self.success_map_cache:
                return self.success_map_cache[symbol]
            symbol_dir = self.get_symbol_directory(symbol)
            success_map_file = symbol_dir / "success_map.json"
            if success_map_file.exists():
                loop = asyncio.get_event_loop()
                success_map = await loop.run_in_executor(
                    self.executor, self._read_json_file, success_map_file
                )
            else:
                success_map = {}
            # Явное приведение типа для mypy
            typed_success_map: Dict[str, float] = {}
            for key, value in success_map.items():
                if isinstance(value, (int, float)):
                    typed_success_map[str(key)] = float(value)
            self.success_map_cache[symbol] = typed_success_map
            return typed_success_map
        except Exception as e:
            print(f"Error getting success map for {symbol}: {e}")
            return {}

    def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}

    async def update_success_map(
        self, symbol: str, pattern_type: str, success_rate: float
    ) -> bool:
        try:
            success_map = await self.get_success_map(symbol)
            success_map[pattern_type] = success_rate
            symbol_dir = self.get_symbol_directory(symbol)
            success_map_file = symbol_dir / "success_map.json"
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._write_json_file, success_map_file, success_map
            )
            self.success_map_cache[symbol] = success_map
            return True
        except Exception as e:
            print(f"Error updating success map for {symbol}: {e}")
            return False

    def _write_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def save_behavior_history(
        self, symbol: str, behavior_data: Dict[str, Any]
    ) -> bool:
        try:
            symbol_dir = self.get_symbol_directory(symbol)
            behavior_db = symbol_dir / "behavior_history.db"
            behavior_data["timestamp"] = datetime.now().isoformat()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self._save_behavior_to_db, behavior_db, behavior_data
            )
            if symbol in self.behavior_history_cache:
                del self.behavior_history_cache[symbol]
            return True
        except Exception as e:
            print(f"Error saving behavior history for {symbol}: {e}")
            return False

    def _save_behavior_to_db(
        self, behavior_db: Path, behavior_data: Dict[str, Any]
    ) -> None:
        with sqlite3.connect(behavior_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    behavior_type TEXT,
                    data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                INSERT INTO behavior_history (timestamp, behavior_type, data)
                VALUES (?, ?, ?)
            """,
                (
                    behavior_data["timestamp"],
                    behavior_data.get("behavior_type", "unknown"),
                    json.dumps(behavior_data),
                ),
            )

    async def get_behavior_history(
        self, symbol: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        try:
            if symbol in self.behavior_history_cache:
                return self.behavior_history_cache[symbol]
            symbol_dir = self.get_symbol_directory(symbol)
            behavior_db = symbol_dir / "behavior_history.db"
            if not behavior_db.exists():
                return []
            start_date = datetime.now() - timedelta(days=days)
            loop = asyncio.get_event_loop()
            history = await loop.run_in_executor(
                self.executor, self._read_behavior_from_db, behavior_db, start_date
            )
            self.behavior_history_cache[symbol] = history
            return history
        except Exception as e:
            print(f"Error getting behavior history for {symbol}: {e}")
            return []

    def _read_behavior_from_db(
        self, behavior_db: Path, start_date: datetime
    ) -> List[Dict[str, Any]]:
        history = []
        with sqlite3.connect(behavior_db) as conn:
            cursor = conn.execute(
                """
                SELECT data FROM behavior_history 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """,
                (start_date.isoformat(),),
            )
            for row in cursor:
                try:
                    data = json.loads(row[0])
                    history.append(data)
                except json.JSONDecodeError:
                    continue
        return history

    async def cleanup_old_data(self, symbol: str, days: int = 30) -> int:
        try:
            cleaned_count = 0
            patterns = await self.get_patterns_by_symbol(symbol)
            cutoff_date = datetime.now() - timedelta(days=days)
            new_patterns = []
            for pattern in patterns:
                if pattern.pattern.timestamp >= cutoff_date:
                    new_patterns.append(pattern)
                else:
                    cleaned_count += 1
            if cleaned_count > 0:
                symbol_dir = self.get_symbol_directory(symbol)
                pattern_file = symbol_dir / "pattern_memory.jsonl"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self._write_patterns_to_file,
                    pattern_file,
                    [dict(pattern.to_dict()) for pattern in new_patterns],
                )
                if symbol in self.pattern_cache:
                    del self.pattern_cache[symbol]
            return cleaned_count
        except Exception as e:
            print(f"Error cleaning old data for {symbol}: {e}")
            return 0

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Получить статистику хранилища."""
        try:
            await self._update_storage_stats()
            main_db_path = Path(self.base_path) / "storage_stats.db"
            if main_db_path.exists():
                return self._read_storage_stats_from_db(main_db_path)
            else:
                return {
                    "total_patterns": 0,
                    "total_symbols": 0,
                    "storage_size_mb": 0.0,
                    "last_updated": datetime.now().isoformat(),
                }
        except Exception as e:
            print(f"Error getting storage statistics: {e}")
            return {
                "total_patterns": 0,
                "total_symbols": 0,
                "storage_size_mb": 0.0,
                "last_updated": datetime.now().isoformat(),
                "error": str(e),
            }

    async def _update_storage_stats(self) -> None:
        try:
            main_db_path = self.base_path / "mm_patterns_metadata.db"
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.executor, self._read_storage_stats_from_db, main_db_path
            )
            self.storage_stats.update(stats)
        except Exception as e:
            print(f"Error updating storage stats: {e}")

    def _read_storage_stats_from_db(self, main_db_path: Path) -> Dict[str, Any]:
        stats = {
            "total_patterns": 0,
            "total_symbols": 0,
            "total_successful_patterns": 0,
        }
        if main_db_path.exists():
            with sqlite3.connect(main_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM symbols")
                stats["total_symbols"] = cursor.fetchone()[0]
                cursor = conn.execute("SELECT SUM(total_patterns) FROM symbols")
                total_patterns = cursor.fetchone()[0]
                stats["total_patterns"] = total_patterns or 0
                cursor = conn.execute("SELECT SUM(successful_patterns) FROM symbols")
                successful_patterns = cursor.fetchone()[0]
                stats["total_successful_patterns"] = successful_patterns or 0
        return stats

    def __del__(self) -> None:
        # Реализация финализатора, если требуется
        pass
