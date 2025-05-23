import asyncio
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

DATASETS_DIR = Path("datasets")


@dataclass
class DatasetConfig:
    """Конфигурация менеджера датасетов"""

    data_dir: str = "data"
    cache_size: int = 1000
    max_history_days: int = 365
    min_samples: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    compression: bool = True
    backup_enabled: bool = True
    backup_interval: int = 3600  # секунд
    max_backups: int = 10
    feature_columns: List[str] = None
    target_columns: List[str] = None
    time_column: str = "timestamp"


@dataclass
class DatasetMetrics:
    """Метрики датасета"""

    total_samples: int
    feature_stats: Dict[str, Dict[str, float]]
    target_stats: Dict[str, Dict[str, float]]
    missing_values: Dict[str, int]
    last_update: datetime
    memory_usage: int
    compression_ratio: float


class DatasetManager:
    """Управление датасетами"""

    def __init__(self, config: Optional[DatasetConfig] = None):
        """Инициализация менеджера датасетов"""
        self.config = config or DatasetConfig()
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Пути к файлам
        self.datasets_path = self.data_dir / "datasets"
        self.datasets_path.mkdir(exist_ok=True)
        self.metrics_path = self.data_dir / "metrics.json"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Кэш
        self._dataset_cache = {}
        self._metrics_cache = {}

        # Метрики
        self.metrics = self._load_metrics()

        # Задачи
        self.backup_task = None

        # Запуск задач
        self._start_tasks()

    def _start_tasks(self):
        """Запуск фоновых задач"""
        if self.config.backup_enabled:
            loop = asyncio.get_event_loop()
            self.backup_task = loop.create_task(self._periodic_backup())

    async def _periodic_backup(self):
        """Периодическое резервное копирование"""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self._create_backup()
            except Exception as e:
                logger.error(f"Ошибка резервного копирования: {e}")

    async def _create_backup(self):
        """Создание резервной копии"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)

            # Копирование файлов
            for file in self.datasets_path.glob("*.parquet"):
                await self._copy_file(file, backup_path / file.name)

            # Очистка старых бэкапов
            backups = sorted(self.backup_dir.glob("backup_*"))
            if len(backups) > self.config.max_backups:
                for old_backup in backups[: -self.config.max_backups]:
                    await self._remove_dir(old_backup)

            logger.info(f"Создана резервная копия: {backup_path}")

        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")

    async def _copy_file(self, src: Path, dst: Path):
        """Асинхронное копирование файла"""
        try:
            async with aiofiles.open(src, "rb") as fsrc:
                async with aiofiles.open(dst, "wb") as fdst:
                    await fdst.write(await fsrc.read())
        except Exception as e:
            logger.error(f"Ошибка копирования файла {src}: {e}")
            raise

    async def _remove_dir(self, path: Path):
        """Асинхронное удаление директории"""
        try:
            for file in path.glob("**/*"):
                if file.is_file():
                    file.unlink()
            path.rmdir()
        except Exception as e:
            logger.error(f"Ошибка удаления директории {path}: {e}")
            raise

    @lru_cache(maxsize=100)
    def _load_metrics(self) -> Dict:
        """Загрузка метрик с кэшированием"""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return self._create_default_metrics()
        except Exception as e:
            logger.error(f"Ошибка загрузки метрик: {e}")
            return self._create_default_metrics()

    def _create_default_metrics(self) -> Dict:
        """Создание метрик по умолчанию"""
        return {
            "total_samples": 0,
            "feature_stats": {},
            "target_stats": {},
            "missing_values": {},
            "last_update": datetime.now().isoformat(),
            "memory_usage": 0,
            "compression_ratio": 1.0,
        }

    async def _save_metrics(self):
        """Сохранение метрик"""
        try:
            async with aiofiles.open(self.metrics_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.metrics, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")

    def _update_metrics(self, df: pd.DataFrame):
        """Обновление метрик"""
        try:
            self.metrics["total_samples"] = len(df)
            self.metrics["last_update"] = datetime.now().isoformat()
            self.metrics["memory_usage"] = df.memory_usage(deep=True).sum()

            # Статистика признаков
            for col in self.config.feature_columns or df.columns:
                if col in df.columns:
                    self.metrics["feature_stats"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "missing": int(df[col].isna().sum()),
                    }

            # Статистика целевых переменных
            for col in self.config.target_columns or []:
                if col in df.columns:
                    self.metrics["target_stats"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "missing": int(df[col].isna().sum()),
                    }

            # Пропущенные значения
            self.metrics["missing_values"] = {
                col: int(df[col].isna().sum()) for col in df.columns
            }

            # Коэффициент сжатия
            if self.config.compression:
                original_size = df.memory_usage(deep=True).sum()
                compressed_size = df.to_parquet(compression="gzip").nbytes
                self.metrics["compression_ratio"] = original_size / compressed_size

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    async def save_dataset(self, name: str, df: pd.DataFrame):
        """Сохранение датасета"""
        try:
            # Проверка данных
            if len(df) < self.config.min_samples:
                raise ValueError(
                    f"Недостаточно данных: {len(df)} < {self.config.min_samples}"
                )

            # Обработка временной колонки
            if self.config.time_column in df.columns:
                df[self.config.time_column] = pd.to_datetime(
                    df[self.config.time_column]
                )

            # Обновление метрик
            self._update_metrics(df)

            # Сохранение
            file_path = self.datasets_path / f"{name}.parquet"
            df.to_parquet(
                file_path,
                compression="gzip" if self.config.compression else None,
                index=False,
            )

            # Кэширование
            self._dataset_cache[name] = df

            # Сохранение метрик
            await self._save_metrics()

            logger.info(f"Датасет {name} сохранен: {len(df)} строк")

        except Exception as e:
            logger.error(f"Ошибка сохранения датасета {name}: {e}")
            raise

    @lru_cache(maxsize=100)
    def load_dataset(self, name: str) -> pd.DataFrame:
        """Загрузка датасета с кэшированием"""
        try:
            # Проверка кэша
            if name in self._dataset_cache:
                return self._dataset_cache[name]

            # Загрузка из файла
            file_path = self.datasets_path / f"{name}.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"Датасет {name} не найден")

            df = pd.read_parquet(file_path)

            # Кэширование
            self._dataset_cache[name] = df

            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки датасета {name}: {e}")
            raise

    def get_dataset_info(self, name: str) -> Dict:
        """Получение информации о датасете"""
        try:
            df = self.load_dataset(name)
            return {
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "missing_values": df.isna().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "last_update": self.metrics["last_update"],
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о датасете {name}: {e}")
            return {}

    def split_dataset(
        self, name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Разделение датасета на обучающую, валидационную и тестовую выборки"""
        try:
            df = self.load_dataset(name)

            # Сортировка по времени
            if self.config.time_column in df.columns:
                df = df.sort_values(self.config.time_column)

            # Разделение
            test_size = int(len(df) * self.config.test_split)
            val_size = int(len(df) * self.config.validation_split)

            test_df = df.iloc[-test_size:]
            val_df = df.iloc[-(test_size + val_size) : -test_size]
            train_df = df.iloc[: -(test_size + val_size)]

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Ошибка разделения датасета {name}: {e}")
            raise

    def get_metrics(self) -> Dict:
        """Получение метрик"""
        return self.metrics.copy()

    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.backup_task:
                self.backup_task.cancel()
            await self._save_metrics()
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")

    @staticmethod
    def _get_pair_path(pair: str) -> Path:
        DATASETS_DIR.mkdir(exist_ok=True)
        return DATASETS_DIR / f"{pair.upper()}.jsonl"

    @classmethod
    def save_backtest_example(cls, data: Dict):
        """Сохраняет пример в файл по паре с дедупликацией и фильтрацией."""
        pair = data.get("trading_pair", "UNKNOWN").upper()
        path = cls._get_pair_path(pair)
        # Загрузка существующих
        examples = cls.load_dataset(pair)
        # Фильтрация и дедупликация
        if not cls._is_informative(data):
            return
        if any(cls._is_duplicate(data, ex) for ex in examples):
            return
        examples.append(data)
        # Балансировка
        examples = cls._balance_examples(examples)
        # Сохраняем
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    @classmethod
    def load_dataset(cls, pair: str) -> List[Dict]:
        path = cls._get_pair_path(pair)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @classmethod
    def get_statistics(cls, pair: str) -> Dict:
        data = cls.load_dataset(pair)
        if not data:
            return {}
        total = len(data)
        win = sum(1 for x in data if x.get("result", {}).get("win"))
        loss = total - win
        avg_pnl = sum(x.get("result", {}).get("PnL", 0) for x in data) / total
        avg_drawdown = (
            sum(x.get("result", {}).get("drawdown", 0) or 0 for x in data) / total
        )
        return {
            "total": total,
            "win": win,
            "loss": loss,
            "avg_pnl": avg_pnl,
            "avg_drawdown": avg_drawdown,
        }

    @classmethod
    def clear_invalid_data(cls, pair: str):
        data = cls.load_dataset(pair)
        filtered = [x for x in data if cls._is_informative(x)]
        path = cls._get_pair_path(pair)
        with open(path, "w", encoding="utf-8") as f:
            for ex in filtered:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    @staticmethod
    def _is_duplicate(a: Dict, b: Dict) -> bool:
        # Сравниваем по ключевым признакам
        keys = ["features", "entry_price", "exit_price", "result"]
        return all(a.get(k) == b.get(k) for k in keys)

    @staticmethod
    def _is_informative(x: Dict) -> bool:
        # Фильтруем примеры с низкой волатильностью, маленьким drawdown, пустыми признаками
        features = x.get("features", {})
        result = x.get("result", {})
        if not features or features.get("volatility") is None:
            return False
        if abs(result.get("drawdown", 0) or 0) < 0.001:
            return False
        if abs(features.get("volatility", 0)) < 1e-6:
            return False
        return True

    @staticmethod
    def _balance_examples(examples: List[Dict]) -> List[Dict]:
        # Балансировка win/loss
        win = [x for x in examples if x.get("result", {}).get("win")]
        loss = [x for x in examples if not x.get("result", {}).get("win")]
        n = min(len(win), len(loss))
        return win[:n] + loss[:n]

    @classmethod
    def cli(cls):
        import argparse

        parser = argparse.ArgumentParser(description="DatasetManager CLI")
        parser.add_argument(
            "action",
            choices=["purge", "export", "stats", "clear_invalid"],
            help="Действие",
        )
        parser.add_argument("pair", nargs="?", help="Пара (например, BTCUSDT)")
        args = parser.parse_args()
        if args.action == "purge":
            if args.pair:
                path = cls._get_pair_path(args.pair)
                if path.exists():
                    path.unlink()
                    print(f"Удалено: {path}")
            else:
                for f in DATASETS_DIR.glob("*.jsonl"):
                    f.unlink()
                print("Удалены все датасеты.")
        elif args.action == "export":
            if args.pair:
                data = cls.load_dataset(args.pair)
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                for f in DATASETS_DIR.glob("*.jsonl"):
                    print(f"=== {f.name} ===")
                    data = [json.loads(line) for line in f.open("r", encoding="utf-8")]
                    print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == "stats":
            if args.pair:
                print(cls.get_statistics(args.pair))
            else:
                for f in DATASETS_DIR.glob("*.jsonl"):
                    pair = f.stem
                    print(f"{pair}: {cls.get_statistics(pair)}")
        elif args.action == "clear_invalid":
            if args.pair:
                cls.clear_invalid_data(args.pair)
                print(f"Очищены неинформативные примеры для {args.pair}")
            else:
                for f in DATASETS_DIR.glob("*.jsonl"):
                    cls.clear_invalid_data(f.stem)
                print("Очищены неинформативные примеры для всех пар.")


if __name__ == "__main__":
    DatasetManager.cli()
