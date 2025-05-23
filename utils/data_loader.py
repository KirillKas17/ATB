import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import ccxt.async_support as ccxt
import pandas as pd
import yfinance as yf
from loguru import logger


class DataSource(Enum):
    BYBIT = "bybit"
    YFINANCE = "yfinance"
    CACHE = "cache"
    FILE = "file"


@dataclass
class DataQualityMetrics:
    missing_values: int
    duplicates: int
    gaps: int
    quality_score: float
    outliers: int
    data_consistency: float
    timestamp_continuity: float
    volume_consistency: float
    price_consistency: float


class DataLoader:
    def __init__(
        self,
        cache_dir: str = "cache",
        use_testnet: bool = True,
        max_workers: int = 4,
        cache_ttl: int = 3600,
        retry_attempts: int = 3,
        retry_delay: int = 1,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_testnet = use_testnet
        self.exchange = self._init_exchange()
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.cache_ttl = cache_ttl
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_lock = asyncio.Lock()
        self._exchange_lock = asyncio.Lock()

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection with advanced error handling"""
        try:
            exchange = ccxt.bybit(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "future", "testnet": self.use_testnet},
                    "timeout": 30000,
                    "enableLastJsonNumbers": True,
                    "recvWindow": 60000,
                }
            )
            return exchange
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {str(e)}")
            raise

    async def _fetch_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                async with self._exchange_lock:
                    return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def _fetch_ohlcv_bybit(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Bybit with advanced error handling"""
        try:
            ohlcv = await self._fetch_with_retry(
                self.exchange.fetch_ohlcv, symbol, timeframe, since, limit
            )
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Validate data
            df = self._validate_ohlcv_data(df)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from Bybit: {str(e)}")
            return pd.DataFrame()

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        if df.empty:
            return df

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Validate price relationships
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ]

        # Remove outliers
        for col in ["open", "high", "low", "close"]:
            mean = df[col].mean()
            std = df[col].std()
            df = df[abs(df[col] - mean) <= 3 * std]

        # Validate volume
        df = df[df["volume"] >= 0]

        return df

    async def _save_to_cache(
        self, data: pd.DataFrame, symbol: str, timeframe: str
    ) -> None:
        """Save data to cache with metadata"""
        async with self._cache_lock:
            cache_file = self.cache_dir / f"{symbol}_{timeframe}.pkl"
            metadata = {
                "timestamp": datetime.now().timestamp(),
                "rows": len(data),
                "columns": list(data.columns),
                "quality_metrics": self.check_data_quality(data),
            }

            cache_data = {"data": data, "metadata": metadata}

            async with aiofiles.open(cache_file, "wb") as f:
                await f.write(pickle.dumps(cache_data))

    async def _load_from_cache(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache with validation"""
        async with self._cache_lock:
            cache_file = self.cache_dir / f"{symbol}_{timeframe}.pkl"
            if not cache_file.exists():
                return None

            try:
                async with aiofiles.open(cache_file, "rb") as f:
                    cache_data = pickle.loads(await f.read())

                # Check cache TTL
                if (
                    datetime.now().timestamp() - cache_data["metadata"]["timestamp"]
                    > self.cache_ttl
                ):
                    return None

                return cache_data["data"]
            except Exception as e:
                self.logger.error(f"Error loading from cache: {str(e)}")
                return None

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get OHLCV data with advanced caching and validation"""
        # Try cache first
        if use_cache:
            cached_data = await self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                if start_time and end_time:
                    mask = (cached_data.index >= start_time) & (
                        cached_data.index <= end_time
                    )
                    return cached_data[mask]
                return cached_data

        # Try multiple sources
        data = pd.DataFrame()
        sources = [self._fetch_ohlcv_bybit, self._fetch_yfinance]

        for source in sources:
            try:
                data = await source(symbol, timeframe, start_time, end_time)
                if not data.empty:
                    break
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source.__name__}: {str(e)}")

        if not data.empty and use_cache:
            await self._save_to_cache(data, symbol, timeframe)

        return data

    async def _fetch_yfinance(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch data from yfinance with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_time, end=end_time, interval=timeframe)
            return self._validate_ohlcv_data(data)
        except Exception as e:
            self.logger.error(f"Failed to fetch from yfinance: {str(e)}")
            return pd.DataFrame()

    def check_data_quality(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Check comprehensive data quality metrics"""
        if data.empty:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Basic metrics
        missing = data.isnull().sum().sum()
        duplicates = data.index.duplicated().sum()

        # Check for gaps
        expected_freq = pd.infer_freq(data.index)
        if expected_freq:
            gaps = len(
                pd.date_range(data.index[0], data.index[-1], freq=expected_freq)
            ) - len(data)
        else:
            gaps = 0

        # Detect outliers
        outliers = 0
        for col in ["open", "high", "low", "close"]:
            mean = data[col].mean()
            std = data[col].std()
            outliers += len(data[abs(data[col] - mean) > 3 * std])

        # Check data consistency
        price_consistency = self._check_price_consistency(data)
        volume_consistency = self._check_volume_consistency(data)
        timestamp_continuity = self._check_timestamp_continuity(data)

        # Calculate overall quality score
        total_points = len(data) * len(data.columns)
        quality_score = 100 * (
            1 - (missing + duplicates + gaps + outliers) / total_points
        )

        return DataQualityMetrics(
            missing_values=int(missing),
            duplicates=int(duplicates),
            gaps=int(gaps),
            quality_score=float(quality_score),
            outliers=int(outliers),
            data_consistency=float(price_consistency * volume_consistency),
            timestamp_continuity=float(timestamp_continuity),
            volume_consistency=float(volume_consistency),
            price_consistency=float(price_consistency),
        )

    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """Check price data consistency"""
        if data.empty:
            return 0.0

        valid_rows = len(
            data[
                (data["high"] >= data["low"])
                & (data["high"] >= data["open"])
                & (data["high"] >= data["close"])
                & (data["low"] <= data["open"])
                & (data["low"] <= data["close"])
            ]
        )

        return valid_rows / len(data)

    def _check_volume_consistency(self, data: pd.DataFrame) -> float:
        """Check volume data consistency"""
        if data.empty:
            return 0.0

        valid_rows = len(data[data["volume"] >= 0])
        return valid_rows / len(data)

    def _check_timestamp_continuity(self, data: pd.DataFrame) -> float:
        """Check timestamp continuity"""
        if data.empty:
            return 0.0

        expected_freq = pd.infer_freq(data.index)
        if not expected_freq:
            return 0.0

        expected_timestamps = pd.date_range(
            data.index[0], data.index[-1], freq=expected_freq
        )
        actual_timestamps = data.index

        return len(set(actual_timestamps) & set(expected_timestamps)) / len(
            expected_timestamps
        )

    async def close(self):
        """Close all connections and cleanup"""
        await self.exchange.close()
        self._executor.shutdown(wait=True)

    @staticmethod
    def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data with advanced aggregation"""
        if data.empty:
            return data

        resampled = data.resample(timeframe).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Add additional metrics
        resampled["vwap"] = (data["close"] * data["volume"]).resample(
            timeframe
        ).sum() / data["volume"].resample(timeframe).sum()
        resampled["trades"] = data["volume"].resample(timeframe).count()

        return resampled


async def load_market_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Загрузка рыночных данных с кэшированием"""
    loader = DataLoader()
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        data = await loader.get_ohlcv(
            symbol=symbol,
            timeframe=interval,
            start_time=datetime.strptime(start_date, "%Y-%m-%d"),
            end_time=datetime.strptime(end_date, "%Y-%m-%d"),
            use_cache=use_cache,
        )

        return data
    finally:
        await loader.close()


async def save_market_data(
    data: pd.DataFrame, symbol: str, directory: str = "data"
) -> bool:
    """Сохранение рыночных данных с валидацией"""
    try:
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Validate data before saving
        loader = DataLoader()
        quality_metrics = loader.check_data_quality(data)

        if quality_metrics.quality_score < 80:
            logger.warning(f"Low quality data for {symbol}: {quality_metrics}")

        filename = directory / f"{symbol}_{datetime.now().strftime('%Y%m%d')}.pkl"
        async with aiofiles.open(filename, "wb") as f:
            await f.write(
                pickle.dumps(
                    {
                        "data": data,
                        "metadata": {
                            "timestamp": datetime.now().timestamp(),
                            "quality_metrics": quality_metrics,
                        },
                    }
                )
            )
        return True
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {str(e)}")
        return False


async def load_saved_data(
    symbol: str, directory: str = "data"
) -> Optional[pd.DataFrame]:
    """Загрузка сохраненных данных с валидацией"""
    try:
        directory = Path(directory)
        files = list(directory.glob(f"{symbol}_*.pkl"))
        if not files:
            return None

        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        async with aiofiles.open(latest_file, "rb") as f:
            saved_data = pickle.loads(await f.read())

        # Validate loaded data
        loader = DataLoader()
        quality_metrics = loader.check_data_quality(saved_data["data"])

        if quality_metrics.quality_score < 80:
            logger.warning(f"Low quality data loaded for {symbol}: {quality_metrics}")

        return saved_data["data"]
    except Exception as e:
        logger.error(f"Error loading saved data for {symbol}: {str(e)}")
        return None


async def get_market_metadata(symbol: str) -> Dict[str, Any]:
    """Получение расширенных метаданных инструмента"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get additional data
        loader = DataLoader()
        data = await loader.get_ohlcv(symbol, "1d", limit=100)
        quality_metrics = loader.check_data_quality(data)

        return {
            "symbol": symbol,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "currency": info.get("currency", ""),
            "market_cap": info.get("marketCap", 0),
            "volume_avg": info.get("averageVolume", 0),
            "data_quality": quality_metrics.__dict__,
            "last_update": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {symbol}: {str(e)}")
        return {}


async def load_dynamic_window_data(
    symbol: str,
    timeframe: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    min_candles: int = 100,
    max_candles: int = 1000,
    volatility_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Загрузка данных с динамическим окном в зависимости от волатильности.

    Args:
        symbol: Торговая пара
        timeframe: Временной фрейм
        start_time: Начальное время
        end_time: Конечное время
        min_candles: Минимальное количество свечей
        max_candles: Максимальное количество свечей
        volatility_threshold: Порог волатильности для определения размера окна

    Returns:
        DataFrame с OHLCV данными
    """
    try:
        # Загружаем базовые данные
        data = await load_market_data(symbol, timeframe, start_time, end_time)

        if len(data) < min_candles:
            logger.warning(
                f"Недостаточно данных для {symbol}: {len(data)} < {min_candles}"
            )
            return data

        # Рассчитываем волатильность
        returns = data["close"].pct_change()
        volatility = returns.std()

        # Определяем размер окна
        if volatility > volatility_threshold:
            window_size = min_candles
        else:
            window_size = min(
                max_candles, int(min_candles * (volatility_threshold / volatility))
            )

        # Обрезаем данные до нужного размера
        if len(data) > window_size:
            data = data.iloc[-window_size:]

        return data

    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {str(e)}")
        raise
