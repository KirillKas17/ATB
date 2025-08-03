"""
Unit тесты для RateLimiter
"""
import asyncio
import time
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
class TestRateLimiter:
    """Тесты для RateLimiter."""
    @pytest.fixture
    def rate_limiter(self) -> Any:
        """Экземпляр RateLimiter."""
        return RateLimiter(rate_limit=10, window=60)
    @pytest.fixture
    def fast_rate_limiter(self) -> Any:
        """RateLimiter с быстрым окном для тестов."""
        return RateLimiter(rate_limit=3, window=1)  # 3 запроса в секунду
    def test_init(self) -> None:
        """Тест инициализации."""
        rate_limiter = RateLimiter(rate_limit=100, window=60)
        assert rate_limiter.rate_limit == 100
        assert rate_limiter.window == 60
        assert rate_limiter.requests == []
        assert rate_limiter.lock is not None
    def test_init_default_window(self) -> None:
        """Тест инициализации с окном по умолчанию."""
        rate_limiter = RateLimiter(rate_limit=50)
        assert rate_limiter.rate_limit == 50
        assert rate_limiter.window == 60  # По умолчанию
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self, rate_limiter) -> None:
        """Тест получения разрешения в пределах лимита."""
        # Выполняем несколько запросов в пределах лимита
        start_time = time.time()
        for _ in range(5):
            await rate_limiter.acquire()
        end_time = time.time()
        # Запросы должны выполняться быстро
        assert end_time - start_time < 1.0
        assert len(rate_limiter.requests) == 5
    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self, fast_rate_limiter) -> None:
        """Тест превышения лимита запросов."""
        # Выполняем запросы до лимита
        for _ in range(3):
            await fast_rate_limiter.acquire()
        # Следующий запрос должен задержаться
        start_time = time.time()
        await fast_rate_limiter.acquire()
        end_time = time.time()
        # Должна быть задержка
        assert end_time - start_time >= 0.9  # Почти секунда
        assert len(fast_rate_limiter.requests) == 4
    @pytest.mark.asyncio
    async def test_acquire_old_requests_cleanup(self, fast_rate_limiter) -> None:
        """Тест очистки старых запросов."""
        # Выполняем запросы
        for _ in range(3):
            await fast_rate_limiter.acquire()
        # Ждем, пока окно истечет
        await asyncio.sleep(1.1)
        # Выполняем еще один запрос
        await fast_rate_limiter.acquire()
        # Старые запросы должны быть очищены
        assert len(fast_rate_limiter.requests) == 1
    @pytest.mark.asyncio
    async def test_get_remaining_requests_empty(self, rate_limiter) -> None:
        """Тест получения оставшихся запросов (пустой список)."""
        remaining = await rate_limiter.get_remaining_requests()
        assert remaining == 10  # Полный лимит
    @pytest.mark.asyncio
    async def test_get_remaining_requests_partial(self, rate_limiter) -> None:
        """Тест получения оставшихся запросов (частично использовано)."""
        # Выполняем несколько запросов
        for _ in range(3):
            await rate_limiter.acquire()
        remaining = await rate_limiter.get_remaining_requests()
        assert remaining == 7  # 10 - 3
    @pytest.mark.asyncio
    async def test_get_remaining_requests_full(self, rate_limiter) -> None:
        """Тест получения оставшихся запросов (лимит исчерпан)."""
        # Выполняем максимальное количество запросов
        for _ in range(10):
            await rate_limiter.acquire()
        remaining = await rate_limiter.get_remaining_requests()
        assert remaining == 0
    @pytest.mark.asyncio
    async def test_get_remaining_requests_with_cleanup(self, fast_rate_limiter) -> None:
        """Тест получения оставшихся запросов с очисткой."""
        # Выполняем запросы
        for _ in range(3):
            await fast_rate_limiter.acquire()
        # Ждем, пока окно истечет
        await asyncio.sleep(1.1)
        remaining = await fast_rate_limiter.get_remaining_requests()
        # После очистки должен быть полный лимит
        assert remaining == 3
    @pytest.mark.asyncio
    async def test_get_reset_time_empty(self, rate_limiter) -> None:
        """Тест получения времени сброса (пустой список)."""
        reset_time = await rate_limiter.get_reset_time()
        assert reset_time == 0.0
    @pytest.mark.asyncio
    async def test_get_reset_time_with_requests(self, fast_rate_limiter) -> None:
        """Тест получения времени сброса с запросами."""
        # Выполняем запрос
        await fast_rate_limiter.acquire()
        reset_time = await fast_rate_limiter.get_reset_time()
        # Время сброса должно быть положительным и меньше окна
        assert 0.0 < reset_time <= 1.0
    @pytest.mark.asyncio
    async def test_get_reset_time_after_window(self, fast_rate_limiter) -> None:
        """Тест получения времени сброса после истечения окна."""
        # Выполняем запрос
        await fast_rate_limiter.acquire()
        # Ждем, пока окно истечет
        await asyncio.sleep(1.1)
        reset_time = await fast_rate_limiter.get_reset_time()
        # Время сброса должно быть 0
        assert reset_time == 0.0
    @pytest.mark.asyncio
    async def test_reset(self, rate_limiter) -> None:
        """Тест сброса лимита."""
        # Выполняем несколько запросов
        for _ in range(5):
            await rate_limiter.acquire()
        assert len(rate_limiter.requests) == 5
        # Сбрасываем лимит
        await rate_limiter.reset()
        assert len(rate_limiter.requests) == 0
        # Проверяем, что можно снова делать запросы
        remaining = await rate_limiter.get_remaining_requests()
        assert remaining == 10
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter) -> None:
        """Тест конкурентных запросов."""
        async def make_request() -> Any:
            await rate_limiter.acquire()
            return True
        # Выполняем несколько запросов одновременно
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert all(results)
        assert len(rate_limiter.requests) == 5
    @pytest.mark.asyncio
    async def test_concurrent_requests_exceed_limit(self, fast_rate_limiter) -> None:
        """Тест конкурентных запросов с превышением лимита."""
        async def make_request() -> Any:
            await fast_rate_limiter.acquire()
            return True
        # Выполняем больше запросов, чем разрешено
        tasks = [make_request() for _ in range(5)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        assert all(results)
        assert len(fast_rate_limiter.requests) == 5
        # Должна быть задержка из-за превышения лимита
        assert end_time - start_time >= 1.0
    @pytest.mark.asyncio
    async def test_rate_limiter_thread_safety(self, rate_limiter) -> None:
        """Тест потокобезопасности."""
        async def concurrent_operations() -> Any:
            await rate_limiter.acquire()
            remaining = await rate_limiter.get_remaining_requests()
            reset_time = await rate_limiter.get_reset_time()
            return remaining, reset_time
        # Выполняем конкурентные операции
        tasks = [concurrent_operations() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        # Все операции должны завершиться без ошибок
        assert len(results) == 10
        assert all(isinstance(result, tuple) for result in results)
    @pytest.mark.asyncio
    async def test_rate_limiter_edge_cases(self) -> None:
        """Тест граничных случаев."""
        # RateLimiter с очень маленьким лимитом
        tiny_limiter = RateLimiter(rate_limit=1, window=0.1)
        await tiny_limiter.acquire()
        # Следующий запрос должен задержаться
        start_time = time.time()
        await tiny_limiter.acquire()
        end_time = time.time()
        assert end_time - start_time >= 0.09  # Почти 0.1 секунды
    @pytest.mark.asyncio
    async def test_rate_limiter_zero_limit(self) -> None:
        """Тест RateLimiter с нулевым лимитом."""
        zero_limiter = RateLimiter(rate_limit=0, window=1)
        # Любой запрос должен задержаться
        start_time = time.time()
        await zero_limiter.acquire()
        end_time = time.time()
        assert end_time - start_time >= 0.9  # Почти секунда
    @pytest.mark.asyncio
    async def test_rate_limiter_large_window(self) -> None:
        """Тест RateLimiter с большим окном."""
        large_window_limiter = RateLimiter(rate_limit=5, window=3600)  # 1 час
        # Выполняем запросы
        for _ in range(5):
            await large_window_limiter.acquire()
        remaining = await large_window_limiter.get_remaining_requests()
        assert remaining == 0
        reset_time = await large_window_limiter.get_reset_time()
        assert reset_time > 0
    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup_accuracy(self, fast_rate_limiter) -> None:
        """Тест точности очистки старых запросов."""
        # Выполняем запросы с интервалом
        for i in range(3):
            await fast_rate_limiter.acquire()
            if i < 2:  # Не ждем после последнего запроса
                await asyncio.sleep(0.1)
        # Проверяем количество запросов до очистки
        assert len(fast_rate_limiter.requests) == 3
        # Ждем, пока окно истечет
        await asyncio.sleep(1.1)
        # Проверяем количество запросов после очистки
        remaining = await fast_rate_limiter.get_remaining_requests()
        assert remaining == 3  # Все запросы должны быть очищены 
