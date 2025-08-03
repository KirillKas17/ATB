"""AB Test Manager."""

import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ABTestManager:
    """�������� A/B ������������ � ������������ �����������."""

    def __init__(self, storage_path: Path = Path("./ab_tests")):
        self.storage_path = storage_path
        self._tests: Dict[str, Dict[str, Any]] = {}
        self._tests_lock = threading.RLock()

        logger.info("ABTestManager ���������������")

    async def create_test(
        self,
        test_name: str,
        control_variant: Dict[str, Any],
        treatment_variant: Dict[str, Any],
        metrics: List[str],
        sample_size: int = 1000,
    ) -> str:
        """�������� ������ A/B �����."""
        with self._tests_lock:
            test_id = (
                f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._tests)}"
            )

            test_config = {
                "id": test_id,
                "name": test_name,
                "control_variant": control_variant,
                "treatment_variant": treatment_variant,
                "metrics": metrics,
                "sample_size": sample_size,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "control_data": [],
                "treatment_data": [],
            }

            self._tests[test_id] = test_config

            logger.info(f"������ A/B ���� {test_id}: {test_name}")
            return test_id

    async def assign_variant(self, test_id: str, user_id: str) -> str:
        """���������� �������� ������������."""
        with self._tests_lock:
            if test_id not in self._tests:
                raise ValueError(f"���� {test_id} �� ������")

            variant = random.choice(["control", "treatment"])
            logger.debug(f"������������ {user_id} �������� � ������ {variant}")
            return variant

    async def record_event(
        self,
        test_id: str,
        user_id: str,
        variant: str,
        event_name: str,
        event_value: float,
    ) -> bool:
        """������ ������� ��� A/B �����."""
        with self._tests_lock:
            if test_id not in self._tests:
                return False

            event_data = {
                "user_id": user_id,
                "variant": variant,
                "event_name": event_name,
                "event_value": event_value,
                "timestamp": datetime.now().isoformat(),
            }

            if variant == "control":
                self._tests[test_id]["control_data"].append(event_data)
            else:
                self._tests[test_id]["treatment_data"].append(event_data)

            return True

    async def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """��������� ����������� �����."""
        with self._tests_lock:
            if test_id not in self._tests:
                return None

            test_config = self._tests[test_id]
            control_data = test_config["control_data"]
            treatment_data = test_config["treatment_data"]

            if not control_data or not treatment_data:
                return {"test_id": test_id, "status": "insufficient_data"}

            control_values = [event["event_value"] for event in control_data]
            treatment_values = [event["event_value"] for event in treatment_data]

            control_mean = sum(control_values) / len(control_values)
            treatment_mean = sum(treatment_values) / len(treatment_values)

            improvement = ((treatment_mean - control_mean) / control_mean) * 100

            return {
                "test_id": test_id,
                "test_name": test_config["name"],
                "status": "completed",
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "improvement_percent": improvement,
            }

    async def list_tests(self) -> List[Dict[str, Any]]:
        """��������� ������ ���� ������."""
        with self._tests_lock:
            return [
                {
                    "id": test_id,
                    "name": config["name"],
                    "status": config["status"],
                    "created_at": config["created_at"],
                }
                for test_id, config in self._tests.items()
            ]

    async def start(self) -> None:
        """������ ��������� A/B ������������."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("ABTestManager �������")

    async def stop(self) -> None:
        """��������� ��������� A/B ������������."""
        logger.info("ABTestManager ����������")
