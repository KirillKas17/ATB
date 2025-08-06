"""
Система автоматического развёртывания для ATB.

Обеспечивает автоматическое развёртывание, проверку здоровья,
откат изменений и мониторинг процесса развёртывания.
"""

import os
import sys
import subprocess
import time
import json
import yaml
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable, cast
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger


class DeploymentStatus(Enum):
    """Статусы развёртывания."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"
    HEALTH_CHECK = "health_check"


class HealthStatus(Enum):
    """Статусы здоровья системы."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Конфигурация развёртывания."""
    
    # Основные настройки
    app_name: str
    version: str
    environment: str
    
    # Настройки развёртывания
    deployment_type: str = "rolling"  # rolling, blue-green, canary
    max_parallel_deployments: int = 1
    health_check_timeout: int = 300  # секунды
    rollback_threshold: int = 3  # количество неудачных проверок здоровья
    
    # Настройки сервисов
    services: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Настройки мониторинга
    health_check_endpoints: List[str] = field(default_factory=list)
    metrics_endpoints: List[str] = field(default_factory=list)
    
    # Настройки отката
    auto_rollback: bool = True
    rollback_timeout: int = 600  # секунды
    
    # Настройки уведомлений
    notification_webhooks: List[str] = field(default_factory=list)
    notification_email: Optional[str] = None


@dataclass
class DeploymentStep:
    """Шаг развёртывания."""
    
    name: str
    command: str
    timeout: int = 300
    retries: int = 3
    critical: bool = True
    rollback_command: Optional[str] = None
    health_check: Optional[str] = None


@dataclass
class DeploymentResult:
    """Результат развёртывания."""
    
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_performed: bool = False
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)


class HealthChecker:
    """Проверка здоровья системы."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.session = requests.Session()
        
    def check_endpoint_health(self, endpoint: str) -> Dict[str, Any]:
        """
        Проверить здоровье endpoint.
        
        Args:
            endpoint: URL для проверки
            
        Returns:
            Результат проверки
        """
        try:
            response = self.session.get(endpoint)
            return {
                "endpoint": endpoint,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def check_system_health(self) -> HealthStatus:
        """
        Проверить общее здоровье системы.
        
        Returns:
            Статус здоровья
        """
        if not self.config.health_check_endpoints:
            return HealthStatus.UNKNOWN
            
        results = []
        for endpoint in self.config.health_check_endpoints:
            result = self.check_endpoint_health(endpoint)
            results.append(result)
            
        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        total_count = len(results)
        
        if healthy_count == total_count:
            return HealthStatus.HEALTHY
        elif healthy_count > total_count * 0.5:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
            
    def wait_for_health(self, timeout: Optional[int] = None) -> bool:
        """
        Ожидать восстановления здоровья системы.
        
        Args:
            timeout: Таймаут ожидания
            
        Returns:
            True если система здорова
        """
        timeout = timeout or self.config.health_check_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_system_health()
            if status == HealthStatus.HEALTHY:
                return True
            elif status == HealthStatus.UNHEALTHY:
                logger.warning("System is unhealthy, waiting for recovery...")
            else:
                logger.info("System is degraded, monitoring...")
                
            time.sleep(10)
            
        return False


class DeploymentManager:
    """
    Менеджер развёртывания.
    
    Управляет процессом развёртывания, проверкой здоровья
    и откатом изменений при необходимости.
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_checker = HealthChecker(config)
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        self.lock = threading.Lock()
        
    def deploy(self, deployment_id: str, steps: List[DeploymentStep]) -> DeploymentResult:
        """
        Выполнить развёртывание.
        
        Args:
            deployment_id: ID развёртывания
            steps: Шаги развёртывания
            
        Returns:
            Результат развёртывания
        """
        with self.lock:
            if len(self.active_deployments) >= self.config.max_parallel_deployments:
                raise RuntimeError("Maximum parallel deployments reached")
                
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
            
            self.active_deployments[deployment_id] = result
            
        try:
            logger.info(f"Starting deployment {deployment_id}")
            self._notify_deployment_start(deployment_id)
            
            # Проверка здоровья перед развёртыванием
            if not self._pre_deployment_health_check():
                raise RuntimeError("System is unhealthy before deployment")
                
            # Выполнение шагов развёртывания
            for step in steps:
                if not self._execute_step(step, result):
                    raise RuntimeError(f"Step {step.name} failed")
                    
            # Проверка здоровья после развёртывания
            if not self._post_deployment_health_check(result):
                if self.config.auto_rollback:
                    logger.warning("Health check failed, performing rollback")
                    self._perform_rollback(steps, result)
                else:
                    raise RuntimeError("Health check failed after deployment")
                    
            result.status = DeploymentStatus.SUCCESS
            result.end_time = datetime.now()
            logger.info(f"Deployment {deployment_id} completed successfully")
            self._notify_deployment_success(deployment_id)
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")
            self._notify_deployment_failure(deployment_id, str(e))
            
            # Автоматический откат при критических ошибках
            if self.config.auto_rollback:
                self._perform_rollback(steps, result)
                
        finally:
            with self.lock:
                self.deployment_history.append(result)
                if deployment_id in self.active_deployments:
                    del self.active_deployments[deployment_id]
                    
        return result
        
    def _pre_deployment_health_check(self) -> bool:
        """Проверка здоровья перед развёртыванием."""
        logger.info("Performing pre-deployment health check")
        status = self.health_checker.check_system_health()
        return status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
    def _post_deployment_health_check(self, result: DeploymentResult) -> bool:
        """Проверка здоровья после развёртывания."""
        logger.info("Performing post-deployment health check")
        
        # Ждём восстановления здоровья
        if not self.health_checker.wait_for_health():
            return False
            
        # Выполняем дополнительные проверки
        for endpoint in self.config.health_check_endpoints:
            health_result = self.health_checker.check_endpoint_health(endpoint)
            result.health_checks.append(health_result)
            
            if health_result["status"] != "healthy":
                logger.warning(f"Health check failed for {endpoint}")
                return False
                
        return True
        
    def _execute_step(self, step: DeploymentStep, result: DeploymentResult) -> bool:
        """
        Выполнить шаг развёртывания.
        
        Args:
            step: Шаг для выполнения
            result: Результат развёртывания
            
        Returns:
            True если шаг выполнен успешно
        """
        logger.info(f"Executing step: {step.name}")
        
        for attempt in range(step.retries):
            try:
                # Выполнение команды
                process = subprocess.run(
                    step.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout
                )
                
                if process.returncode == 0:
                    result.steps_completed.append(step.name)
                    result.logs.append(f"Step {step.name} completed successfully")
                    logger.info(f"Step {step.name} completed successfully")
                    return True
                else:
                    error_msg = f"Step {step.name} failed (attempt {attempt + 1}/{step.retries})"
                    result.logs.append(f"{error_msg}: {process.stderr}")
                    logger.warning(error_msg)
                    
            except subprocess.TimeoutExpired:
                error_msg = f"Step {step.name} timed out (attempt {attempt + 1}/{step.retries})"
                result.logs.append(error_msg)
                logger.warning(error_msg)
            except Exception as e:
                error_msg = f"Step {step.name} error (attempt {attempt + 1}/{step.retries}): {e}"
                result.logs.append(error_msg)
                logger.error(error_msg)
                
            if attempt < step.retries - 1:
                time.sleep(5)  # Пауза между попытками
                
        # Все попытки исчерпаны
        result.steps_failed.append(step.name)
        if step.critical:
            return False
        else:
            logger.warning(f"Non-critical step {step.name} failed, continuing...")
            return True
            
    def _perform_rollback(self, steps: List[DeploymentStep], result: DeploymentResult):
        """Выполнить откат развёртывания."""
        logger.info("Performing deployment rollback")
        result.status = DeploymentStatus.ROLLBACK
        
        # Выполняем команды отката в обратном порядке
        for step in reversed(steps):
            if step.rollback_command:
                try:
                    logger.info(f"Executing rollback for step: {step.name}")
                    process = subprocess.run(
                        step.rollback_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=step.timeout
                    )
                    
                    if process.returncode == 0:
                        result.logs.append(f"Rollback for {step.name} completed successfully")
                    else:
                        result.logs.append(f"Rollback for {step.name} failed: {process.stderr}")
                        
                except Exception as e:
                    result.logs.append(f"Rollback for {step.name} error: {e}")
                    
        result.rollback_performed = True
        result.end_time = datetime.now()
        
        # Проверяем здоровье после отката
        if self.health_checker.wait_for_health(timeout=300):
            result.status = DeploymentStatus.SUCCESS
            logger.info("Rollback completed successfully")
        else:
            result.status = DeploymentStatus.FAILED
            logger.error("Rollback failed - system still unhealthy")
            
    def _notify_deployment_start(self, deployment_id: str):
        """Уведомить о начале развёртывания."""
        message = {
            "type": "deployment_start",
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__
        }
        self._send_notifications(message)
        
    def _notify_deployment_success(self, deployment_id: str):
        """Уведомить об успешном развёртывании."""
        message = {
            "type": "deployment_success",
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat()
        }
        self._send_notifications(message)
        
    def _notify_deployment_failure(self, deployment_id: str, error: str):
        """Уведомить о неудачном развёртывании."""
        message = {
            "type": "deployment_failure",
            "deployment_id": deployment_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self._send_notifications(message)
        
    def _send_notifications(self, message: Dict[str, Any]):
        """Отправить уведомления."""
        # Webhook уведомления
        for webhook in self.config.notification_webhooks:
            try:
                requests.post(webhook, json=message, timeout=10)
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")
                
        # Email уведомления (упрощённая реализация)
        if self.config.notification_email:
            logger.info(f"Email notification would be sent to {self.config.notification_email}")
            
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Получить статус развёртывания."""
        with self.lock:
            if deployment_id in self.active_deployments:
                return self.active_deployments[deployment_id]
                
            for result in self.deployment_history:
                if result.deployment_id == deployment_id:
                    return result
                    
        return None
        
    def get_active_deployments(self) -> List[DeploymentResult]:
        """Получить активные развёртывания."""
        with self.lock:
            return list(self.active_deployments.values())
            
    def get_deployment_history(self, limit: int = 10) -> List[DeploymentResult]:
        """Получить историю развёртываний."""
        with self.lock:
            return self.deployment_history[-limit:]
            
    def cancel_deployment(self, deployment_id: str) -> bool:
        """Отменить развёртывание."""
        with self.lock:
            if deployment_id not in self.active_deployments:
                return False
                
            result = self.active_deployments[deployment_id]
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = "Deployment cancelled by user"
            
            return True


class DeploymentOrchestrator:
    """
    Оркестратор развёртываний.
    
    Управляет множественными развёртываниями и обеспечивает
    координацию между различными сервисами.
    """
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config()
        self.managers: Dict[str, DeploymentManager] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузить конфигурацию."""
        with open(self.config_file, 'r') as f:
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                return cast(Dict[str, Any], yaml.safe_load(f))
            else:
                return cast(Dict[str, Any], json.load(f))
                
    def create_deployment_manager(self, service_name: str) -> DeploymentManager:
        """Создать менеджер развёртывания для сервиса."""
        if service_name not in self.config['services']:
            raise ValueError(f"Service {service_name} not found in configuration")
            
        service_config = self.config['services'][service_name]
        deployment_config = DeploymentConfig(
            app_name=service_name,
            version=service_config.get('version', 'latest'),
            environment=self.config.get('environment', 'production'),
            deployment_type=service_config.get('deployment_type', 'rolling'),
            health_check_endpoints=service_config.get('health_check_endpoints', []),
            auto_rollback=service_config.get('auto_rollback', True),
            notification_webhooks=self.config.get('notification_webhooks', [])
        )
        
        return DeploymentManager(deployment_config)
        
    def deploy_service(self, service_name: str, version: str, 
                      deployment_steps: List[DeploymentStep]) -> str:
        """
        Развернуть сервис.
        
        Args:
            service_name: Название сервиса
            version: Версия для развёртывания
            deployment_steps: Шаги развёртывания
            
        Returns:
            ID развёртывания
        """
        deployment_id = f"{service_name}-{version}-{int(time.time())}"
        
        if service_name not in self.managers:
            self.managers[service_name] = self.create_deployment_manager(service_name)
            
        manager = self.managers[service_name]
        
        # Запуск развёртывания в отдельном потоке
        future = self.executor.submit(manager.deploy, deployment_id, deployment_steps)
        
        return deployment_id
        
    def deploy_all_services(self, version: str) -> Dict[str, str]:
        """
        Развернуть все сервисы.
        
        Args:
            version: Версия для развёртывания
            
        Returns:
            Словарь с ID развёртываний
        """
        deployment_ids = {}
        
        for service_name in self.config['services']:
            try:
                # Создаём базовые шаги развёртывания
                steps = self._create_default_steps(service_name, version)
                deployment_id = self.deploy_service(service_name, version, steps)
                deployment_ids[service_name] = deployment_id
                
            except Exception as e:
                logger.error(f"Failed to deploy service {service_name}: {e}")
                
        return deployment_ids
        
    def _create_default_steps(self, service_name: str, version: str) -> List[DeploymentStep]:
        """Создать стандартные шаги развёртывания."""
        service_config = self.config['services'][service_name]
        
        steps = [
            DeploymentStep(
                name="backup",
                command=f"backup_service {service_name}",
                timeout=300,
                critical=False
            ),
            DeploymentStep(
                name="deploy",
                command=f"deploy_service {service_name} {version}",
                timeout=600,
                critical=True,
                rollback_command=f"rollback_service {service_name} {version}"
            ),
            DeploymentStep(
                name="health_check",
                command=f"check_service_health {service_name}",
                timeout=300,
                critical=True
            )
        ]
        
        return steps
        
    def get_deployment_status(self, service_name: str, deployment_id: str) -> Optional[DeploymentResult]:
        """Получить статус развёртывания."""
        if service_name in self.managers:
            return self.managers[service_name].get_deployment_status(deployment_id)
        return None
        
    def get_all_deployment_statuses(self) -> Dict[str, List[DeploymentResult]]:
        """Получить статусы всех развёртываний."""
        statuses = {}
        for service_name, manager in self.managers.items():
            statuses[service_name] = manager.get_deployment_history()
        return statuses
        
    def shutdown(self):
        """Остановка оркестратора."""
        self.executor.shutdown(wait=True)


def create_deployment_config(app_name: str, version: str, environment: str = "production") -> DeploymentConfig:
    """
    Создать конфигурацию развёртывания.
    
    Args:
        app_name: Название приложения
        version: Версия
        environment: Окружение
        
    Returns:
        Конфигурация развёртывания
    """
    return DeploymentConfig(
        app_name=app_name,
        version=version,
        environment=environment,
        health_check_endpoints=[
            f"http://localhost:8080/api/health",
            f"http://localhost:8080/api/status"
        ],
        notification_webhooks=[
            "http://localhost:8080/api/notifications"
        ]
    )


def main():
    """Основная функция для запуска развёртывания."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ATB Deployment System")
    parser.add_argument("--config", required=True, help="Path to deployment config file")
    parser.add_argument("--service", help="Service to deploy")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--all", action="store_true", help="Deploy all services")
    
    args = parser.parse_args()
    
    try:
        orchestrator = DeploymentOrchestrator(args.config)
        
        if args.all:
            deployment_ids = orchestrator.deploy_all_services(args.version)
            print(f"Deployed all services: {deployment_ids}")
        elif args.service:
            steps = [
                DeploymentStep(
                    name="deploy",
                    command=f"echo 'Deploying {args.service} {args.version}'",
                    timeout=60
                )
            ]
            deployment_id = orchestrator.deploy_service(args.service, args.version, steps)
            print(f"Deployment started: {deployment_id}")
        else:
            print("Please specify --service or --all")
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 