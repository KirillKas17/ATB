"""
Хранилище стратегий для сохранения и извлечения эволюционных стратегий.
Промышленная реализация с полной типизацией, валидацией и обработкой ошибок.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

# Безопасный импорт sqlmodel
try:
    from sqlmodel import Session, SQLModel, create_engine, select
    SQLMODEL_AVAILABLE = True
except ImportError:
    # Простая замена для sqlmodel
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        
        def add(self, *args, **kwargs):
            pass
        
        def commit(self, *args, **kwargs):
            pass
        
        def close(self, *args, **kwargs):
            pass
    
    class SQLModel:
        pass
    
    def create_engine(*args, **kwargs):
        return None
    
    def select(*args, **kwargs):
        return None
    
    SQLMODEL_AVAILABLE = False
    logging.warning("Using mock sqlmodel implementation due to missing dependency")

from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate,
)
from infrastructure.evolution.exceptions import (
    ConnectionError,
    QueryError,
    StorageError,
    ValidationError,
)
from infrastructure.evolution.models import (
    EvolutionContextModel,
    StrategyCandidateModel,
    StrategyEvaluationModel,
)
from infrastructure.evolution.serializers import (
    candidate_to_model,
    context_to_model,
    evaluation_to_model,
    model_to_candidate,
    model_to_context,
    model_to_evaluation,
)
from infrastructure.evolution.types import (
    DatabasePath,
    EvolutionStorageProtocol,
    StorageStatistics,
)

logger = logging.getLogger(__name__)


class StrategyStorage(EvolutionStorageProtocol):
    """
    Инфраструктурное хранилище стратегий эволюции.
    Реализует все методы EvolutionStorageProtocol с полной типизацией,
    обработкой ошибок и валидацией данных.
    """

    def __init__(self, db_path: str = "evolution_strategies.db") -> None:
        """
        Инициализировать хранилище стратегий.
        Args:
            db_path: Путь к файлу базы данных
        Raises:
            ConnectionError: При ошибке подключения к БД
        """
        self.db_path: str = db_path
        try:
            self.engine = create_engine(f"sqlite:///{db_path}")
            self._create_tables()
            logger.info(f"Хранилище стратегий инициализировано: {db_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации хранилища: {e}")
            raise ConnectionError(f"Не удалось подключиться к БД: {e}", db_path)

    def _create_tables(self) -> None:
        """Создать таблицы в базе данных."""
        try:
            SQLModel.metadata.create_all(self.engine)
            logger.debug("Таблицы БД созданы успешно")
        except Exception as e:
            logger.error(f"Ошибка создания таблиц: {e}")
            raise QueryError(f"Не удалось создать таблицы: {e}", "CREATE TABLES")

    def save_strategy_candidate(self, candidate: StrategyCandidate) -> None:
        """
        Сохранить кандидата стратегии.
        Args:
            candidate: Кандидат стратегии для сохранения
        Raises:
            ValidationError: При невалидных данных
            StorageError: При ошибке сохранения
        """
        try:
            # Валидация входных данных
            self._validate_candidate(candidate)
            with Session(self.engine) as session:
                # Проверить, существует ли уже
                existing = session.exec(
                    select(StrategyCandidateModel).where(
                        StrategyCandidateModel.id == str(candidate.id)
                    )
                ).first()
                if existing:
                    # Обновить существующий
                    model = candidate_to_model(candidate)
                    for field, value in model.dict(exclude={"id"}).items():
                        setattr(existing, field, value)
                    existing.updated_at = datetime.now()
                else:
                    # Создать новый
                    model = candidate_to_model(candidate)
                    session.add(model)
                session.commit()
                logger.debug(f"Кандидат стратегии сохранен: {candidate.id}")
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка сохранения кандидата стратегии: {e}")
            raise StorageError(
                f"Не удалось сохранить кандидата стратегии: {e}", "save_error"
            )

    def get_strategy_candidate(self, candidate_id: UUID) -> Optional[StrategyCandidate]:
        """
        Получить кандидата стратегии по ID.
        Args:
            candidate_id: ID кандидата стратегии
        Returns:
            Кандидат стратегии или None, если не найден
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                model = session.exec(
                    select(StrategyCandidateModel).where(
                        StrategyCandidateModel.id == str(candidate_id)
                    )
                ).first()
                if not model:
                    logger.debug(f"Кандидат стратегии не найден: {candidate_id}")
                    return None
                candidate = model_to_candidate(model)
                logger.debug(f"Кандидат стратегии получен: {candidate_id}")
                return candidate
        except Exception as e:
            logger.error(f"Ошибка получения кандидата стратегии: {e}")
            raise StorageError(
                f"Не удалось получить кандидата стратегии: {e}", "get_error"
            )

    def get_strategy_candidates(
        self,
        status: Optional[EvolutionStatus] = None,
        generation: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[StrategyCandidate]:
        """
        Получить список кандидатов стратегий.
        Args:
            status: Фильтр по статусу
            generation: Фильтр по поколению
            limit: Ограничение количества результатов
        Returns:
            Список кандидатов стратегий
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                query = select(StrategyCandidateModel)
                if status:
                    query = query.where(StrategyCandidateModel.status == status.value)
                if generation is not None:
                    query = query.where(StrategyCandidateModel.generation == generation)
                if limit:
                    query = query.limit(limit)
                models = session.exec(query).all()
                candidates = [model_to_candidate(model) for model in models]
                logger.debug(f"Получено кандидатов стратегий: {len(candidates)}")
                return candidates
        except Exception as e:
            logger.error(f"Ошибка получения списка кандидатов стратегий: {e}")
            raise StorageError(
                f"Не удалось получить список кандидатов стратегий: {e}", "get_error"
            )

    def save_evaluation_result(self, evaluation: StrategyEvaluationResult) -> None:
        """
        Сохранить результат оценки.
        Args:
            evaluation: Результат оценки для сохранения
        Raises:
            ValidationError: При невалидных данных
            StorageError: При ошибке сохранения
        """
        try:
            # Валидация входных данных
            self._validate_evaluation(evaluation)
            with Session(self.engine) as session:
                # Проверить, существует ли уже
                existing = session.exec(
                    select(StrategyEvaluationModel).where(
                        StrategyEvaluationModel.id == str(evaluation.id)
                    )
                ).first()
                if existing:
                    # Обновить существующий
                    model = evaluation_to_model(evaluation)
                    for field, value in model.dict(exclude={"id"}).items():
                        setattr(existing, field, value)
                else:
                    # Создать новый
                    model = evaluation_to_model(evaluation)
                    session.add(model)
                session.commit()
                logger.debug(f"Результат оценки сохранен: {evaluation.id}")
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка сохранения результата оценки: {e}")
            raise StorageError(
                f"Не удалось сохранить результат оценки: {e}", "save_error"
            )

    def get_evaluation_result(
        self, evaluation_id: UUID
    ) -> Optional[StrategyEvaluationResult]:
        """
        Получить результат оценки по ID.
        Args:
            evaluation_id: ID результата оценки
        Returns:
            Результат оценки или None, если не найден
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                model = session.exec(
                    select(StrategyEvaluationModel).where(
                        StrategyEvaluationModel.id == str(evaluation_id)
                    )
                ).first()
                if not model:
                    logger.debug(f"Результат оценки не найден: {evaluation_id}")
                    return None
                evaluation = model_to_evaluation(model)
                logger.debug(f"Результат оценки получен: {evaluation_id}")
                return evaluation
        except Exception as e:
            logger.error(f"Ошибка получения результата оценки: {e}")
            raise StorageError(
                f"Не удалось получить результат оценки: {e}", "get_error"
            )

    def get_evaluation_results(
        self,
        strategy_id: Optional[UUID] = None,
        is_approved: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[StrategyEvaluationResult]:
        """
        Получить список результатов оценки.
        Args:
            strategy_id: Фильтр по ID стратегии
            is_approved: Фильтр по статусу одобрения
            limit: Ограничение количества результатов
        Returns:
            Список результатов оценки
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                query = select(StrategyEvaluationModel)
                if strategy_id:
                    query = query.where(
                        StrategyEvaluationModel.strategy_id == str(strategy_id)
                    )
                if is_approved is not None:
                    query = query.where(
                        StrategyEvaluationModel.is_approved == is_approved
                    )
                if limit:
                    query = query.limit(limit)
                models = session.exec(query).all()
                evaluations = [model_to_evaluation(model) for model in models]
                logger.debug(f"Получено результатов оценки: {len(evaluations)}")
                return evaluations
        except Exception as e:
            logger.error(f"Ошибка получения списка результатов оценки: {e}")
            raise StorageError(
                f"Не удалось получить список результатов оценки: {e}", "get_error"
            )

    def save_evolution_context(self, context: EvolutionContext) -> None:
        """
        Сохранить контекст эволюции.
        Args:
            context: Контекст эволюции для сохранения
        Raises:
            ValidationError: При невалидных данных
            StorageError: При ошибке сохранения
        """
        try:
            # Валидация входных данных
            self._validate_context(context)
            with Session(self.engine) as session:
                # Проверить, существует ли уже
                existing = session.exec(
                    select(EvolutionContextModel).where(
                        EvolutionContextModel.id == str(context.id)
                    )
                ).first()
                if existing:
                    # Обновить существующий
                    model = context_to_model(context)
                    for field, value in model.dict(exclude={"id"}).items():
                        setattr(existing, field, value)
                    existing.updated_at = datetime.now()
                else:
                    # Создать новый
                    model = context_to_model(context)
                    session.add(model)
                session.commit()
                logger.debug(f"Контекст эволюции сохранен: {context.id}")
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка сохранения контекста эволюции: {e}")
            raise StorageError(
                f"Не удалось сохранить контекст эволюции: {e}", "save_error"
            )

    def get_evolution_context(self, context_id: UUID) -> Optional[EvolutionContext]:
        """
        Получить контекст эволюции по ID.
        Args:
            context_id: ID контекста эволюции
        Returns:
            Контекст эволюции или None, если не найден
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                model = session.exec(
                    select(EvolutionContextModel).where(
                        EvolutionContextModel.id == str(context_id)
                    )
                ).first()
                if not model:
                    logger.debug(f"Контекст эволюции не найден: {context_id}")
                    return None
                context = model_to_context(model)
                logger.debug(f"Контекст эволюции получен: {context_id}")
                return context
        except Exception as e:
            logger.error(f"Ошибка получения контекста эволюции: {e}")
            raise StorageError(
                f"Не удалось получить контекст эволюции: {e}", "get_error"
            )

    def get_evolution_contexts(
        self, limit: Optional[int] = None
    ) -> List[EvolutionContext]:
        """
        Получить список контекстов эволюции.
        Args:
            limit: Ограничение количества результатов
        Returns:
            Список контекстов эволюции
        Raises:
            StorageError: При ошибке получения данных
        """
        try:
            with Session(self.engine) as session:
                query = select(EvolutionContextModel)
                if limit:
                    query = query.limit(limit)
                models = session.exec(query).all()
                contexts = [model_to_context(model) for model in models]
                logger.debug(f"Получено контекстов эволюции: {len(contexts)}")
                return contexts
        except Exception as e:
            logger.error(f"Ошибка получения списка контекстов эволюции: {e}")
            raise StorageError(
                f"Не удалось получить список контекстов эволюции: {e}", "get_error"
            )

    def get_statistics(self) -> StorageStatistics:
        """
        Получить статистику хранилища.
        Returns:
            Статистика хранилища
        Raises:
            StorageError: При ошибке получения статистики
        """
        try:
            with Session(self.engine) as session:
                # Подсчет кандидатов по статусам
                status_counts = {}
                for status in EvolutionStatus:
                    result = session.exec(
                        select(StrategyCandidateModel).where(
                            StrategyCandidateModel.status == status.value
                        )
                    )
                    count = len(result.all())
                    status_counts[status.value] = count
                # Подсчет оценок
                evaluations_result = session.exec(
                    select(StrategyEvaluationModel)
                )
                total_evaluations = len(evaluations_result.all())
                approved_result = session.exec(
                    select(StrategyEvaluationModel).where(
                        StrategyEvaluationModel.is_approved == True
                    )
                )
                approved_evaluations = len(approved_result.all())
                # Подсчет контекстов
                contexts_result = session.exec(select(EvolutionContextModel))
                total_contexts = len(contexts_result.all())
                # Расчет approval rate
                approval_rate = (
                    approved_evaluations / total_evaluations
                    if total_evaluations > 0
                    else 0.0
                )
                # Размер хранилища (приблизительно)
                storage_size_bytes = self._calculate_storage_size()
                # Время последнего бэкапа
                last_backup_time = self._get_last_backup_time()
                # Hit rate кэша
                cache_hit_rate = self._calculate_cache_hit_rate()
                # Среднее время запроса
                average_query_time = self._calculate_average_query_time()
                return StorageStatistics(
                    total_candidates=sum(status_counts.values()),
                    total_evaluations=total_evaluations,
                    total_contexts=total_contexts,
                    candidates_by_status=status_counts,
                    approval_rate=approval_rate,
                    storage_size_bytes=storage_size_bytes,
                    last_backup_time=last_backup_time.isoformat() if last_backup_time else None,  # Исправляю тип для TypedDict
                    cache_hit_rate=cache_hit_rate,
                    average_query_time=average_query_time,
                )
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise StorageError(
                f"Не удалось получить статистику: {e}", "statistics_error"
            )

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Очистить старые данные.
        Args:
            days_to_keep: Количество дней для хранения данных
        Returns:
            Количество удаленных записей
        Raises:
            StorageError: При ошибке очистки данных
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            with Session(self.engine) as session:
                # Удалить старые кандидаты
                old_candidates = session.exec(
                    select(StrategyCandidateModel).where(
                        StrategyCandidateModel.created_at < cutoff_date
                    )
                ).all()
                for candidate in old_candidates:
                    session.delete(candidate)
                    deleted_count += 1
                # Удалить старые оценки
                old_evaluations = session.exec(
                    select(StrategyEvaluationModel).where(
                        StrategyEvaluationModel.evaluation_time < cutoff_date
                    )
                ).all()
                for evaluation in old_evaluations:
                    session.delete(evaluation)
                    deleted_count += 1
                session.commit()
                logger.info(f"Удалено старых записей: {deleted_count}")
                return deleted_count
        except Exception as e:
            logger.error(f"Ошибка очистки старых данных: {e}")
            raise StorageError(
                f"Не удалось очистить старые данные: {e}", "cleanup_error"
            )

    def export_data(self, export_path: str) -> None:
        """
        Экспортировать данные в файл.
        Args:
            export_path: Путь для экспорта
        Raises:
            StorageError: При ошибке экспорта
        """
        try:
            export_data: Dict[str, Any] = {
                "candidates": [],
                "evaluations": [],
                "contexts": [],
                "export_time": datetime.now().isoformat(),
            }
            # Экспорт кандидатов
            candidates = self.get_strategy_candidates()
            candidates_list = list(candidates)
            for candidate in candidates_list:
                export_data["candidates"].append(candidate.to_dict())
            # Экспорт оценок
            evaluations = self.get_evaluation_results()
            evaluations_list = list(evaluations)
            for evaluation in evaluations_list:
                export_data["evaluations"].append(evaluation.to_dict())
            # Экспорт контекстов
            contexts = self.get_evolution_contexts()
            contexts_list = list(contexts)
            for context in contexts_list:
                export_data["contexts"].append(context.to_dict())
            # Сохранить в файл
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Данные экспортированы: {export_path}")
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}")
            raise StorageError(f"Не удалось экспортировать данные: {e}", "export_error")

    def import_data(self, import_path: str) -> int:
        """
        Импортировать данные из файла.
        Args:
            import_path: Путь к файлу для импорта
        Returns:
            Количество импортированных записей
        Raises:
            StorageError: При ошибке импорта
        """
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)
            imported_count = 0
            # Импорт кандидатов
            for candidate_data in import_data.get("candidates", []):
                candidate = StrategyCandidate.from_dict(candidate_data)
                self.save_strategy_candidate(candidate)
                imported_count += 1
            # Импорт контекстов
            for context_data in import_data.get("contexts", []):
                context = EvolutionContext.from_dict(context_data)
                self.save_evolution_context(context)
                imported_count += 1
            logger.info(f"Импортировано записей: {imported_count}")
            return imported_count
        except Exception as e:
            logger.error(f"Ошибка импорта данных: {e}")
            raise StorageError(f"Не удалось импортировать данные: {e}", "import_error")

    def _validate_candidate(self, candidate: StrategyCandidate) -> None:
        """Валидировать кандидата стратегии."""
        if not candidate.name:
            raise ValidationError(
                "Имя кандидата стратегии обязательно", "name", candidate.name
            )
        if candidate.position_size_pct <= 0 or candidate.position_size_pct > 1:
            raise ValidationError(
                "Размер позиции должен быть между 0 и 1",
                "position_size_pct",
                candidate.position_size_pct,
            )

    def _validate_evaluation(self, evaluation: StrategyEvaluationResult) -> None:
        """Валидировать результат оценки."""
        if evaluation.total_trades < 0:
            raise ValidationError(
                "Общее количество сделок не может быть отрицательным",
                "total_trades",
                evaluation.total_trades,
            )

    def _validate_context(self, context: EvolutionContext) -> None:
        """Валидировать контекст эволюции."""
        if not context.name:
            raise ValidationError(
                "Имя контекста эволюции обязательно", "name", context.name
            )
        if context.population_size <= 0:
            raise ValidationError(
                "Размер популяции должен быть положительным",
                "population_size",
                context.population_size,
            )

    def _calculate_storage_size(self) -> int:
        """Подсчет размера хранилища в байтах."""
        try:
            db_path = Path(self.db_path)
            if db_path.exists():
                return db_path.stat().st_size
            return 0
        except Exception as e:
            logger.warning(f"Ошибка подсчета размера хранилища: {e}")
            return 0

    def _get_last_backup_time(self) -> Optional[datetime]:
        """Получение времени последнего бэкапа."""
        try:
            backup_dir = Path(self.db_path).parent / "backups"
            if not backup_dir.exists():
                return None
            backup_files = list(backup_dir.glob(f"*{Path(self.db_path).stem}*.db"))
            if not backup_files:
                return None
            # Находим самый новый файл
            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
            return datetime.fromtimestamp(latest_backup.stat().st_mtime)
        except Exception as e:
            logger.warning(f"Ошибка получения времени бэкапа: {e}")
            return None

    def _calculate_cache_hit_rate(self) -> float:
        """Расчет hit rate кэша."""
        try:
            if not hasattr(self, "_cache_stats"):
                self._cache_stats = {"hits": 0, "misses": 0}
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            if total_requests == 0:
                return 0.0
            return self._cache_stats["hits"] / total_requests
        except Exception as e:
            logger.warning(f"Ошибка расчета hit rate кэша: {e}")
            return 0.0

    def _calculate_average_query_time(self) -> float:
        """Расчет среднего времени запроса."""
        try:
            if not hasattr(self, "_query_times"):
                self._query_times: list[float] = []
            if not self._query_times:
                return 0.0
            return sum(self._query_times) / len(self._query_times)
        except Exception as e:
            logger.warning(f"Ошибка расчета среднего времени запроса: {e}")
            return 0.0
