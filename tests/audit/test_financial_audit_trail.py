#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты аудита и логирования финансовых операций.
Критически важно для финансовой системы - каждая операция должна быть залогирована.
"""

import pytest
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, call
import logging
from dataclasses import asdict

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.portfolio import Portfolio
from domain.entities.audit_log import AuditLog, AuditEvent, AuditLevel
from infrastructure.logging.audit_logger import AuditLogger
from infrastructure.logging.compliance_logger import ComplianceLogger
from domain.exceptions import AuditError, ComplianceViolation


class TestFinancialAuditTrail:
    """Comprehensive тесты аудита финансовых операций."""

    @pytest.fixture
    def audit_logger(self) -> AuditLogger:
        """Фикстура audit logger."""
        return AuditLogger(
            log_level=AuditLevel.ALL,
            encryption_enabled=True,
            compliance_mode=True,
            retention_days=2555  # 7 лет
        )

    @pytest.fixture
    def compliance_logger(self) -> ComplianceLogger:
        """Фикстура compliance logger."""
        return ComplianceLogger(
            jurisdiction="US",
            regulation_type="SOX",
            real_time_monitoring=True
        )

    @pytest.fixture
    def sample_order_event(self) -> Dict[str, Any]:
        """Фикстура события ордера для аудита."""
        return {
            "event_type": "ORDER_CREATED",
            "order_id": str(uuid.uuid4()),
            "user_id": "user_12345",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": "0.001",
            "price": "45000.00",
            "timestamp": datetime.utcnow().isoformat(),
            "source_ip": "192.168.1.100",
            "session_id": str(uuid.uuid4()),
            "compliance_flags": []
        }

    def test_order_creation_audit_trail(
        self, 
        audit_logger: AuditLogger,
        sample_order_event: Dict[str, Any]
    ) -> None:
        """Тест аудита создания ордера."""
        # Логируем создание ордера
        audit_entry = audit_logger.log_order_event(
            event=AuditEvent.ORDER_CREATED,
            order_data=sample_order_event,
            metadata={
                "risk_score": 0.25,
                "validation_passed": True,
                "regulatory_checks": ["AML", "KYC", "SANCTIONS"]
            }
        )
        
        # Проверяем структуру аудит записи
        assert audit_entry.event_id is not None
        assert audit_entry.event_type == AuditEvent.ORDER_CREATED
        assert audit_entry.user_id == sample_order_event["user_id"]
        assert audit_entry.order_id == sample_order_event["order_id"]
        assert audit_entry.timestamp is not None
        assert audit_entry.checksum is not None
        
        # Проверяем что данные зашифрованы
        assert audit_entry.encrypted_data is not None
        assert audit_entry.encrypted_data != json.dumps(sample_order_event)
        
        # Проверяем цифровую подпись
        assert audit_logger.verify_signature(audit_entry) is True

    def test_order_modification_audit_chain(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест аудита цепочки модификаций ордера."""
        order_id = str(uuid.uuid4())
        user_id = "user_12345"
        
        # 1. Создание ордера
        creation_event = {
            "event_type": "ORDER_CREATED",
            "order_id": order_id,
            "user_id": user_id,
            "original_price": "45000.00",
            "original_quantity": "0.001"
        }
        
        creation_audit = audit_logger.log_order_event(
            AuditEvent.ORDER_CREATED, creation_event
        )
        
        # 2. Модификация цены
        modification_event = {
            "event_type": "ORDER_MODIFIED",
            "order_id": order_id,
            "user_id": user_id,
            "field_changed": "price",
            "old_value": "45000.00",
            "new_value": "44500.00",
            "reason": "Market conditions changed"
        }
        
        modification_audit = audit_logger.log_order_event(
            AuditEvent.ORDER_MODIFIED, modification_event
        )
        
        # 3. Отмена ордера
        cancellation_event = {
            "event_type": "ORDER_CANCELLED",
            "order_id": order_id,
            "user_id": user_id,
            "reason": "User request",
            "remaining_quantity": "0.001"
        }
        
        cancellation_audit = audit_logger.log_order_event(
            AuditEvent.ORDER_CANCELLED, cancellation_event
        )
        
        # Проверяем цепочку аудита
        audit_chain = audit_logger.get_audit_chain(order_id)
        
        assert len(audit_chain) == 3
        assert audit_chain[0].event_type == AuditEvent.ORDER_CREATED
        assert audit_chain[1].event_type == AuditEvent.ORDER_MODIFIED
        assert audit_chain[2].event_type == AuditEvent.ORDER_CANCELLED
        
        # Проверяем целостность цепочки
        for i in range(1, len(audit_chain)):
            current_entry = audit_chain[i]
            previous_entry = audit_chain[i-1]
            assert current_entry.previous_hash == previous_entry.checksum

    def test_financial_transaction_audit(
        self, 
        audit_logger: AuditLogger,
        compliance_logger: ComplianceLogger
    ) -> None:
        """Тест аудита финансовых транзакций."""
        transaction_data = {
            "transaction_id": str(uuid.uuid4()),
            "transaction_type": "TRADE_EXECUTION",
            "user_id": "user_12345",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "executed_quantity": "0.001",
            "executed_price": "45000.50",
            "commission": "0.45",
            "net_amount": "44.5505",
            "settlement_currency": "USD",
            "counterparty": "EXCHANGE_BYBIT",
            "execution_timestamp": datetime.utcnow().isoformat()
        }
        
        # Логируем транзакцию
        audit_entry = audit_logger.log_financial_transaction(transaction_data)
        
        # Compliance логирование
        compliance_entry = compliance_logger.log_transaction(
            transaction_data,
            regulatory_requirements=["MiFID_II", "EMIR", "DODD_FRANK"]
        )
        
        # Проверяем аудит запись
        assert audit_entry.transaction_id == transaction_data["transaction_id"]
        assert audit_entry.amount == Decimal(transaction_data["net_amount"])
        assert audit_entry.compliance_status == "COMPLIANT"
        
        # Проверяем compliance запись
        assert compliance_entry.regulation_compliance["MiFID_II"] is True
        assert compliance_entry.reporting_status == "PENDING_REPORT"
        assert compliance_entry.risk_assessment is not None

    def test_portfolio_rebalancing_audit(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест аудита ребалансировки портфеля."""
        rebalancing_data = {
            "rebalancing_id": str(uuid.uuid4()),
            "user_id": "user_12345",
            "portfolio_id": "portfolio_001",
            "rebalancing_type": "AUTOMATIC",
            "trigger_reason": "RISK_LIMIT_BREACH",
            "before_allocation": {
                "BTCUSDT": "60.0",
                "ETHUSDT": "30.0", 
                "ADAUSDT": "10.0"
            },
            "after_allocation": {
                "BTCUSDT": "50.0",
                "ETHUSDT": "35.0",
                "ADAUSDT": "15.0"
            },
            "trades_executed": [
                {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "quantity": "0.0022",
                    "price": "45000.00"
                },
                {
                    "symbol": "ETHUSDT", 
                    "side": "BUY",
                    "quantity": "0.156",
                    "price": "3200.00"
                }
            ],
            "risk_metrics_before": {
                "var_95": "1250.00",
                "max_drawdown": "0.15"
            },
            "risk_metrics_after": {
                "var_95": "1100.00",
                "max_drawdown": "0.12"
            }
        }
        
        # Логируем ребалансировку
        audit_entry = audit_logger.log_portfolio_rebalancing(rebalancing_data)
        
        # Проверяем детали аудита
        assert audit_entry.rebalancing_id == rebalancing_data["rebalancing_id"]
        assert audit_entry.trigger_reason == "RISK_LIMIT_BREACH"
        assert len(audit_entry.trades_executed) == 2
        assert audit_entry.risk_improvement is True  # VaR снизился

    def test_regulatory_reporting_audit(
        self, 
        compliance_logger: ComplianceLogger
    ) -> None:
        """Тест аудита регуляторной отчетности."""
        report_data = {
            "report_id": str(uuid.uuid4()),
            "report_type": "DAILY_POSITION_REPORT",
            "reporting_date": datetime.utcnow().date().isoformat(),
            "regulator": "SEC",
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_type": "LONG",
                    "quantity": "1.5",
                    "market_value": "67500.00",
                    "unrealized_pnl": "2500.00"
                },
                {
                    "symbol": "ETHUSDT",
                    "position_type": "LONG", 
                    "quantity": "15.0",
                    "market_value": "48000.00",
                    "unrealized_pnl": "1200.00"
                }
            ],
            "total_portfolio_value": "115500.00",
            "risk_metrics": {
                "var_95": "2300.00",
                "leverage_ratio": "1.5",
                "concentration_risk": "LOW"
            }
        }
        
        # Создаем регуляторный отчет
        report_audit = compliance_logger.create_regulatory_report(report_data)
        
        # Проверяем отчет
        assert report_audit.report_id == report_data["report_id"]
        assert report_audit.submission_status == "READY_FOR_SUBMISSION"
        assert report_audit.validation_errors == []
        assert report_audit.digital_signature is not None
        
        # Симулируем отправку отчета
        submission_result = compliance_logger.submit_report(report_audit)
        
        assert submission_result.submitted is True
        assert submission_result.submission_timestamp is not None
        assert submission_result.confirmation_number is not None

    def test_aml_kyc_audit_trail(
        self, 
        compliance_logger: ComplianceLogger
    ) -> None:
        """Тест аудита AML/KYC проверок."""
        user_id = "user_12345"
        
        # KYC проверка
        kyc_data = {
            "user_id": user_id,
            "verification_type": "IDENTITY_VERIFICATION",
            "documents_provided": ["PASSPORT", "UTILITY_BILL"],
            "verification_status": "APPROVED",
            "verification_score": 95,
            "risk_level": "LOW",
            "sanctions_check": "CLEAR",
            "pep_check": "CLEAR",
            "adverse_media_check": "CLEAR"
        }
        
        kyc_audit = compliance_logger.log_kyc_verification(kyc_data)
        
        # AML мониторинг
        aml_data = {
            "user_id": user_id,
            "transaction_id": str(uuid.uuid4()),
            "monitoring_type": "TRANSACTION_MONITORING",
            "amount": "50000.00",
            "currency": "USD",
            "suspicious_patterns": [],
            "risk_score": 25,
            "alert_triggered": False,
            "manual_review_required": False
        }
        
        aml_audit = compliance_logger.log_aml_monitoring(aml_data)
        
        # Проверяем KYC аудит
        assert kyc_audit.user_id == user_id
        assert kyc_audit.verification_status == "APPROVED"
        assert kyc_audit.compliance_score >= 90
        
        # Проверяем AML аудит
        assert aml_audit.user_id == user_id
        assert aml_audit.risk_score <= 50  # Низкий риск
        assert aml_audit.alert_triggered is False

    def test_audit_log_encryption_and_integrity(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест шифрования и целостности аудит логов."""
        sensitive_data = {
            "user_id": "user_12345",
            "account_number": "ACC-987654321",
            "bank_routing": "121000248",
            "transaction_amount": "100000.00",
            "personal_data": {
                "ssn": "123-45-6789",
                "date_of_birth": "1990-01-01"
            }
        }
        
        # Логируем чувствительные данные
        audit_entry = audit_logger.log_sensitive_operation(
            operation_type="BANK_TRANSFER",
            sensitive_data=sensitive_data,
            encryption_level="AES_256"
        )
        
        # Проверяем что данные зашифрованы
        assert audit_entry.encrypted_data is not None
        assert "123-45-6789" not in audit_entry.encrypted_data  # SSN не в открытом виде
        assert "ACC-987654321" not in audit_entry.encrypted_data  # Account не в открытом виде
        
        # Проверяем цифровую подпись
        assert audit_entry.digital_signature is not None
        assert audit_logger.verify_integrity(audit_entry) is True
        
        # Проверяем расшифровку (только с правильным ключом)
        decrypted_data = audit_logger.decrypt_audit_entry(audit_entry)
        assert decrypted_data["account_number"] == sensitive_data["account_number"]
        assert decrypted_data["personal_data"]["ssn"] == sensitive_data["personal_data"]["ssn"]

    def test_audit_retention_and_archival(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест retention и архивирования аудит логов."""
        # Создаем аудит записи разного возраста
        old_entries = []
        for i in range(10):
            days_old = 365 * 8 + i  # 8+ лет назад
            old_timestamp = datetime.utcnow() - timedelta(days=days_old)
            
            entry_data = {
                "event_id": str(uuid.uuid4()),
                "user_id": f"user_{i}",
                "timestamp": old_timestamp.isoformat(),
                "data": f"Old transaction {i}"
            }
            
            old_entry = audit_logger.create_audit_entry(
                AuditEvent.TRANSACTION_EXECUTED,
                entry_data,
                timestamp_override=old_timestamp
            )
            old_entries.append(old_entry)
        
        # Проверяем retention policy
        retention_report = audit_logger.check_retention_compliance()
        
        assert retention_report.entries_to_archive > 0
        assert retention_report.entries_to_delete > 0  # Старше 7 лет
        assert retention_report.compliance_status == "ACTION_REQUIRED"
        
        # Выполняем архивирование
        archive_result = audit_logger.archive_old_entries(
            older_than_days=365 * 7  # 7 лет
        )
        
        assert archive_result.archived_count > 0
        assert archive_result.archive_location is not None
        assert archive_result.archive_integrity_verified is True

    def test_real_time_audit_monitoring(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест real-time мониторинга аудита."""
        suspicious_activities = []
        
        # Устанавливаем real-time мониторинг
        def audit_alert_handler(alert):
            suspicious_activities.append(alert)
        
        audit_logger.set_real_time_monitoring(
            enabled=True,
            alert_handler=audit_alert_handler,
            alert_thresholds={
                "large_transaction": Decimal("100000.00"),
                "rapid_transactions": 10,  # 10 транзакций в минуту
                "failed_login_attempts": 5
            }
        )
        
        # Симулируем подозрительную активность
        large_transaction = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": "user_12345",
            "amount": "150000.00",  # Превышает порог
            "currency": "USD",
            "transaction_type": "WITHDRAWAL"
        }
        
        audit_logger.log_financial_transaction(large_transaction)
        
        # Симулируем множественные транзакции
        for i in range(12):  # Превышает порог 10/минуту
            rapid_transaction = {
                "transaction_id": str(uuid.uuid4()),
                "user_id": "user_suspicious",
                "amount": "1000.00",
                "timestamp": datetime.utcnow().isoformat()
            }
            audit_logger.log_financial_transaction(rapid_transaction)
        
        # Проверяем что алерты сработали
        assert len(suspicious_activities) >= 2
        
        large_tx_alert = next(
            alert for alert in suspicious_activities 
            if alert.alert_type == "LARGE_TRANSACTION"
        )
        assert large_tx_alert.user_id == "user_12345"
        assert large_tx_alert.severity == "HIGH"
        
        rapid_tx_alert = next(
            alert for alert in suspicious_activities
            if alert.alert_type == "RAPID_TRANSACTIONS"
        )
        assert rapid_tx_alert.user_id == "user_suspicious"
        assert rapid_tx_alert.transaction_count >= 10

    def test_audit_search_and_forensics(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест поиска и форензики аудит логов."""
        # Создаем различные аудит записи для поиска
        test_user = "forensic_user_001"
        
        transactions = [
            {"type": "DEPOSIT", "amount": "5000.00", "currency": "USD"},
            {"type": "TRADE", "symbol": "BTCUSDT", "amount": "2000.00"},
            {"type": "TRADE", "symbol": "ETHUSDT", "amount": "1500.00"},
            {"type": "WITHDRAWAL", "amount": "500.00", "currency": "USD"},
        ]
        
        for transaction in transactions:
            audit_logger.log_financial_transaction({
                **transaction,
                "user_id": test_user,
                "transaction_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Поиск по пользователю
        user_audit_trail = audit_logger.search_audit_logs(
            filters={"user_id": test_user},
            date_range=(
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow()
            )
        )
        
        assert len(user_audit_trail) == 4
        assert all(entry.user_id == test_user for entry in user_audit_trail)
        
        # Поиск больших транзакций
        large_transactions = audit_logger.search_audit_logs(
            filters={
                "transaction_type": ["DEPOSIT", "WITHDRAWAL"],
                "amount_range": (Decimal("1000.00"), Decimal("10000.00"))
            }
        )
        
        assert len(large_transactions) >= 1
        
        # Форензический анализ паттернов
        forensic_analysis = audit_logger.analyze_transaction_patterns(
            user_id=test_user,
            analysis_period_days=1
        )
        
        assert forensic_analysis.total_transactions == 4
        assert forensic_analysis.total_volume == Decimal("9000.00")
        assert forensic_analysis.risk_indicators == []  # Нормальная активность

    def test_cross_system_audit_correlation(
        self, 
        audit_logger: AuditLogger,
        compliance_logger: ComplianceLogger
    ) -> None:
        """Тест корреляции аудита между системами."""
        correlation_id = str(uuid.uuid4())
        user_id = "correlation_user_001"
        
        # Логируем события в разных системах
        trading_event = {
            "correlation_id": correlation_id,
            "system": "TRADING_ENGINE",
            "user_id": user_id,
            "event": "ORDER_EXECUTED",
            "order_id": str(uuid.uuid4()),
            "symbol": "BTCUSDT",
            "quantity": "0.1"
        }
        
        settlement_event = {
            "correlation_id": correlation_id,
            "system": "SETTLEMENT_ENGINE", 
            "user_id": user_id,
            "event": "TRADE_SETTLED",
            "settlement_id": str(uuid.uuid4()),
            "amount": "4500.00"
        }
        
        risk_event = {
            "correlation_id": correlation_id,
            "system": "RISK_MANAGEMENT",
            "user_id": user_id,
            "event": "RISK_CHECK_PASSED",
            "risk_score": 15
        }
        
        # Логируем в audit системе
        audit_logger.log_system_event(trading_event)
        audit_logger.log_system_event(settlement_event)
        
        # Логируем в compliance системе
        compliance_logger.log_cross_system_event(risk_event)
        
        # Коррелируем события
        correlated_events = audit_logger.correlate_events(correlation_id)
        
        assert len(correlated_events) >= 2
        assert all(event.correlation_id == correlation_id for event in correlated_events)
        
        # Проверяем целостность транзакции
        transaction_integrity = audit_logger.verify_transaction_integrity(
            correlation_id
        )
        
        assert transaction_integrity.all_systems_logged is True
        assert transaction_integrity.sequence_complete is True
        assert transaction_integrity.no_inconsistencies is True

    def test_audit_performance_under_load(
        self, 
        audit_logger: AuditLogger
    ) -> None:
        """Тест производительности аудита под нагрузкой."""
        import time
        import threading
        
        # Счетчики для мониторинга
        logged_events = []
        failed_events = []
        
        def log_events_worker(worker_id: int, events_count: int):
            """Worker функция для логирования событий."""
            for i in range(events_count):
                try:
                    event_data = {
                        "worker_id": worker_id,
                        "event_number": i,
                        "user_id": f"load_test_user_{worker_id}",
                        "transaction_id": str(uuid.uuid4()),
                        "amount": f"{1000 + i}.00"
                    }
                    
                    audit_entry = audit_logger.log_financial_transaction(event_data)
                    logged_events.append(audit_entry)
                    
                except Exception as e:
                    failed_events.append((worker_id, i, str(e)))
        
        # Запускаем нагрузочное тестирование
        start_time = time.time()
        
        threads = []
        workers_count = 10
        events_per_worker = 100
        
        for worker_id in range(workers_count):
            thread = threading.Thread(
                target=log_events_worker,
                args=(worker_id, events_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Анализируем производительность
        total_events = workers_count * events_per_worker
        execution_time = end_time - start_time
        events_per_second = total_events / execution_time
        
        # Проверяем результаты
        assert len(logged_events) >= total_events * 0.95  # 95% успешность
        assert len(failed_events) <= total_events * 0.05  # 5% максимум ошибок
        assert events_per_second >= 100  # Минимум 100 событий/сек
        
        # Проверяем целостность логов
        unique_events = set(event.event_id for event in logged_events)
        assert len(unique_events) == len(logged_events)  # Все события уникальны

    def test_audit_compliance_validation(
        self, 
        compliance_logger: ComplianceLogger
    ) -> None:
        """Тест валидации соответствия требованиям."""
        # Тест SOX compliance
        sox_transaction = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": "sox_user_001",
            "amount": "75000.00",
            "transaction_type": "LARGE_TRANSFER",
            "approval_required": True,
            "approver_id": "manager_001",
            "business_justification": "Client withdrawal request"
        }
        
        sox_compliance = compliance_logger.validate_sox_compliance(sox_transaction)
        
        assert sox_compliance.compliant is True
        assert sox_compliance.approval_documented is True
        assert sox_compliance.segregation_of_duties is True
        
        # Тест MiFID II compliance (EU)
        mifid_trade = {
            "trade_id": str(uuid.uuid4()),
            "user_id": "eu_user_001",
            "instrument_type": "CRYPTO_DERIVATIVE",
            "trade_amount": "25000.00",
            "best_execution_policy": "FOLLOWED",
            "client_categorization": "RETAIL",
            "risk_warnings_provided": True
        }
        
        mifid_compliance = compliance_logger.validate_mifid_compliance(mifid_trade)
        
        assert mifid_compliance.compliant is True
        assert mifid_compliance.best_execution_documented is True
        assert mifid_compliance.client_protection_adequate is True
        
        # Тест GDPR compliance
        gdpr_data_processing = {
            "user_id": "gdpr_user_001",
            "data_type": "PERSONAL_FINANCIAL_DATA",
            "processing_purpose": "TRADE_EXECUTION",
            "consent_obtained": True,
            "data_minimization": True,
            "retention_period": "7_YEARS"
        }
        
        gdpr_compliance = compliance_logger.validate_gdpr_compliance(gdpr_data_processing)
        
        assert gdpr_compliance.compliant is True
        assert gdpr_compliance.consent_valid is True
        assert gdpr_compliance.data_minimized is True