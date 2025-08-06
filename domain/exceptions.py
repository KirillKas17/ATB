"""
Domain exceptions.
"""


class DomainError(Exception):
    """Base domain exception."""
    pass


class ValidationError(DomainError):
    """Validation error."""
    pass


class OrderError(DomainError):
    """Order-related error."""
    pass


class StrategyError(DomainError):
    """Strategy-related error."""
    pass


class TradingError(DomainError):
    """Trading-related error."""
    pass


class MLModelError(DomainError):
    """ML model error."""
    pass


class PredictionError(DomainError):
    """Prediction error."""
    pass


class CryptographyError(DomainError):
    """Cryptography error."""
    pass


class SecurityError(DomainError):
    """Security error."""
    pass


class FinancialArithmeticError(DomainError):
    """Financial arithmetic error."""
    pass


class BusinessRuleViolation(DomainError):
    """Business rule violation."""
    pass


class AuditError(DomainError):
    """Audit error."""
    pass


class ComplianceViolation(DomainError):
    """Compliance violation."""
    pass


class NetworkError(DomainError):
    """Network error."""
    pass


class ServiceUnavailableError(DomainError):
    """Service unavailable error."""
    pass


class DataIntegrityError(DomainError):
    """Data integrity error."""
    pass


class SystemOverloadError(DomainError):
    """System overload error."""
    pass


class DisasterRecoveryError(DomainError):
    """Disaster recovery error."""
    pass


class PerformanceError(DomainError):
    """Performance error."""
    pass


class ResourceError(DomainError):
    """Resource error."""
    pass


class AlertingError(DomainError):
    """Alerting error."""
    pass


class MonitoringError(DomainError):
    """Monitoring error."""
    pass


class StreamingError(DomainError):
    """Streaming error."""
    pass


class ConnectionError(DomainError):
    """Connection error."""
    pass


class DataValidationError(DomainError):
    """Data validation error."""
    pass


class RegulatoryError(DomainError):
    """Regulatory error."""
    pass
