"""Custom exceptions for the application."""
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

logger = structlog.get_logger()


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class UsageLimitExceeded(AppException):
    """Raised when user exceeds their subscription usage limits."""

    def __init__(
        self,
        resource: str,
        current: int,
        limit: int,
        upgrade_message: Optional[str] = None,
    ):
        message = f"Limite de {resource} atingido. Atual: {current}, Limite: {limit}."
        super().__init__(
            message=message,
            code="USAGE_LIMIT_EXCEEDED",
            status_code=status.HTTP_403_FORBIDDEN,
            details={
                "resource": resource,
                "current": current,
                "limit": limit,
                "upgrade_message": upgrade_message or "Faca upgrade do seu plano para continuar.",
            },
        )


class SubscriptionRequired(AppException):
    """Raised when a feature requires a paid subscription."""

    def __init__(self, feature: str, required_plan: str):
        message = f"O recurso '{feature}' requer o plano {required_plan} ou superior."
        super().__init__(
            message=message,
            code="SUBSCRIPTION_REQUIRED",
            status_code=status.HTTP_403_FORBIDDEN,
            details={
                "feature": feature,
                "required_plan": required_plan,
            },
        )


class ResourceNotFound(AppException):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, identifier: Any):
        message = f"{resource} n�o encontrado(a)."
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={
                "resource": resource,
                "identifier": str(identifier),
            },
        )


class AuthenticationError(AppException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Falha na autenticacao."):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class TokenRevokedError(AppException):
    """Raised when a token has been revoked (user logged out)."""

    def __init__(self):
        super().__init__(
            message="Token foi revogado. Por favor, faca login novamente.",
            code="TOKEN_REVOKED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"action": "login_required"},
        )


class TokenExpiredError(AppException):
    """Raised when a token has expired."""

    def __init__(self):
        super().__init__(
            message="Token expirado. Por favor, faca login novamente.",
            code="TOKEN_EXPIRED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"action": "login_required"},
        )


class InvalidTokenError(AppException):
    """Raised when a token is invalid or malformed."""

    def __init__(self, reason: str = "Token invalido ou malformado."):
        super().__init__(
            message=reason,
            code="INVALID_TOKEN",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"action": "login_required"},
        )


class TokenValidationError(AppException):
    """Raised when token validation fails due to service unavailability."""

    def __init__(self):
        super().__init__(
            message="Nao foi possivel validar o token. Tente novamente.",
            code="TOKEN_VALIDATION_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"action": "retry"},
        )


class AuthorizationError(AppException):
    """Raised when user is not authorized to perform an action."""

    def __init__(self, message: str = "Voc� n�o tem permiss�o para esta a��o."):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
        )


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"field": field} if field else {},
        )


class ExternalServiceError(AppException):
    """Raised when an external service fails."""

    def __init__(self, service: str, message: str = "Servi�o externo indispon�vel."):
        super().__init__(
            message=message,
            code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service},
        )


class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        message = f"Muitas requisi��es. Tente novamente em {retry_after} segundos."
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after},
        )


# ===========================================
# Exception Handlers
# ===========================================

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    logger.warning(
        "Application exception",
        code=exc.code,
        message=exc.message,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": {},
            }
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=errors,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Erro de valida��o nos dados enviados.",
                "details": {"errors": errors},
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        "Unexpected error",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Ocorreu um erro inesperado. Tente novamente mais tarde.",
                "details": {},
            }
        },
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
