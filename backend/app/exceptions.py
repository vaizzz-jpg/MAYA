"""Domain / API exceptions for MAYA product layer."""

from __future__ import annotations


class MayaProductError(Exception):
    """Base product-layer error."""

    status_code: int = 400
    error_code: str = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthenticationError(MayaProductError):
    status_code = 401
    error_code = "authentication_error"


class AuthorizationError(MayaProductError):
    status_code = 403
    error_code = "authorization_error"


class ValidationError(MayaProductError):
    status_code = 400
    error_code = "validation_error"


class ConflictError(MayaProductError):
    status_code = 409
    error_code = "conflict"


class NotFoundError(MayaProductError):
    status_code = 404
    error_code = "not_found"


class CaseNotFoundError(NotFoundError):
    error_code = "case_not_found"


class EvidenceNotFoundError(NotFoundError):
    error_code = "evidence_not_found"


class AnalysisNotFoundError(NotFoundError):
    error_code = "analysis_not_found"


class InvalidEvidenceError(ValidationError):
    error_code = "invalid_evidence"


class IntegrityCheckError(MayaProductError):
    status_code = 500
    error_code = "integrity_error"


class AnalysisProcessingError(MayaProductError):
    status_code = 500
    error_code = "analysis_processing_error"
