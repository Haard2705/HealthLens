"""The standard error envelope: every error is {"error": {"code": ..., "message": ...}}."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

NOT_FOUND_PATIENT_MESSAGE = "Patient not found."


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def patient_not_found() -> AppError:
    """Same body whether the patient is missing or merely hidden from the caller."""
    return AppError(404, "not_found", NOT_FOUND_PATIENT_MESSAGE)


_STATUS_CODES = {
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
}


def _respond(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(envelope(code, message), status_code=status, headers=headers)


async def _app_error(_: Request, exc: AppError) -> JSONResponse:
    return _respond(exc.status, exc.code, exc.message)


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _respond(exc.status_code, code, message)


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Only field locations and messages; never echo the submitted values back.
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
    )
    return _respond(422, "validation_error", problems or "Invalid request.")


async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    return _respond(500, "internal_error", "Something went wrong.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
