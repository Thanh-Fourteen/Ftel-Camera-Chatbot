from fastapi import Request
from fastapi.responses import JSONResponse
from botocore.exceptions import EndpointConnectionError


def minio_exception_handler(request: Request, exc: EndpointConnectionError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "MinIO unavailable",
            "detail": str(exc),
        },
    )
