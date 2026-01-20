import time
import asyncio
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError, EndpointConnectionError
import boto3


def get_minio_client(request: Request):
    client = getattr(request.app.state, "minio_client", None)

    if not client:
        raise HTTPException(
            status_code=500,
            detail="MinIO client not initialized in application state.",
        )
    return client


router = APIRouter(prefix="/health", tags=["Health"])

TIMEOUT = 2.0


async def check_minio(client: boto3.client):
    loop = asyncio.get_running_loop()

    try:
        start = time.time()

        await asyncio.wait_for(
            loop.run_in_executor(None, client.list_buckets),
            timeout=TIMEOUT,
        )

        latency = time.time() - start

        return {
            "status": "ok",
            "latency_ms": round(latency * 1000, 2),
        }

    except asyncio.TimeoutError:
        return {
            "status": "fail",
            "error": "MinIO connection timeout",
        }
    except EndpointConnectionError:
        return {
            "status": "fail",
            "error": "Cannot connect to MinIO endpoint",
        }
    except ClientError as e:
        return {
            "status": "fail",
            "error": e.response["Error"]["Message"],
        }
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e),
        }


@router.get("/live")
async def liveness():
    return {"status": "alive"}


@router.get("/ready")
async def readiness(minio_client=Depends(get_minio_client)):
    minio = await check_minio(minio_client)

    if minio["status"] == "fail":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "minio": minio,
            },
        )

    return {
        "status": "ready",
        "minio": minio,
    }
