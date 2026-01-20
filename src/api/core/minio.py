import boto3
from botocore.client import Config
from botocore.exceptions import EndpointConnectionError
from api.core.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)

def get_s3_client():
    try:
        return boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
    except EndpointConnectionError:
        raise RuntimeError("Cannot connect to MinIO")
