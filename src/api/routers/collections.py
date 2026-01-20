from fastapi import APIRouter, HTTPException
from botocore.exceptions import ClientError, EndpointConnectionError
from api.core.minio import get_s3_client
from api.schemas.collections import CollectionCreate

router = APIRouter(prefix="/v1/collections", tags=["Collections"])
s3 = get_s3_client()


@router.get("")
def list_collections():
    resp = s3.list_buckets()
    return {"collections": [b["Name"] for b in resp["Buckets"]]}

@router.get("")
def list_collections():
    try:
        resp = s3.list_buckets()
        return {"collections": [b["Name"] for b in resp["Buckets"]]}

    except EndpointConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to MinIO",
        )

    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=e.response["Error"]["Message"],
        )



@router.post("")
def create_collection(data: CollectionCreate):
    try:
        s3.create_bucket(Bucket=data.name)
        return {"status": "created", "collection": data.name}
    except ClientError:
        raise HTTPException(400, "Cannot create collection")


@router.delete("/{collection}")
def delete_collection(collection: str):
    try:
        s3.delete_bucket(Bucket=collection)
        return {"status": "deleted", "collection": collection}
    except ClientError:
        raise HTTPException(404, "Collection not found")


@router.get("/{collection}")
def list_cameras(collection: str):
    try:
        resp = s3.list_objects_v2(
            Bucket=collection,
            Delimiter="/"
        )
        cameras = [
            p["Prefix"].rstrip("/")
            for p in resp.get("CommonPrefixes", [])
        ]
        return {"collection": collection, "cameras": cameras}
    except ClientError:
        raise HTTPException(404, "Collection not found")
