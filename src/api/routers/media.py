import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
from api.core.minio import get_s3_client
from api.schemas.media import MediaRequest

router = APIRouter(prefix="/v1", tags=["Media"])
s3 = get_s3_client()

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4"}


def detect_media(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "image", "image/jpeg"
    if ext in VIDEO_EXT:
        return "video", "video/mp4"
    return None, None


@router.post("/media")
def get_media(req: MediaRequest, request: Request):
    media_type, content_type = detect_media(req.path)
    if not media_type:
        raise HTTPException(400, "Unsupported media type")

    parts = req.path.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(400, "Invalid path format")

    collection, key = parts

    try:
        if media_type == "video" and request.headers.get("range"):
            obj = s3.get_object(
                Bucket=collection,
                Key=key,
                Range=request.headers["range"]
            )
            return StreamingResponse(
                obj["Body"],
                status_code=206,
                headers={
                    "Content-Range": obj["ContentRange"],
                    "Accept-Ranges": "bytes",
                    "Content-Type": content_type,
                },
            )

        obj = s3.get_object(Bucket=collection, Key=key)
        return StreamingResponse(obj["Body"], media_type=content_type)

    except ClientError:
        raise HTTPException(404, "Media not found")
