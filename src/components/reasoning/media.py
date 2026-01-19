import os
import uuid
import boto3
import cv2
import shutil
from urllib.parse import unquote
from botocore.client import Config
from botocore.exceptions import ClientError
from datetime import timedelta

# --- CONFIG chi code them lay từ env---
MINIO_ENDPOINT = "http://192.168.2.21:9000"
MINIO_BUCKET_IMAGES = "camera-frames"
MINIO_BUCKET_VIDEOS = "camera-videos"

AWS_ID = "minioadmin"
AWS_KEY = "minioadmin"

FPS = 25
PRESIGNED_EXPIRY = timedelta(days=7)

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=AWS_ID,
    aws_secret_access_key=AWS_KEY,
    config=Config(signature_version="s3v4"),
)

# ------------------ DOWNLOAD FRAME ------------------
def download_frame_from_minio(image_key: str, tmp_dir: str) -> str | None:
    os.makedirs(tmp_dir, exist_ok=True)

    image_key = unquote(image_key)
    local_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")

    try:
        s3_client.download_file(
            MINIO_BUCKET_IMAGES,
            image_key,
            local_path
        )
        return local_path
    except Exception as e:
        print(f"[ERROR] Download {image_key}: {e}")
        return None

# ------------------ CREATE VIDEO ------------------
def create_video_from_frames(frame_paths: list, video_path: str) -> bool:
    if not frame_paths:
        return False

    first = cv2.imread(frame_paths[0])
    if first is None:
        return False

    h, w, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    for p in frame_paths:
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)

    writer.release()
    return os.path.exists(video_path)

# ------------------ UPLOAD VIDEO ------------------
def upload_video_to_minio(video_path: str, base_path: str) -> str | None:
    video_id = str(uuid.uuid4())
    key = f"{base_path}{video_id}.mp4"

    try:
        s3_client.upload_file(
            video_path,
            MINIO_BUCKET_VIDEOS,
            key,
            ExtraArgs={"ContentType": "video/mp4"}
        )

        return s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": MINIO_BUCKET_VIDEOS,
                "Key": key,
            },
            ExpiresIn=int(PRESIGNED_EXPIRY.total_seconds()),
        )

    except ClientError as e:
        print(f"[ERROR] Upload video: {e}")
        return None

# ------------------ PIPELINE ------------------
def frames_to_video(image_keys: list) -> str | None:
    """
    image_keys: list image_path từ ES
    """
    tmp_dir = f"/tmp/frames_{uuid.uuid4()}"
    video_path = f"/tmp/{uuid.uuid4()}.mp4"

    frame_files = []
    for key in image_keys:
        f = download_frame_from_minio(key, tmp_dir)
        if f:
            frame_files.append(f)

    if not frame_files:
        return None

    if not create_video_from_frames(frame_files, video_path):
        return None

    base_path = image_keys[0].rsplit("/", 1)[0] + "/"

    url = upload_video_to_minio(video_path, base_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(video_path)

    return url


import os
import uuid
import boto3
import cv2
import shutil
from urllib.parse import unquote
from datetime import timedelta
from typing import List, Dict, Any
from botocore.client import Config
from botocore.exceptions import ClientError


class MediaProcessor:
    def __init__(
        self,
        minio_endpoint: str = "http://192.168.2.21:9000",
        image_bucket: str = "camera-frames",
        video_bucket: str = "camera-videos",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        fps: int = 25,
        presigned_expiry_days: int = 7,
        tmp_root: str = "/tmp",
    ):
        self.image_bucket = image_bucket
        self.video_bucket = video_bucket
        self.fps = fps
        self.tmp_root = tmp_root
        self.presigned_expiry = timedelta(days=presigned_expiry_days)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

        self._ensure_bucket_exists(self.image_bucket)
        self._ensure_bucket_exists(self.video_bucket)

    # ---------------- PRIVATE ----------------
    def _ensure_bucket_exists(self, bucket_name: str):
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError:
            print(f"[INFO] Creating bucket: {bucket_name}")
            self.s3.create_bucket(Bucket=bucket_name)

    def _download_frame(self, image_key: str, tmp_dir: str) -> str | None:
        image_key = unquote(image_key)
        local_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")

        try:
            self.s3.download_file(
                self.image_bucket,
                image_key,
                local_path,
            )
            return local_path
        except Exception as e:
            print(f"[ERROR] Download {image_key}: {e}")
            return None

    def _create_video(self, frame_paths: List[str], video_path: str) -> bool:
        if not frame_paths:
            return False

        first = cv2.imread(frame_paths[0])
        if first is None:
            return False

        h, w, _ = first.shape
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )

        for p in frame_paths:
            img = cv2.imread(p)
            if img is not None:
                writer.write(img)

        writer.release()
        return os.path.exists(video_path)

    def _upload_video(self, video_path: str, base_path: str) -> Dict[str, str] | None:
        video_id = str(uuid.uuid4())
        key = f"{base_path}{video_id}.mp4"

        try:
            self.s3.upload_file(
                video_path,
                self.video_bucket,
                key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

            return {"key": key}

        except ClientError as e:
            print(f"[ERROR] Upload video {key}: {e}")
            return None

    # ---------------- PUBLIC ----------------
    def build_videos_from_sequences(
        self,
        sequences: List[List[Dict[str, Any]]],
        image_field: str = "image_path",
        strip_prefix: str = "/data/frames/",
    ) -> List[Dict[str, Any]]:
        results = []

        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq = sorted(seq, key=lambda x: x.get("frame_ts", 0))
            tmp_dir = os.path.join(self.tmp_root, f"frames_{uuid.uuid4()}")
            os.makedirs(tmp_dir, exist_ok=True)
            video_path = os.path.join(self.tmp_root, f"{uuid.uuid4()}.mp4")

            frame_files = []
            image_keys = []

            for d in seq:
                raw_path = d.get(image_field)
                if not raw_path:
                    continue

                key = raw_path.replace(strip_prefix, "", 1)
                image_keys.append(key)

                local = self._download_frame(key, tmp_dir)
                if local:
                    frame_files.append(local)

            if not frame_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            if not self._create_video(frame_files, video_path):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            base_path = image_keys[0].rsplit("/", 1)[0] + "/"
            uploaded = self._upload_video(video_path, base_path)

            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.remove(video_path)

            if uploaded:
                results.append(
                    {
                        "sequence_index": idx,
                        "num_frames": len(frame_files),
                        "video_key": uploaded["key"],
                    }
                )

        return results