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

from src.components.reasoning.utils.plot import visualize
from src.components.reasoning.utils.adapter import parse_es_doc_to_result

class MediaProcessor:
    def __init__(
        self,
        minio_endpoint: str = "http://192.168.2.21:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        fps: int = 25,
        presigned_expiry_days: int = 7,
        tmp_root: str = "/tmp",
    ):
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

    # ---------------- PRIVATE ----------------
    def _ensure_bucket_exists(self, bucket_name: str):
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError:
            print(f"[INFO] Creating bucket: {bucket_name}")
            self.s3.create_bucket(Bucket=bucket_name)

    def _download_frame(self, image_key: str, tmp_dir: str) -> str | None:
        image_key = unquote(image_key)
        collection, image_path = image_key.split("/", 1)
        local_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")

        try:
            self.s3.download_file(
                collection,
                image_path,
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
        collection, base_path = base_path.split("/", 1)
        base_path = base_path.replace("/images/", "/videos/")
        key = f"{base_path}{video_id}.mp4"

        try:
            self.s3.upload_file(
                video_path,
                collection,
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
        show_visualize: int = False,
        modes: List[str] = ("tracking", "pose", "action", "face", "counting", "fire"),
        show_critical_alert: bool = True,
        inv_scale: bool = True,
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
                
                # xử lý temp
                parts = raw_path.strip("/").split("/")
                key = os.path.join("floor-01", parts[2], "20262001", "images", parts[-1])
                
                image_keys.append(key)

                local = self._download_frame(key, tmp_dir)
                if not local:
                    continue

                if show_visualize:
                    img = cv2.imread(local)
                    if img is None:
                        continue
                    data = parse_es_doc_to_result(d)
                    
                    img = visualize(
                        img, data, modes=modes,
                        show_critical_alert=show_critical_alert,
                        inv_scale=inv_scale,
                    )
                    cv2.imwrite(local, img)

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