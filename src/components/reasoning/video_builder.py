import os
import uuid
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from datetime import timedelta
import cv2
import re

# --- CONFIG MinIO ---
MINIO_ENDPOINT = "http://192.168.2.21:9000"
MINIO_BUCKET = "camera-frames"
MINIO_PREFIX = ""
MINIO_VIDEO_PREFIX = "videos/"

# Hardcode tạm (nên chuyển sang wmill resource sau)
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    config=Config(signature_version="s3v4"),
)

FPS = 10
PRESIGNED_EXPIRY = timedelta(days=7)

VIDEO_ROOT = "/tmp/videos"
os.makedirs(VIDEO_ROOT, exist_ok=True)


def download_frame_from_minio(cam_id: str, frame_id: str, tmp_dir: str) -> str:
    key = f"{MINIO_PREFIX}{cam_id}/{frame_id}.jpg"
    local_path = f"{tmp_dir}/{uuid.uuid4()}.jpg"

    try:
        s3_client.download_file(MINIO_BUCKET, key, local_path)
        print(f"Downloaded: {key}")
        return local_path
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return None


def create_video_from_frames(frame_paths: list, video_path: str) -> bool:
    """Tạo video từ list đường dẫn ảnh bằng OpenCV"""
    if not frame_paths:
        return False

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print("Cannot read first frame")
        return False

    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"Skip invalid frame: {path}")

    video_writer.release()
    print(f"Video created: {video_path}")
    return os.path.exists(video_path)


def upload_video_to_minio(video_path: str, video_id: str) -> str:
    key = f"{MINIO_VIDEO_PREFIX}{video_id}.mp4"

    try:
        s3_client.upload_file(
            video_path, MINIO_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"}
        )
        print(f"Uploaded: {key}")

        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": MINIO_BUCKET,
                "Key": key,
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=int(PRESIGNED_EXPIRY.total_seconds()),
        )
        return presigned_url

    except ClientError as e:
        print(f"Failed upload/presign {key}: {e}")
        return None


def extract_frame_number(frame_id):
    # frame_000123 → 123
    m = re.search(r"(\d+)$", frame_id)
    return int(m.group(1)) if m else None


def build_continuous_segments(frames, max_frame_gap=3, max_time_gap=0.8):
    """
    frames = list of {
        frame_id,
        timestamp,
        camera_id
    }
    """
    frames = sorted(
        frames,
        key=lambda f: (
            f["camera_id"],
            extract_frame_number(f["frame_id"]) or 0,
            f["timestamp"],
        ),
    )

    segments = []
    current = []

    last_frame_num = None
    last_time = None
    last_cam = None

    for f in frames:
        frame_num = extract_frame_number(f["frame_id"])
        t = f["timestamp"]
        cam = f["camera_id"]

        new_segment = False

        if not current:
            new_segment = True
        else:
            if cam != last_cam:
                new_segment = True
            elif frame_num is not None and last_frame_num is not None:
                if frame_num - last_frame_num > max_frame_gap:
                    new_segment = True
            elif abs(t - last_time) > max_time_gap:
                new_segment = True

        if new_segment:
            if current:
                segments.append(current)
            current = [f]
        else:
            current.append(f)

        last_frame_num = frame_num
        last_time = t
        last_cam = cam

    if current:
        segments.append(current)

    return segments


def main(session_result):
    all_frames = []
    videos = []

    for s in session_result["sessions"]:
        for f in s["frames"]:
            all_frames.append(
                {
                    "frame_id": f["frame_id"],
                    "timestamp": f["timestamp"],
                    "camera_id": s["camera_id"],
                }
            )

    segments = build_continuous_segments(all_frames)

    for seg in segments:
        cam = seg[0]["camera_id"]

        tmp_dir = f"/tmp/frames_{uuid.uuid4()}"
        os.makedirs(tmp_dir, exist_ok=True)

        frame_paths = []

        for f in seg:
            fid = f["frame_id"]
            downloaded = download_frame_from_minio(cam, fid, tmp_dir)
            if downloaded:
                frame_paths.append(downloaded)

        if len(frame_paths) < FPS:  # < 1 giây → bỏ
            os.system(f"rm -rf {tmp_dir}")
            continue

        video_id = str(uuid.uuid4())
        video_path = f"{VIDEO_ROOT}/{video_id}.mp4"

        create_video_from_frames(frame_paths, video_path)
        url = upload_video_to_minio(video_path, video_id)

        # videos.append({
        #     "camera": cam,
        #     "start_frame": seg[0]["frame_id"],
        #     "end_frame": seg[-1]["frame_id"],
        #     "start_time": seg[0]["timestamp"],
        #     "end_time": seg[-1]["timestamp"],
        #     "video_url": url
        # })
        videos.append(video_id)

    return videos
