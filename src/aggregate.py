from datetime import datetime

FRAME_GAP = 30


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", ""))


def get_frame_number(frame_id: str) -> int:
    return int(frame_id.split("_")[1])


def main(es_result):
    frames = es_result["results"]

    tracks = {}

    # 1️⃣ Build track timelines
    for f in frames:
        cam = f["camera_id"]

        for t in f.get("track", []):
            if t["label"] != "person":
                continue

            tid = t["track_id"]
            key = f"{cam}:{tid}"

            tracks.setdefault(key, []).append(
                {
                    "camera_id": cam,
                    "track_id": tid,
                    "frame_id": f["frame_id"],
                    "timestamp": f["timestamp"],
                    "crowd_count": f.get("crowd_count"),
                    "box": t.get("box"),
                    "track_confidence": t.get("confidence"),
                    "actions": [
                        a for a in f.get("action_events", []) if a["track_id"] == tid
                    ],
                    "faces": [
                        fe for fe in f.get("face_events", []) if fe["track_id"] == tid
                    ],
                    "raw_frame": f,  # giữ frame gốc
                }
            )

    # 2️⃣ Split by gaps into sessions
    sessions = []

    for key, items in tracks.items():
        items.sort(key=lambda x: x["frame_id"])

        cur = [items[0]]

        for prev, now in zip(items, items[1:]):
            now_id = get_frame_number(now["frame_id"])
            prev_id = get_frame_number(prev["frame_id"])

            if now_id - prev_id <= FRAME_GAP:
                cur.append(now)
            else:
                sessions.append(cur)
                cur = [now]

        sessions.append(cur)

    # 3️⃣ Build rich person sessions
    result = []

    for s in sessions:
        all_actions = []
        all_faces = []

        for f in s:
            all_actions += f["actions"]
            all_faces += f["faces"]

        best_face = max(all_faces, key=lambda x: x["confidence"], default={})

        result.append(
            {
                "camera_id": s[0]["camera_id"],
                "track_id": s[0]["track_id"],
                "start_time": s[0]["timestamp"],
                "end_time": s[-1]["timestamp"],
                "start_frame": s[0]["frame_id"],
                "end_frame": s[-1]["frame_id"],
                "person_id": best_face.get("personal_id"),
                "person_type": best_face.get("person_type"),
                "event_types": sorted(set(a["label"] for a in all_actions)),
                "is_critical": any(a.get("is_critical") for a in all_actions),
                "crowd_min": min(x["crowd_count"] or 0 for x in s),
                "crowd_max": max(x["crowd_count"] or 0 for x in s),
                "frames": s,
            }
        )
    # return {"total_person_sessions": 2, "sessions": result[4:6]}
    return {"total_person_sessions": len(result), "sessions": result}
