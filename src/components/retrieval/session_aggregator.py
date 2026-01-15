from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PersonSessionAggregator:
    """
    Component: Frame hits → Person sessions
    """

    def __init__(self, frame_gap: int = 30):
        self.frame_gap = frame_gap

    # ---------------------------
    # Public API
    # ---------------------------
    def aggregate(self, es_result: Dict[str, Any]) -> Dict[str, Any]:
        frames = es_result.get("results", [])

        tracks = self._build_track_timelines(frames)
        sessions = self._split_into_sessions(tracks)
        result = self._build_sessions(sessions)

        return {
            "total_person_sessions": len(result),
            "sessions": result,
        }

    # ---------------------------
    # Step 1: build per-track timelines
    # ---------------------------
    def _build_track_timelines(self, frames: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        tracks = {}

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
                            a
                            for a in f.get("action_events", [])
                            if a["track_id"] == tid
                        ],
                        "faces": [
                            fe
                            for fe in f.get("face_events", [])
                            if fe["track_id"] == tid
                        ],
                        "raw_frame": f,
                    }
                )

        return tracks

    # ---------------------------
    # Step 2: split by frame gaps
    # ---------------------------
    def _split_into_sessions(self, tracks: Dict[str, List[Dict]]) -> List[List[Dict]]:
        sessions = []

        for _, items in tracks.items():
            items.sort(key=lambda x: self._get_frame_number(x["frame_id"]))

            cur = [items[0]]

            for prev, now in zip(items, items[1:]):
                now_id = self._get_frame_number(now["frame_id"])
                prev_id = self._get_frame_number(prev["frame_id"])

                if now_id - prev_id <= self.frame_gap:
                    cur.append(now)
                else:
                    sessions.append(cur)
                    cur = [now]

            sessions.append(cur)

        return sessions

    # ---------------------------
    # Step 3: build rich sessions
    # ---------------------------
    def _build_sessions(self, sessions: List[List[Dict]]) -> List[Dict]:
        result = []

        for s in sessions:
            all_actions = []
            all_faces = []

            for f in s:
                all_actions.extend(f["actions"])
                all_faces.extend(f["faces"])

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

        return result

    # ---------------------------
    # Utilities
    # ---------------------------
    def _get_frame_number(self, frame_id: str) -> int:
        return int(frame_id.split("_")[1])
