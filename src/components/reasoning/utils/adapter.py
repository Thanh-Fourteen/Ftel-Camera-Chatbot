import json
from typing import Dict, Any

def parse_es_doc_to_result(doc: Dict[str, Any]) -> Dict[str, Any]:
    def loads_safe(v, default):
        if not v:
            return default
        if isinstance(v, (dict, list)):
            return v
        try:
            return json.loads(v)
        except Exception:
            return default

    result = {
        "scale_factor": doc.get("scale_factor", 1.0),
    }

    # ===== TRACK / DETECTION =====
    tracks = loads_safe(doc.get("tracks"), [])
    if tracks:
        result["track"] = []
        result["detection"] = []

        for track_obj in tracks:
            result["track"].append(track_obj)
            result["detection"].append(track_obj)

    # ===== ACTION =====
    actions = loads_safe(doc.get("actions"), [])
    if actions:
        result["action_events"] = actions

    # ===== FIRE =====
    fire = loads_safe(doc.get("fire"), {})
    if fire:
        result["fire"] = fire

    # ===== FACE =====
    faces = loads_safe(doc.get("faces"), {})
    if faces:
        result["face_event"] = faces.get("events", [])
        result["face_detection"] = faces.get("detections", [])
        
    # ===== COUNTING =====
    counting = loads_safe(doc.get("counting"), {})
    if counting:
        result["count"] = counting

    return result