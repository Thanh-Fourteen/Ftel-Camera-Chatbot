"""
Node 2: Build Elasticsearch DSL query from structured intent
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ==================== TIME RANGE PARSER ====================
def parse_time_range(time_desc: str) -> Dict[str, str] | None:
    """
    Convert free-form time description into Elasticsearch range query
    """
    if not time_desc:
        return None

    desc = time_desc.lower()

    if "minute" in desc:
        number = "".join(filter(str.isdigit, desc))
        if number:
            return {"gte": f"now-{number}m", "lte": "now"}

    if "hour" in desc:
        number = "".join(filter(str.isdigit, desc))
        if number:
            return {"gte": f"now-{number}h", "lte": "now"}

    if "today" in desc:
        return {"gte": "now/d", "lte": "now"}

    if "yesterday" in desc:
        return {"gte": "now-1d/d", "lte": "now/d"}

    if "morning" in desc:
        return {"gte": "now/d+6h", "lte": "now/d+12h"}

    return None


# ==================== MAIN ====================
def main(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: intent JSON from Node 1
    Output: Elasticsearch DSL query
    """

    must_clauses: List[Dict[str, Any]] = []
    filter_clauses: List[Dict[str, Any]] = []

    # ---------- Camera ----------
    if intent.get("camera_id"):
        must_clauses.append({"match_phrase": {"camera_id": intent["camera_id"]}})

    # ---------- Time range ----------
    time_range = parse_time_range(intent.get("time_range_description"))
    if time_range:
        filter_clauses.append({"range": {"timestamp": time_range}})

    # ---------- Action events (nested) ----------
    if intent.get("event_type") == "action" and intent.get("action_label"):
        action_must = [
            {"term": {"action_events.label": intent["action_label"]}},
            {
                "range": {
                    "action_events.confidence": {
                        "gte": intent.get("confidence_threshold", 0.7)
                    }
                }
            },
        ]

        if intent.get("track_id") is not None:
            action_must.append({"term": {"action_events.track_id": intent["track_id"]}})

        must_clauses.append(
            {
                "nested": {
                    "path": "action_events",
                    "query": {"bool": {"must": action_must}},
                }
            }
        )

    # ---------- Face / person type (nested) ----------
    if intent.get("person_type"):
        must_clauses.append(
            {
                "nested": {
                    "path": "face_events",
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "face_events.person_type": intent["person_type"]
                                    }
                                },
                                {
                                    "range": {
                                        "face_events.confidence": {
                                            "gte": intent.get(
                                                "confidence_threshold", 0.7
                                            )
                                        }
                                    }
                                },
                            ]
                        }
                    },
                }
            }
        )

    # ---------- 5️⃣ Final DSL ----------
    es_query = {
        "size": 50,
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "filter": filter_clauses,
            }
        },
        "sort": [{"timestamp": {"order": "desc"}}],
    }

    logger.info("Built Elasticsearch query: %s", es_query)

    return es_query
