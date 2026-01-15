import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TimeRangeParser:
    """
    Convert free-form time description into Elasticsearch time range
    """

    def parse(self, time_desc: Optional[str]) -> Optional[Dict[str, str]]:
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


class ElasticsearchQueryBuilder:
    """
    Component: QueryIntent → Elasticsearch DSL
    """

    def __init__(self, time_parser: TimeRangeParser | None = None):
        self.time_parser = time_parser or TimeRangeParser()

    # ---------------------------
    # Public API
    # ---------------------------
    def build(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        must_clauses: List[Dict[str, Any]] = []
        filter_clauses: List[Dict[str, Any]] = []

        self._apply_camera(intent, must_clauses)
        self._apply_time(intent, filter_clauses)
        self._apply_action(intent, must_clauses)
        self._apply_person(intent, must_clauses)

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

    # ---------------------------
    # Sub builders
    # ---------------------------
    def _apply_camera(self, intent, must):
        if intent.get("camera_id"):
            must.append({"match_phrase": {"camera_id": intent["camera_id"]}})

    def _apply_time(self, intent, filters):
        time_range = self.time_parser.parse(intent.get("time_range_description"))
        if time_range:
            filters.append({"range": {"timestamp": time_range}})

    def _apply_action(self, intent, must):
        if intent.get("event_type") != "action" or not intent.get("action_label"):
            return

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
            action_must.append(
                {"term": {"action_events.track_id": intent["track_id"]}}
            )

        must.append(
            {
                "nested": {
                    "path": "action_events",
                    "query": {"bool": {"must": action_must}},
                }
            }
        )

    def _apply_person(self, intent, must):
        if not intent.get("person_type"):
            return

        must.append(
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
