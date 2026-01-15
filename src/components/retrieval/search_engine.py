import logging
from typing import Dict, Any, List
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchSearchEngine:
    """
    Component: Elasticsearch DSL → Search Results
    """

    def __init__(self, host: str, index: str):
        self.host = host
        self.index = index
        self.client = Elasticsearch(self.host)

    # ---------------------------
    # Public API
    # ---------------------------
    def search(self, es_query: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing Elasticsearch query on index [%s]", self.index)

        response = self.client.search(index=self.index, body=es_query)

        hits = response.get("hits", {})
        total = hits.get("total", {}).get("value", 0)

        results: List[Dict[str, Any]] = []

        for hit in hits.get("hits", []):
            results.append(self._map_hit(hit))

        output = {
            "total": total,
            "count": len(results),
            "results": results,
        }

        logger.info("Search completed: %d results", len(results))
        return output

    # ---------------------------
    # Hit mapper
    # ---------------------------
    def _map_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        src = hit.get("_source", {})

        return {
            "id": hit.get("_id"),
            "score": hit.get("_score"),
            "camera_id": src.get("camera_id"),
            "timestamp": src.get("timestamp"),
            "frame_id": src.get("frame_id"),
            "crowd_count": src.get("crowd_count"),
            "action_events": src.get("action_events", []),
            "face_events": src.get("face_events", []),
            "track": src.get("track", []),
        }