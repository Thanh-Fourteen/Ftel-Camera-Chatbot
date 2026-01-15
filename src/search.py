"""
Node 3: Execute Elasticsearch search using DSL query
"""

import logging
from typing import Dict, Any, List
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
ES_HOST = "http://192.168.2.41:9200"
ES_INDEX = "camera-events"


# ==================== MAIN ====================
def main(es_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: Elasticsearch DSL query (from Node 2)
    Output: Search results
    """

    es = Elasticsearch(ES_HOST)

    logger.info("Executing Elasticsearch query on index [%s]", ES_INDEX)

    response = es.search(index=ES_INDEX, body=es_query)

    hits = response.get("hits", {})
    total = hits.get("total", {}).get("value", 0)

    results: List[Dict[str, Any]] = []

    for hit in hits.get("hits", []):
        src = hit.get("_source", {})
        results.append(
            {
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
        )

    output = {"total": total, "count": len(results), "results": results}

    logger.info("Search completed: %d results", len(results))

    return output
