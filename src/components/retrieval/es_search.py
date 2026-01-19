import json
import csv
from typing import List, Dict, Any, Optional
from elasticsearch.helpers import scan
from elasticsearch import Elasticsearch
import elastic_transport

class ElasticsearchSearcher:
    """
    A class to handle connections, queries, and searches in an Elasticsearch database.
    Designed for product-level code with OOP principles, error handling, and extensibility.
    Supports searching based on a query dictionary, aggregations by frame/time, and finding longest sequences.
    """

    def __init__(self, 
                 es_host: str = "http://localhost:19200", 
                 index_name: str = "joined_alerts_hcm_thanh", 
                 timeout: int = 30,
                 keyword_fields = {"camera_id"},
                 numeric_fields = {"frame_id", "frame_ts", "scale_factor"},
                 text_fields = {"image_path"},
                 json_string_fields = {"tracks", "actions", "faces", "fire", "counting"}):
        """
        Initialize the ElasticsearchSearcher with connection parameters.

        :param es_host: Elasticsearch host URL (e.g., "http://localhost:19200").
        :param index_name: Name of the Elasticsearch index to query.
        :param timeout: Request timeout in seconds.
        """
        self.es_host = es_host
        self.index_name = index_name
        self.timeout = timeout
        self.keyword_fields = keyword_fields
        self.numeric_fields = numeric_fields
        self.text_fields = text_fields
        self.json_string_fields = json_string_fields

        self.es = self._connect()

    def _connect(self) -> Elasticsearch:
        """
        Establish a connection to Elasticsearch.

        :return: Elasticsearch client instance.
        :raises RuntimeError: If connection fails.
        """
        try:
            es = Elasticsearch(
                hosts=[self.es_host],
                request_timeout=self.timeout
            )
            if not es.ping():
                raise RuntimeError("Failed to connect to Elasticsearch.")

            return es
        except elastic_transport.TransportError as e:
            raise RuntimeError(f"Connection error: {str(e)}") from e
    
    def _normalize_json_field_input(self, field: str, value: Any) -> Any:
        """
        Normalize shorthand user input to structured JSON-like dict.

        Example:
            tracks = 3  → {"track_id": 3}
            fire = "normal" → {"state": "normal"}
        """
        if field == "tracks" and isinstance(value, int):
            return {"track_id": value}

        if field == "actions" and isinstance(value, str):
            return {"label": value}

        if field == "fire" and isinstance(value, str):
            return {"state": value}

        return value

    def build_query_from_dict(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Elasticsearch Query DSL from structured dict input.

        - Exact match for keyword & numeric fields
        - match_phrase for text fields
        - JSON-string fields support:
            + raw string match
            + dict → converted to AND text query
        """

        must_clauses: List[Dict[str, Any]] = []

        for field, value in query_dict.items():

            # ---------- Keyword ----------
            if field in self.keyword_fields:
                must_clauses.append(self._term_query(field, value))

            # ---------- Numeric ----------
            elif field in self.numeric_fields:
                must_clauses.append(self._numeric_query(field, value))

            # ---------- Text ----------
            elif field in self.text_fields:
                must_clauses.append(self._match_phrase_query(field, value))

            # ---------- JSON String ----------
            elif field in self.json_string_fields:
                normalized_value = self._normalize_json_field_input(field, value)
                must_clauses.extend(
                    self._json_string_query(field, normalized_value)
                )

        return {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            }
        }

    def _term_query(self, field: str, value: Any) -> Dict[str, Any]:
        return {"term": {field: value}}

    def _numeric_query(self, field: str, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return {"range": {field: value}}
        return {"term": {field: value}}

    def _match_phrase_query(self, field: str, value: str) -> Dict[str, Any]:
        return {"match_phrase": {field: value}}

    def _json_string_query(self, field: str, value: Any) -> List[Dict[str, Any]]:
        clauses = []

        if isinstance(value, (int, float)):
            clauses.append({
                "match_phrase": {field: f": {value}"}
            })
            return clauses

        if isinstance(value, str):
            clauses.append({
                "match_phrase": {field: f'"{value}"'}
            })
            return clauses

        if isinstance(value, dict):
            for k, v in value.items():
                formatted_v = self._format_json_value(v)
                clauses.append({
                    "match_phrase": {
                        field: f'"{k}": {formatted_v}'
                    }
                })
            return clauses

        if isinstance(value, list):
            for item in value:
                clauses.extend(self._json_string_query(field, item))

        return clauses

    def _format_json_value(self, value: Any) -> str:
        """
        Format value to match raw JSON string in ES
        """
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def search(self, query: Dict[str, Any], size: int = 10, sort: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Perform a search in Elasticsearch using the provided query.

        :param query: Elasticsearch query DSL.
        :param size: Number of results to return.
        :param sort: Optional sort criteria (e.g., [{"frame_ts": {"order": "asc"}}]).
        :return: List of matching documents' _source.
        :raises RuntimeError: If search fails.
        """
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": query.get("query", {"match_all": {}}),
                    "size": size,
                    "sort": sort or []
                }
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except elastic_transport.TransportError as e:
            raise RuntimeError(f"Search error: {str(e)}") from e

    def search_by_dict(self, query_dict: Dict[str, Any], size: int = 10, sort_by_time: bool = False) -> List[Dict[str, Any]]:
        """
        Search using a query dictionary. Optionally sort by frame_ts ascending.

        :param query_dict: Input dictionary from chatbot.
        :param size: Number of results.
        :param sort_by_time: If True, sort by frame_ts ascending.
        :return: List of matching documents.
        """
        es_query = self.build_query_from_dict(query_dict)
        sort = [{"frame_ts": {"order": "asc"}}] if sort_by_time else None
        return self.search(es_query, size=size, sort=sort)

    def search_by_camera_id(self, camera_id: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by camera_id."""
        return self.search_by_dict({"camera_id": camera_id}, size=size)

    def search_by_frame_id(self, frame_id: int, size: int = 10) -> List[Dict[str, Any]]:
        """Search by frame_id."""
        return self.search_by_dict({"frame_id": frame_id}, size=size)

    def search_by_frame_ts(self, frame_ts: int, size: int = 10) -> List[Dict[str, Any]]:
        """Search by frame_ts."""
        return self.search_by_dict({"frame_ts": frame_ts}, size=size)

    def search_by_scale_factor(self, scale_factor: float, size: int = 10) -> List[Dict[str, Any]]:
        """Search by scale_factor."""
        return self.search_by_dict({"scale_factor": scale_factor}, size=size)

    def search_by_image_path(self, image_path: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by image_path."""
        return self.search_by_dict({"image_path": image_path}, size=size)

    def search_by_tracks(self, tracks: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by tracks JSON string (partial match)."""
        return self.search_by_dict({"tracks": tracks}, size=size)

    def search_by_actions(self, actions: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by actions JSON string."""
        return self.search_by_dict({"actions": actions}, size=size)

    def search_by_fire(self, fire: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by fire JSON string."""
        return self.search_by_dict({"fire": fire}, size=size)

    def search_by_fire_state(self, fire_state: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by fire state (extract from JSON)."""
        # Assuming fire is JSON string, use match_phrase for the state
        query = {"query": {"match_phrase": {"fire": fire_state}}}
        return self.search(query, size=size)

    def search_by_faces(self, faces: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by faces JSON string."""
        return self.search_by_dict({"faces": faces}, size=size)

    def search_by_counting(self, counting: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search by counting JSON string."""
        return self.search_by_dict({"counting": counting}, size=size)
    
    def search_by_confidence(self, field: str, confidence_range: Dict[str, float], size: int = 10) -> List[Dict[str, Any]]:
        """
        Support range query for confidence in JSON fields like tracks.confidence >= 0.9
        Uses script filter since JSON string.
        Example: field='tracks', confidence_range={'gte': 0.9}
        """
        query = {
            "query": {
                "script": {
                    "script": {
                        "source": f"""
                        def json = doc['{field}'].value;
                        // Parse JSON logic here, but ES script has limits on string parsing.
                        // For simplicity, assume we use regex or partial match, but true range needs numeric subfield.
                        // Recommend mapping confidence as numeric keyword in index for proper range.
                        """,  # Placeholder: Proper impl requires better mapping
                        "params": confidence_range
                    }
                }
            }
        }
        # Note: For accurate range, reindex with confidence as nested numeric.
        return self.search(query, size=size)

    def find_frames_with_multiple_tracks(self, min_tracks=3, size=10):
        query = {
            "query": {
                "range": {
                    "track_count": {
                        "gte": min_tracks
                    }
                }
            }
        }
        return self.search(query, size=size)

    def find_action_changes(self, action_sequence: List[str], time_threshold: int = 1000, max_docs: int = 10000) -> List[Dict[str, Any]]:
        """
        Find sequences where actions change in the specified order within time_threshold.
        Example: ['Walking', 'Standing', 'Sitting']
        This scans sorted docs and checks sequence.
        """
        query_dict = {}  # Add filters if needed
        es_query = self.build_query_from_dict(query_dict)
        scan_iter = scan(
            self.es,
            index=self.index_name,
            query={
                "_source": ["frame_ts", "actions", "track_id"],  # Assume actions per track
                "query": es_query["query"],
                "sort": [{"frame_ts": "asc"}]
            },
            size=1000
        )

        results = []
        cur_track_sequences: Dict[Any, List[Dict]] = {}  # Group by track_id

        for hit in scan_iter:
            src = hit["_source"]
            track_id = src.get("track_id")  # Assume track_id available
            action = json.loads(src["actions"]).get("label") if src.get("actions") else None
            if not action or not track_id:
                continue

            if track_id not in cur_track_sequences:
                cur_track_sequences[track_id] = []

            seq = cur_track_sequences[track_id]
            seq.append(src)

            # Check if sequence matches
            if len(seq) >= len(action_sequence):
                matched = True
                for i, expected in enumerate(action_sequence):
                    if json.loads(seq[-len(action_sequence) + i]["actions"]).get("label") != expected:
                        matched = False
                        break
                if matched and seq[-1]["frame_ts"] - seq[-len(action_sequence)]["frame_ts"] <= time_threshold:
                    results.extend(seq[-len(action_sequence):])
                    # Optionally reset or continue

        return results[:max_docs]

    def find_persistent_tracks(self, min_consecutive_frames: int = 10, threshold: int = 100, max_docs: int = 100_000) -> Dict[int, List[Dict[str, Any]]]:
        """
        Find tracks that appear in at least N consecutive frames.
        Groups by track_id and finds sequences.
        """
        query_dict = {}  # Add filters if needed
        es_query = self.build_query_from_dict(query_dict)
        scan_iter = scan(
            self.es,
            index=self.index_name,
            query={
                "_source": ["track_id", "frame_ts", "frame_id"],
                "query": es_query["query"],
                "sort": [{"track_id": "asc"}, {"frame_ts": "asc"}]
            },
            size=2000
        )

        persistent_tracks: Dict[int, List[Dict[str, Any]]] = {}
        cur_track: Optional[int] = None
        cur_seq: List[Dict[str, Any]] = []
        prev_time: Optional[int] = None

        for i, hit in enumerate(scan_iter):
            if i >= max_docs:
                break
            src = hit["_source"]
            track_id = src.get("track_id")
            t = src.get("frame_ts")

            if track_id != cur_track:
                if len(cur_seq) >= min_consecutive_frames:
                    persistent_tracks[cur_track] = cur_seq.copy()
                cur_seq = [src]
                cur_track = track_id
                prev_time = t
                continue

            if t - prev_time <= threshold:
                cur_seq.append(src)
            else:
                if len(cur_seq) >= min_consecutive_frames:
                    persistent_tracks[cur_track] = cur_seq.copy()
                cur_seq = [src]

            prev_time = t

        if len(cur_seq) >= min_consecutive_frames:
            persistent_tracks[cur_track] = cur_seq.copy()

        return persistent_tracks

    def find_tracks_across_cameras(self, min_cameras: int = 2) -> List[Dict[str, Any]]:
        """
        Find track_ids that appear in at least min_cameras distinct cameras.
        Uses aggregation.
        """
        query = {
            "query": {"match_all": {}},
            "size": 0,
            "aggs": {
                "by_track": {
                    "terms": {"field": "track_id", "size": 10000},
                    "aggs": {
                        "distinct_cameras": {
                            "cardinality": {"field": "camera_id"}
                        },
                        "having": {
                            "bucket_selector": {
                                "buckets_path": {"count": "distinct_cameras"},
                                "script": f"params.count >= {min_cameras}"
                            }
                        }
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=query)
        return [bucket["key"] for bucket in response["aggregations"]["by_track"]["buckets"]]

    def find_known_faces(self, size: int = 10) -> List[Dict[str, Any]]:
        """
        Find frames with faces where personal_id != "Unknown".
        """
        query = {
            "query": {
                "bool": {
                    "must_not": {
                        "match_phrase": {"faces": '"personal_id": "Unknown"'}
                    }
                }
            }
        }
        return self.search(query, size=size)

    def find_counting_changes(self, min_increase: int = 1, time_field: str = "frame_ts", threshold: int = 1000, max_docs: int = 10000) -> List[Dict[str, Any]]:
        """
        Find frames where count_in or count_out increased compared to previous.
        Scans sorted docs.
        """
        query_dict = {}  # Add filters
        es_query = self.build_query_from_dict(query_dict)
        scan_iter = scan(
            self.es,
            index=self.index_name,
            query={
                "_source": ["frame_ts", "counting"],
                "query": es_query["query"],
                "sort": [{time_field: "asc"}]
            },
            size=1000
        )

        results = []
        prev_count_in = None
        prev_count_out = None
        prev_src = None

        for hit in scan_iter:
            src = hit["_source"]
            counting = json.loads(src.get("counting", "{}"))
            count_in = counting.get("count_in", 0)
            count_out = counting.get("count_out", 0)

            if prev_count_in is not None and (count_in - prev_count_in >= min_increase or count_out - prev_count_out >= min_increase):
                results.append(src)

            prev_count_in = count_in
            prev_count_out = count_out
            prev_src = src

        return results
    
    def find_longest_sequence(
        self,
        query_dict: Dict[str, Any],
        time_field: str = "frame_ts",
        source_fields: Optional[List[str]] = ["camera_id", "frame_id", "frame_ts"],
        threshold: int = 100,
        max_docs: int = 100_000
    ) -> List[Dict[str, Any]]:
        """
        Find longest consecutive sequence based on time difference.
        :param query_dict: Filters
        :param source_fields: Fields to return (default: [camera_id, frame_id, frame_ts])
        :param threshold: Max diff to be considered continuous
        :param max_docs: Hard limit to protect memory
        """

        es_query = self.build_query_from_dict(query_dict)

        try:
            scan_iter = scan(
                self.es,
                index=self.index_name,
                query={
                    "_source": source_fields,
                    "query": es_query["query"],
                    "sort": [{time_field: "asc"}]
                },
                size=2000
            )

            max_seq: List[Dict[str, Any]] = []
            cur_seq: List[Dict[str, Any]] = []

            prev_time: Optional[int] = None

            for i, hit in enumerate(scan_iter):
                if i >= max_docs:
                    break

                src = hit["_source"]
                t = src.get(time_field)

                if t is None:
                    continue  # safety

                if prev_time is None or t - prev_time <= threshold:
                    cur_seq.append(src)
                else:
                    if len(cur_seq) > len(max_seq):
                        max_seq = cur_seq.copy()
                    cur_seq = [src]

                prev_time = t

            if len(cur_seq) > len(max_seq):
                max_seq = cur_seq.copy()

            return max_seq

        except elastic_transport.TransportError as e:
            raise RuntimeError(f"Sequence search error: {str(e)}") from e

    def aggregate_by_time(self, query_dict, time_field="frame_ts", interval=1000):
        es_query = self.build_query_from_dict(query_dict)
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": es_query["query"],
                    "size": 0,
                    "aggs": {
                        "by_time": {
                            "histogram": {
                                "field": time_field,
                                "interval": interval
                            }
                        }
                    }
                }
            )
            return response["aggregations"]["by_time"]
        except elastic_transport.TransportError as e:
            raise RuntimeError(f"Aggregation error: {str(e)}") from e

    def aggregate_by_frame(self, query_dict: Dict[str, Any], frame_field: str = "frame_id", min_doc_count: int = 1) -> Dict[str, Any]:
        """
        Aggregate results by frame_id using terms aggregation.

        :param query_dict: Query filters.
        :param frame_field: Field to aggregate on (e.g., 'frame_id').
        :param min_doc_count: Minimum document count for buckets.
        :return: Aggregation results.
        :raises RuntimeError: If aggregation fails.
        """
        es_query = self.build_query_from_dict(query_dict)
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": es_query.get("query", {"match_all": {}}),
                    "size": 0,
                    "aggs": {
                        "by_frame": {
                            "terms": {
                                "field": frame_field,
                                "min_doc_count": min_doc_count
                            }
                        }
                    }
                }
            )
            return response["aggregations"]["by_frame"]
        except elastic_transport.TransportError as e:
            raise RuntimeError(f"Aggregation error: {str(e)}") from e

    def aggregate_track_durations(self) -> Dict[str, Any]:
        """
        Group by track_id with min/max frame_ts.
        """
        query = {
            "query": {"match_all": {}},
            "size": 0,
            "aggs": {
                "by_track": {
                    "terms": {"field": "tracks.track_id"},
                    "aggs": {
                        "min_ts": {"min": {"field": "frame_ts"}},
                        "max_ts": {"max": {"field": "frame_ts"}}
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=query)
        return response["aggregations"]["by_track"]

    def refresh_index(self):
        """Refresh index to make recent changes visible."""
        self.es.indices.refresh(index=self.index_name)

    def export_results(self, results: List[Dict[str, Any]], filename: str, format: str = "csv"):
        """
        Export results to CSV or JSONL.
        """
        if format == "csv":
            if results:
                keys = results[0].keys()
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, keys)
                    writer.writeheader()
                    writer.writerows(results)
        elif format == "jsonl":
            with open(filename, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    def build_advanced_query(self, must: List[Dict] = [], should: List[Dict] = [], must_not: List[Dict] = []) -> Dict[str, Any]:
        """
        Support OR / NOT by building bool query with must/should/must_not.
        Example: must=[term], should=[match1, match2]
        """
        return {
            "query": {
                "bool": {
                    "must": must,
                    "should": should,
                    "must_not": must_not
                }
            }
        }