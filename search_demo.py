import json
from src.components.retrieval.es_search import ElasticsearchSearcher

if __name__ == "__main__":
    searcher = ElasticsearchSearcher()

    sample_input = {
        "camera_id": "video.raw.hcm_thanh",
        "tracks": 3
    }

    # Basic search
    results = searcher.search_by_dict(sample_input)
    print("Search Results:", json.dumps(results, indent=2))

    # Longest sequence
    seq = searcher.find_longest_sequence(
        {"tracks": 3},
        source_fields=["frame_id", "frame_ts", "image_path"]
    )
    # seq = searcher.find_longest_sequence(
    #     {"tracks": {"track_id": 3}},)
    print("Longest Sequence:", json.dumps(seq, indent=2))

    agg_time = searcher.aggregate_by_time(sample_input)
    print("Time Aggregation:", json.dumps(agg_time, indent=2))

    multi_tracks = searcher.find_frames_with_multiple_tracks(min_tracks=3)
    print("Frames with multiple tracks:", json.dumps(multi_tracks, indent=2))