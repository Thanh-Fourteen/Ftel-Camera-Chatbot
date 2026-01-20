import time
from src.components.retrieval.es_search import ElasticsearchSearcher
from src.components.reasoning.media import MediaProcessor

if __name__ == "__main__":
    searcher = ElasticsearchSearcher()

    query = {
        "actions": {
            "track_id": 3,
            "label": ["Fall Down", "Lying Down"]
        }
    }

    start_search = time.perf_counter()
    docs = searcher.search_by_dict(
        query,
        size=10000,
    )
    end_search = time.perf_counter()
    print("Time search local: ", end_search - start_search)

    sequences = searcher.split_into_sequences_from_docs(
        docs,
        time_field="frame_ts",
        threshold=100,
        min_length=2
    )

    if not sequences:
        print("No valid sequence")
        exit(0)

    media = MediaProcessor(
        minio_endpoint="http://192.168.2.21:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        fps=25,
    )

    start_media = time.perf_counter()
    results = media.build_videos_from_sequences(sequences)
    end_media = time.perf_counter()
    print("Time media local: ", end_media - start_media)

    for r in results:
        print(r)

    # check video in: http://192.168.2.21:9001/browser/camera-videos