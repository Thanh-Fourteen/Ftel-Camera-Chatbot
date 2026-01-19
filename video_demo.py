from src.components.retrieval.es_search import ElasticsearchSearcher
from src.components.reasoning.media import MediaProcessor

if __name__ == "__main__":
    searcher = ElasticsearchSearcher()

    query = {
        "actions": {
            # "track_id": 3,
            "label": ["Fall Down", "Lying Down"]
        }
    }

    docs = searcher.search_by_dict(
        query,
        size=10000,
    )

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
        image_bucket="camera-frames",
        video_bucket="camera-videos",
        access_key="minioadmin",
        secret_key="minioadmin",
        fps=25,
    )

    results = media.build_videos_from_sequences(sequences)

    for r in results:
        print(r)

    # check video in: http://192.168.2.21:9001/browser/camera-videos