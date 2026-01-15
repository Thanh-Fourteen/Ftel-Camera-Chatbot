import logging
from typing import Dict, Any

from src.components.input.normalizer import InputNormalizer
from src.components.input.intent_parser import NemotronClient, IntentParser
from src.components.retrieval.query_builder import ElasticsearchQueryBuilder
from src.components.retrieval.search_engine import ElasticsearchSearchEngine
from src.components.retrieval.session_aggregator import PersonSessionAggregator
from src.components.reasoning.video_builder import main as build_videos
from src.components.reasoning.answer_generator import main as llm_answer


logger = logging.getLogger("CAMERA_PIPELINE")
logging.basicConfig(level=logging.INFO)


def run_pipeline(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    FULL CAMERA AI PIPELINE
    """

    logger.info("🚀 Starting Camera AI Pipeline")

    # ------------------------------------------------------------------
    # 1. Normalize input
    # ------------------------------------------------------------------    
    normalizer = InputNormalizer()
    normalized = normalizer.normalize(raw_input)

    question = normalized["question"]
    session_id = normalized["session_id"]
    correlation_id = normalized["correlation_id"]
    language = normalized["language"]
    history = normalized["history"]

    callback_stream = normalized["callback"]["stream"]
    callback_update = normalized["callback"]["update"]

    # ------------------------------------------------------------------
    # 2. Parse intent (Text -> json)
    # ------------------------------------------------------------------
    llm = NemotronClient(callback_stream)
    parser = IntentParser(llm)

    intent = parser.parse(
        question=normalized["question"],
        session_id=normalized["session_id"]
    )

    # ------------------------------------------------------------------
    # 3. Build Elasticsearch Query
    # ------------------------------------------------------------------
    query_builder = ElasticsearchQueryBuilder()
    es_query = query_builder.build(intent)
    # ------------------------------------------------------------------
    # 4. Search in Elasticsearch
    # ------------------------------------------------------------------
    search_engine = ElasticsearchSearchEngine(
        host="http://192.168.2.41:19200",
        index="camera-events",
    )

    es_result = search_engine.search(es_query)
    # ------------------------------------------------------------------
    # 5. Aggregate frames → person sessions
    # ------------------------------------------------------------------
    aggregator = PersonSessionAggregator(frame_gap=30)
    session_result = aggregator.aggregate(es_result)

    # ------------------------------------------------------------------
    # 6. Build videos from frames
    # ------------------------------------------------------------------
    video_ids = build_videos(session_result)
    # ------------------------------------------------------------------
    # 7. Generate LLM Answer
    # ------------------------------------------------------------------
    answer_text = llm_answer(
        question=question,
        search_result=session_result,
        callback_stream_api=callback_stream,
        callback_update_api=callback_update,
        language=language,
        history=history,
        session_id=session_id,
        correlation_id=correlation_id,
        config={
            "llm_model": "nemotron",
            "temperature": 0.6,
            "max_tokens": 5000,
        },
    )

    response = {
        "session_id": session_id,
        "correlation_id": correlation_id,
        "intent": intent,
        "es_query": es_query,
        "total_person_sessions": session_result["total_person_sessions"],
        "videos": video_ids,
        "answer": answer_text,
    }

    logger.info("Camera AI Pipeline Completed")
    return response


if __name__ == "__main__":
    test_input = {
        "question": "Có ai bị ngã không?",
        "session_id": "sess_001",
        "correlation_id": "corr_001",
        "language": "vi",
        "history": [],
        "callback_stream_api": "https://chatproto.pythera.ai/windmill/stream-llm",
        "callback_update_api": "https://chatproto.pythera.ai/windmill/update",
    }

    result = run_pipeline(test_input)

    print("\n========== FINAL OUTPUT ==========")
    print(result)
