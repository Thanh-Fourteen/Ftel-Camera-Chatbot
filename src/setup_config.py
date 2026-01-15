import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _normalize_history(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for item in history or []:
        if item.get("role") in ("user", "assistant") and isinstance(
            item.get("content"), str
        ):
            content = item["content"].strip()
            if content:
                normalized.append({"role": item["role"], "content": content})
    return normalized


def main(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 0: Normalize chatbot input (ALWAYS receive raw_input)
    """

    # ---------- Extract ----------
    question = raw_input.get("question")
    session_id = raw_input.get("session_id")
    correlation_id = raw_input.get("correlation_id")
    language = raw_input.get("language", "vi")
    history = raw_input.get("history", [])
    callback_stream_api = raw_input.get("callback_stream_api", "")
    callback_update_api = raw_input.get("callback_update_api", "")

    # ---------- Validate ----------
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing or invalid 'question'")

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Missing or invalid 'session_id'")

    if not isinstance(correlation_id, str) or not correlation_id.strip():
        raise ValueError("Missing or invalid 'correlation_id'")

    # ---------- Normalize ----------
    normalized_history = _normalize_history(history)

    normalized = {
        "question": question.strip(),
        "session_id": session_id.strip(),
        "correlation_id": correlation_id.strip(),
        "language": language if language in ("vi", "en") else "vi",
        "history": normalized_history,
        "callback": {
            "stream": callback_stream_api or "",
            "update": callback_update_api or "",
        },
        "meta": {
            "turn_count": len(normalized_history),
            "has_history": bool(normalized_history),
        },
    }

    logger.info(
        "Normalized input session=%s correlation=%s",
        normalized["session_id"],
        normalized["correlation_id"],
    )

    return normalized
