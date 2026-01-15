"""
Node 1: Parse chatbot question into structured intent using Nemotron (callback LLM)
"""

import time
import re
import logging
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field, TypeAdapter

logger = logging.getLogger(__name__)

# ======================================================
# INTENT SCHEMA
# ======================================================


class QueryIntent(BaseModel):
    camera_id: str | None = Field(None)
    event_type: str | None = Field(None)
    action_label: str | None = Field(None)
    person_type: str | None = Field(None)
    track_id: int | None = Field(None)
    time_range_description: str | None = Field(None)
    confidence_threshold: float = 0.7


intent_adapter = TypeAdapter(QueryIntent)

# ======================================================
# PROMPT
# ======================================================

SYSTEM_PROMPT = """/no_think
You are an expert intent parser for a video surveillance chatbot.

You must extract ONLY what is explicitly stated by the user.
Do not guess.

You MUST output valid JSON matching this schema:

{
  "camera_id": string | null,
  "event_type": "action" | "person" | null,
  "action_label": string | null,
  "person_type": "employee" | "visitor" | null,
  "track_id": integer | null,
  "time_range_description": string | null,
  "confidence_threshold": 0.7
}

Rules:
- camera_id: keep original format ("camera 1", "cửa chính", "cam A")
- time_range_description: keep natural language ("10 phút", "sáng nay", "hôm qua")
- action_label:
  ngã, té, fall → fall
  chạy, running → running
- person_type:
  nhân viên → employee
  khách → visitor
- event_type:
  action if action_label present
  person if person_type present

Return JSON only. No text. No markdown.
"""

# ======================================================
# NEMOTRON CLIENT
# ======================================================


class NemotronClient:
    def __init__(self, stream_url: str, model="nemotron"):
        self.url = (
            stream_url if stream_url.startswith("http") else f"http://{stream_url}"
        )
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def infer(self, messages: list, correlation_id: str):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 5000,
            "top_p": 0.6,
            "stream": False,
            "chunk_type": "text",
            "correlation_id": correlation_id,
            "token_filter": "</think>",
        }

        resp = self.session.post(self.url, json=payload, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        if not data.get("status"):
            raise RuntimeError(data.get("error", "LLM error"))

        return data["result"]["content"]


# ======================================================
# FALLBACK REGEX
# ======================================================


def fallback_regex_parsing(question: str) -> dict:
    q = question.lower()

    camera = re.search(r"(camera|cam)\s*\d+|cửa\s*\w+|lối\s*\w+", q)
    time = re.search(
        r"\d+\s*(phút|giờ)|sáng nay|hôm nay|hôm qua|buổi\s*(sáng|chiều|tối)", q
    )

    intent = {
        "camera_id": camera.group(0) if camera else None,
        "event_type": None,
        "action_label": None,
        "person_type": None,
        "track_id": None,
        "time_range_description": time.group(0) if time else None,
        "confidence_threshold": 0.7,
    }

    if "ngã" in q or "té" in q or "fall" in q:
        intent["event_type"] = "action"
        intent["action_label"] = "fall"

    if "chạy" in q or "running" in q:
        intent["event_type"] = "action"
        intent["action_label"] = "running"

    if "nhân viên" in q:
        intent["event_type"] = "person"
        intent["person_type"] = "employee"

    if "khách" in q:
        intent["event_type"] = "person"
        intent["person_type"] = "visitor"

    return intent


# ======================================================
# PARSER
# ======================================================


def extract_json(text: str) -> str:
    """
    Extract first valid JSON object from LLM output.
    Handles cases like:
    'Sure! Here is the JSON:\n{ ... }\nThanks'
    """
    # Remove think blocks if any
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    return match.group(0)


def parse_intent_with_nemotron(
    question: str, stream_llm_url: str, session_id: str
) -> dict:
    correlation_id = f"intent_{session_id}_{int(time.time())}"

    client = NemotronClient(stream_llm_url)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]

    try:
        raw = client.infer(messages, correlation_id)
        logger.info(f"Nemotron raw intent: {raw}")

        json_text = extract_json(raw)
        logger.info(f"Extracted JSON: {json_text}")

        intent = intent_adapter.validate_json(json_text)
        return intent.model_dump()

    except Exception as e:
        logger.error(f"Nemotron intent parsing failed: {e}")
        return fallback_regex_parsing(question)


# ======================================================
# MAIN
# ======================================================


def main(
    normalized_input: Dict[str, Any],
    callback_stream_api: str,
) -> Dict[str, Any]:
    """
    Windmill Node 1 — Nemotron Intent Parser
    """

    question = normalized_input["question"]
    session_id = normalized_input.get("session_id", "")

    intent = parse_intent_with_nemotron(
        question=question,
        stream_llm_url=callback_stream_api,
        session_id=session_id,
    )

    logger.info("Parsed intent | %s", intent)
    return intent
