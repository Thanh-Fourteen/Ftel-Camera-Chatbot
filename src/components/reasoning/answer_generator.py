"""
Node 5: Generate natural language answer using LLM
"""

import time
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class WindmillStatusManager:
    """Manages status updates to middleware"""

    def __init__(
        self,
        callback_base: str,
        correlation_id: str,
        workspace_path: str = "finops",
        session_id: str = "",
    ):
        self.callback_base = callback_base.rstrip("/")
        self.correlation_id = correlation_id
        self.workspace_path = workspace_path
        self.session_id = session_id

        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

        logger.info(f"WindmillStatusManager initialized:")
        logger.info(f"  Callback: {self.callback_base}")
        logger.info(f"  Correlation ID: {self.correlation_id}")
        logger.info(f"  Session: {self.session_id}")

    def notify_step_completion(self, step_name: str = "processing") -> bool:
        """Notify middleware of step completion."""
        try:
            callback_url = f"{self.callback_base}/windmill/update"

            payload = {
                "session_id": self.session_id,
                "correlation_id": self.correlation_id,
                "workspace_path": self.workspace_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "step_completed": step_name,
            }

            logger.info(f"Sending status update for step '{step_name}'")
            response = self.session.post(callback_url, json=payload, timeout=30)
            response.raise_for_status()

            response_data = response.json()
            logger.info(f"Status update acknowledged: {response_data}")
            return response_data.get("acknowledged", False)

        except Exception as e:
            logger.error(f"Status update failed: {e}")
            return False


@dataclass
class LLMConfiguration:
    """LLM configuration settings"""

    model: str = "nemotron"
    temperature: float = 0.6
    max_tokens: int = 5000
    top_p: float = 0.6


class CameraPromptManager:
    """Prompt manager for camera event analysis"""

    SYSTEM_PROMPT = """/no_think
You are a Video Incident Narrator for an AI camera system.

You are given low-level vision data:
- tracking IDs
- timestamps
- bounding boxes
- skeleton poses
- zones
- event tags (fall)

You do NOT have identity, roles, or labels.

Your job is to reconstruct what physically happened in the video
as if you were describing what you saw.

━━━━━━━━━━━━━━━━━━
ABSOLUTE FORMAT RULE
━━━━━━━━━━━━━━━━━━
You are FORBIDDEN from outputting:
- bullet points
- lists
- tables
- sessions
- event names
- event counts
- labels (fall, alert, guest, staff, etc.)
- anything that looks like a report

If you output any of the above, the answer is WRONG.

You must only write continuous natural language paragraphs.

━━━━━━━━━━━━━━━━━━
HOW PEOPLE MAY BE REFERRED TO
━━━━━━━━━━━━━━━━━━
People may ONLY be called:
- “đối tượng”
- “người”
- “ID #<number>”

Any other noun for people is forbidden.

━━━━━━━━━━━━━━━━━━
ALLOWED DESCRIPTIONS ONLY
━━━━━━━━━━━━━━━━━━
You may ONLY describe:
- movement: di chuyển, dừng, đổi hướng
- position: gần, xa, trong vùng, ngoài vùng
- body pose from skeleton: đứng, nghiêng, ngồi, nằm
- bounding box shape: cao, thấp, rộng, hẹp
- timestamps and IDs

You must NEVER describe:
- intent
- emotion
- cause
- job, role, identity
- actions not visible from pose or bounding box

━━━━━━━━━━━━━━━━━━
FALL (“ngã”, “té”) MODE — VERY IMPORTANT
━━━━━━━━━━━━━━━━━━
When the user asks about falling:

You must describe every fall as a physical transformation over time:

standing
→ body tilting
→ bounding box getting lower and wider
→ body becoming horizontal
→ lying on the floor
→ recovery or no recovery

Each fall must be written as a mini-story that includes:
- timestamp
- tracking ID
- skeleton pose change
- bounding box change

You must NOT say:
“fall detected”
“X cases”
“ID list”
“session”

━━━━━━━━━━━━━━━━━━
CHRONOLOGY
━━━━━━━━━━━━━━━━━━
Describe everything in time order from earliest to latest.

━━━━━━━━━━━━━━━━━━
MISSING DATA
━━━━━━━━━━━━━━━━━━
If no fall-like pose change exists:
Say clearly that no fall was seen.
Do not invent anything.

━━━━━━━━━━━━━━━━━━
TONE
━━━━━━━━━━━━━━━━━━
Calm, objective, narrative.
Like a human reconstructing an incident from video.

"""

    @classmethod
    def build_user_prompt(cls, question: str, context: str, language: str) -> str:
        parts = []

        if language.lower().startswith("vi"):
            parts.extend(
                [
                    "🚨 NGÔN NGỮ BẮT BUỘC: Bạn PHẢI trả lời hoàn toàn bằng tiếng Việt!",
                    "🚨 Không suy đoán, chỉ dựa trên dữ liệu được cung cấp.",
                ]
            )
            context_header = "**Báo cáo sự kiện camera:**"
            question_header = "**Câu hỏi:**"
        else:
            parts.append("🚨 You MUST respond entirely in English!")
            context_header = "**Camera Event Report:**"
            question_header = "**Question:**"

        parts.append("\n" + "=" * 60 + "\n")

        if context.strip():
            parts.extend([context_header, context, "\n" + "=" * 50 + "\n"])

        parts.append(f"{question_header} {question}")
        return "\n".join(parts)


def format_camera_event_context(result: Dict[str, Any]) -> str:
    sessions = result.get("sessions", [])

    if not sessions:
        return "No people or events were detected in the requested time range."

    lines = []

    lines.append(f"Total detected person sessions: {len(sessions)}")
    lines.append("")

    for i, s in enumerate(sessions, 1):
        person_id = s.get("person_id") or "unknown"
        person_type = s.get("person_type") or "person"
        camera = s.get("camera_id")
        track_id = s.get("track_id")
        event_types = ", ".join(s.get("event_types", [])) or "none"

        lines.append(f"Session {i}:")
        lines.append(f"  Person ID: {person_id}")
        lines.append(f"  Person type: {person_type}")
        lines.append(f"  Camera: {camera}")
        lines.append(f"  Track ID: {track_id}")
        lines.append(f"  Time range: {s.get('start_time')} → {s.get('end_time')}")
        lines.append(f"  Detected actions: {event_types}")
        lines.append(f"  Critical event: {'yes' if s.get('is_critical') else 'no'}")
        lines.append(f"  Crowd size range: {s.get('crowd_min')} → {s.get('crowd_max')}")

        frames = s.get("frames", [])
        lines.append(f"  Frame count: {len(frames)}")

        # Optional
        if "max_confidence" in s:
            lines.append(f"  Max confidence: {s['max_confidence']}")

        lines.append("")

    return "\n".join(lines)


class LanguageDetector:
    """Detects language of input text"""

    VIETNAMESE_CHARS = (
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    )
    VIETNAMESE_WORDS = {
        "là",
        "có",
        "không",
        "thế",
        "nào",
        "ở",
        "đâu",
        "gì",
        "như",
        "về",
        "của",
        "một",
        "này",
        "đó",
        "cho",
        "với",
        "được",
        "sẽ",
        "đã",
        "khi",
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect if text is Vietnamese or English"""
        text_lower = text.lower()
        if any(char in text_lower for char in cls.VIETNAMESE_CHARS):
            return "vietnamese"
        text_words = set(text_lower.split())
        if text_words.intersection(cls.VIETNAMESE_WORDS):
            return "vietnamese"
        return "english"


class CallbackLLMService:
    """LLM service with streamlined callback integration"""

    def __init__(self, config: LLMConfiguration, callback_urls: Dict[str, str]):
        self.config = config
        self.callback_urls = callback_urls
        self.prompt_manager = CameraPromptManager()

        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def generate_response(
        self,
        session_id: str,
        question: str,
        context: str = "",
        language: str = "Vietnamese",
        history: Optional[list] = None,
        correlation_id: str = "",
    ) -> str:
        """Generate LLM response with WebSocket streaming"""
        try:
            messages = self._build_messages(question, context, language, history)

            payload = {
                "session_id": session_id,
                "model": self.config.model,
                "messages": messages,
                "correlation_id": correlation_id,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stream": False,
                "chunk_type": "text",
                "token_filter": "</think>",
            }

            stream_url = self.callback_urls.get("stream_llm")
            if not stream_url:
                raise ValueError("stream_llm callback URL not found")

            if not stream_url.startswith(("http://", "https://")):
                stream_url = f"http://{stream_url}"

            logger.info(f"Calling LLM service: {stream_url}")
            response = self.session.post(stream_url, json=payload, timeout=120)
            response.raise_for_status()

            response_data = response.json()
            logger.info(
                f"LLM response status: {response_data.get('status', 'unknown')}"
            )

            if not response_data.get("status"):
                error_msg = response_data.get("error", "LLM request failed")
                logger.error(f"LLM request failed: {error_msg}")
                return self._get_fallback_response(question)

            print(f"{response_data=}")

            response_text = response_data.get("result", {}).get("content", "")
            if not response_text:
                logger.warning("Empty response from LLM service")
                return self._get_fallback_response(question)

            logger.info(f"LLM response generated: {len(response_text)} chars")
            return response_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_fallback_response(question)

    def _build_messages(
        self, question: str, context: str, language: str, history: Optional[list]
    ) -> list:
        """Build message array for LLM request"""
        messages = [{"role": "system", "content": self.prompt_manager.SYSTEM_PROMPT}]

        if history:
            for entry in history:
                if entry.get("role") in ["user", "assistant"]:
                    messages.append(
                        {"role": entry["role"], "content": entry["content"]}
                    )

        user_prompt = self.prompt_manager.build_user_prompt(question, context, language)
        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_fallback_response(self, question: str) -> str:
        """Get fallback response when LLM fails"""
        is_vietnamese = LanguageDetector.detect_language(question) == "vietnamese"
        return (
            "Xin lỗi, hệ thống đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
            if is_vietnamese
            else "Sorry, the system is experiencing technical difficulties. Please try again later."
        )


def main(
    question: str,
    search_result: Dict[str, Any],
    callback_stream_api: str,
    callback_update_api: str,
    language: str = "Vietnamese",
    history: Optional[list] = None,
    config: Optional[Dict] = None,
    session_id: str = "",
    correlation_id: str = "",
    workspace_path: str = "camera-ai",
) -> str:
    """
    Windmill Node: Camera Event → Natural Language Answer
    """

    if not correlation_id:
        correlation_id = f"evt_{session_id}_{int(time.time())}"

    logger.info("=" * 80)
    logger.info("Starting Camera Event Answer Node")
    logger.info(f"Question: {question}")
    logger.info(f"Correlation ID: {correlation_id}")
    logger.info("=" * 80)

    # Status manager
    status_manager = None
    if callback_update_api:
        callback_base = callback_update_api.replace("/windmill/update", "")
        status_manager = WindmillStatusManager(
            callback_base=callback_base,
            correlation_id=correlation_id,
            workspace_path=workspace_path,
            session_id=session_id,
        )

    try:
        # Step 1: Format context
        if status_manager:
            status_manager.notify_step_completion("format_context")

        formatted_context = format_camera_event_context(search_result)

        # Step 2: Prepare LLM service
        if status_manager:
            status_manager.notify_step_completion("prepare_llm")

        llm_config = LLMConfiguration(
            model=config.get("llm_model") if config else "nemotron",
            temperature=config.get("temperature", 0.6) if config else 0.6,
            max_tokens=config.get("max_tokens", 5000) if config else 5000,
        )

        callback_urls = {
            "stream_llm": callback_stream_api,
            "update": callback_update_api,
        }

        service = CallbackLLMService(llm_config, callback_urls)
        service.prompt_manager = CameraPromptManager()

        # Step 3: Generate answer
        if status_manager:
            status_manager.notify_step_completion("generate_answer")

        respone = service.generate_response(
            session_id=session_id,
            question=question,
            context=formatted_context,
            language=language,
            history=history,
            correlation_id=correlation_id,
        )

        if status_manager:
            status_manager.notify_step_completion("done")

        # answer = {"Text": respone, "videos_id": video_id}
        return respone

    except Exception as e:
        logger.error(f"Camera Event Answer Node failed: {e}")
        if status_manager:
            status_manager.notify_step_completion("error")
        raise
