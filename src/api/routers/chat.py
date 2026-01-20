from fastapi import APIRouter
from api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return ChatResponse(
        reply=f"Chatbot trả lời: {req.text}"
    )
