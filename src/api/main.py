from fastapi import FastAPI
from botocore.exceptions import EndpointConnectionError
from api.core.minio import get_s3_client
from api.routers import health, chat, collections, media
from api.core.exceptions import minio_exception_handler

app = FastAPI(title="Chatbot API")

@app.on_event("startup")
async def startup():
    app.state.minio_client = get_s3_client()

app.add_exception_handler(
    EndpointConnectionError,
    minio_exception_handler,
)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(media.router)
app.include_router(collections.router)