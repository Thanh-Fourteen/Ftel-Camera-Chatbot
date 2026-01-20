from pydantic import BaseModel

class MediaRequest(BaseModel):
    path: str
