from pydantic import BaseModel


class CollectionCreate(BaseModel):
    name: str