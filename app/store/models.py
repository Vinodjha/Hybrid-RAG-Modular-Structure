from pydantic import BaseModel

class Query(BaseModel):
    query: str
    session_id: str = "default"
    max_answer_tokens:int | None = 3500
