from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreateConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    # dataclass 같은 일반 객체에서 속성을 읽어 응답을 만들 수 있게 함
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
