from dataclasses import dataclass
from datetime import datetime


@dataclass
class Conversation:
    title: str
    created_at: datetime
    # 저장 전에는 None, 저장소가 save할 때 채움 (JPA의 @GeneratedValue와 같은 아이디어)
    id: int | None = None


class ConversationNotFound(Exception):
    def __init__(self, conversation_id: int) -> None:
        super().__init__(f"conversation {conversation_id} not found")
        self.conversation_id = conversation_id
