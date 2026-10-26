from typing import Protocol

from llm_backend_lab.domain.models import Conversation


class ConversationRepository(Protocol):
    def save(self, conversation: Conversation) -> Conversation:
        """id가 없으면 새 id를 붙여 저장하고, id가 채워진 객체를 돌려준다."""
        ...

    def get(self, conversation_id: int) -> Conversation | None: ...
