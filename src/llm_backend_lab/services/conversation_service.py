from datetime import UTC, datetime

from llm_backend_lab.domain.models import Conversation, ConversationNotFound
from llm_backend_lab.repositories.base import ConversationRepository


class ConversationService:
    """비즈니스 로직. FastAPI와 저장소 구현체를 import하지 않는다."""

    def __init__(self, repo: ConversationRepository) -> None:
        self._repo = repo

    def create(self, title: str) -> Conversation:
        return self._repo.save(Conversation(title=title, created_at=datetime.now(UTC)))

    def get(self, conversation_id: int) -> Conversation:
        conversation = self._repo.get(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)
        return conversation
