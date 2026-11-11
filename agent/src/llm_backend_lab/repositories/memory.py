from dataclasses import replace

from llm_backend_lab.domain.models import Conversation


class InMemoryConversationRepository:
    """dict에 저장하는 구현체. Protocol을 상속하지 않고 메서드 모양만 맞춘다."""

    def __init__(self) -> None:
        self._items: dict[int, Conversation] = {}
        self._next_id = 1

    def save(self, conversation: Conversation) -> Conversation:
        if conversation.id is None:
            # 원본을 바꾸지 않고 id만 채운 복사본을 만든다
            conversation = replace(conversation, id=self._next_id)
            self._next_id += 1
        assert conversation.id is not None
        self._items[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: int) -> Conversation | None:
        return self._items.get(conversation_id)
