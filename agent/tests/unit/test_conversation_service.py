import pytest

from llm_backend_lab.domain.models import ConversationNotFound
from llm_backend_lab.repositories.memory import InMemoryConversationRepository
from llm_backend_lab.services.conversation_service import ConversationService


@pytest.fixture
def service() -> ConversationService:
    return ConversationService(InMemoryConversationRepository())


def test_create_assigns_id(service: ConversationService) -> None:
    conversation = service.create("hello")

    assert conversation.id == 1
    assert conversation.title == "hello"


def test_get_returns_created(service: ConversationService) -> None:
    created = service.create("hello")

    assert service.get(1) == created


def test_get_missing_raises(service: ConversationService) -> None:
    with pytest.raises(ConversationNotFound) as exc_info:
        service.get(999)

    assert exc_info.value.conversation_id == 999
