from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from llm_backend_lab.core.config import Settings, get_settings
from llm_backend_lab.repositories.base import ConversationRepository
from llm_backend_lab.repositories.memory import InMemoryConversationRepository
from llm_backend_lab.services.conversation_service import ConversationService


@lru_cache
def get_conversation_repo() -> ConversationRepository:
    # 앱 전체에 하나 (싱글톤). 다음 시간에 SQLAlchemy 구현으로 바뀌는 곳은 여기뿐
    return InMemoryConversationRepository()


def get_conversation_service(
    repo: Annotated[ConversationRepository, Depends(get_conversation_repo)],
) -> ConversationService:
    # 요청마다 새로 만듦. 가벼운 객체라 비용이 거의 없음
    return ConversationService(repo)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
