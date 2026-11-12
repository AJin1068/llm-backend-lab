from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from taxsplit.core.config import Settings, get_settings
from taxsplit.llm.base import LLMClient
from taxsplit.llm.call_log import InMemoryLlmCallRepository, LlmCallRepository
from taxsplit.llm.factory import build_llm_client
from taxsplit.repositories.base import ConversationRepository
from taxsplit.repositories.memory import InMemoryConversationRepository
from taxsplit.services.conversation_service import ConversationService


@lru_cache
def get_conversation_repo() -> ConversationRepository:
    # 앱 전체에 하나 (싱글톤). 다음 시간에 SQLAlchemy 구현으로 바뀌는 곳은 여기뿐
    return InMemoryConversationRepository()


def get_conversation_service(
    repo: Annotated[ConversationRepository, Depends(get_conversation_repo)],
) -> ConversationService:
    # 요청마다 새로 만듦. 가벼운 객체라 비용이 거의 없음
    return ConversationService(repo)


@lru_cache
def get_llm_call_repo() -> LlmCallRepository:
    # PR 8에서 Postgres 구현으로 바뀌는 곳은 여기뿐
    return InMemoryLlmCallRepository()


@lru_cache
def get_llm_client() -> LLMClient:
    # 앱 전체에 하나. SDK 클라이언트가 연결을 재사용하게 요청마다 새로 만들지 않는다.
    return build_llm_client(get_settings(), get_llm_call_repo())


SettingsDep = Annotated[Settings, Depends(get_settings)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
