from fastapi import APIRouter, status

from llm_backend_lab.api.deps import ConversationServiceDep
from llm_backend_lab.api.schemas import ConversationResponse, CreateConversationRequest

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest, service: ConversationServiceDep
) -> ConversationResponse:
    conversation = service.create(body.title)
    return ConversationResponse.model_validate(conversation)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: int, service: ConversationServiceDep
) -> ConversationResponse:
    conversation = service.get(conversation_id)
    return ConversationResponse.model_validate(conversation)
