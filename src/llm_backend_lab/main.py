from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from llm_backend_lab.api.routes import conversations
from llm_backend_lab.core.config import get_settings
from llm_backend_lab.domain.models import ConversationNotFound


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(conversations.router)
    get_settings()

    # 도메인 예외 → HTTP 응답 변환. Spring의 @ControllerAdvice 역할
    @app.exception_handler(ConversationNotFound)
    async def handle_not_found(request: Request, exc: ConversationNotFound) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
