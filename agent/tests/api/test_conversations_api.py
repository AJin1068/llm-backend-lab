from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from llm_backend_lab.api.deps import get_conversation_repo
from llm_backend_lab.main import app
from llm_backend_lab.repositories.memory import InMemoryConversationRepository


@pytest.fixture
def client() -> Iterator[TestClient]:
    # 테스트마다 새 저장소로 교체 (Spring의 @MockBean과 같은 역할)
    repo = InMemoryConversationRepository()
    app.dependency_overrides[get_conversation_repo] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_then_get(client: TestClient) -> None:
    created = client.post("/conversations", json={"title": "hello"})
    assert created.status_code == 201
    body = created.json()
    assert body["id"] == 1
    assert body["title"] == "hello"

    fetched = client.get(f"/conversations/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_get_missing_returns_404(client: TestClient) -> None:
    response = client.get("/conversations/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation 999 not found"}


def test_create_rejects_empty_title(client: TestClient) -> None:
    response = client.post("/conversations", json={"title": ""})

    assert response.status_code == 422
