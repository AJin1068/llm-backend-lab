from fastapi.testclient import TestClient
from llm_backend_lab.main import app

client = TestClient(app)

# 함수 이름이 test_로 시작하면 자동으로 테스트로 인식
def test_healthcheck():
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}