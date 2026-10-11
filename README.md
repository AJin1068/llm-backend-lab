# llm-backend-lab

LLM 서비스를 트래픽 · 장애 · 비용 관점에서 운영 가능하게 만드는 FastAPI 백엔드입니다.

## 목표

1. **트래픽**: 응답 캐시와 rate limit으로 같은 요청은 LLM을 다시 호출하지 않고, 몰리는 요청은 제어한다. (캐시 hit/miss 응답 시간을 측정해 기록)
2. **장애**: 공급자 장애나 지연이 서비스 장애로 번지지 않게 타임아웃 · 재시도 · 서킷 브레이커 · 폴백 모델을 적용한다. (장애 상황을 테스트로 재현)
3. **비용**: 사용자별 토큰 쿼터와 요청별 토큰 · 비용 기록으로 비용을 예측하고 통제한다.

## 실행

```bash
cp .env.example .env        # 다른 pc에선 .env에 POSTGRES_PASSWORD 채우기
docker compose up --build
```

- API 문서: http://localhost:8080/docs
- 헬스체크: http://localhost:8080/health

로컬 개발 (DB · Redis만 컨테이너로):

```bash
docker compose up -d postgres redis
uv sync
uv run uvicorn llm_backend_lab.main:app --reload --port 8080
```

테스트 · 린트:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 기술 선택 이유

- **uv**: 가상환경 · 의존성 · lock · Python 버전 관리를 한 도구로 처리해 재현 가능한 설치를 보장
- **ruff**: 린트와 포맷을 한 도구로, 설정은 pyproject.toml 한 곳에
- **Docker 레이어 분리**: 의존성 레이어를 먼저 두어 코드만 바뀌면 의존성 설치를 캐시로 건너뜀
- **docker compose + healthcheck**: DB가 접속 가능해진 뒤 API가 뜨도록 순서 보장, 로컬 인프라를 명령 하나로 재현
- **pgvector 이미지**: 별도 벡터 DB 없이 Postgres로 RAG까지 처리 (자세한 비교는 ADR 예정)