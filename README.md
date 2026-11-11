# taxsplit

맞벌이 부부의 연말정산 자료로 공제를 누구에게 넣을지 계산하고, 근거 조문과 함께 설명하는 LLM 에이전트 서비스입니다.
세액 계산은 코드가, 대화와 설명은 LLM이, 근거는 검색(RAG)이 맡습니다.

## 구조

| 폴더 | 언어 | 역할 |
| --- | --- | --- |
| `agent/` | Python (FastAPI) | 세액 계산 엔진, 도구 호출 에이전트, 평가 실행 |
| `gateway/` | Kotlin (Spring) | LLM 게이트웨이. 캐시, 레이트 리밋, 장애 대응, 비용 기록 (11/23부터) |
| `eval/` | - | 채점 규칙, 채점 코드, 정답 데이터. 개발 PR에서는 고칠 수 없음 |

## 목표

1. **정확도**: LLM이 계산하지 않는다. 세액 계산 엔진을 국세청 공식 계산 사례로 검증하고, 검증된 엔진으로 기준 정답을 만들어 에이전트를 채점한다.
2. **신뢰성**: 도구 호출 인자를 검사하고, 실패하면 이유를 모델에 돌려줘 재시도하며, 최대 스텝을 넘으면 멈춘다. 평가 하네스로 전후를 숫자로 비교한다.
3. **운영**: Kotlin 게이트웨이에서 레이트 리밋, 재시도·타임아웃·서킷 브레이커·폴백, 호출별 토큰·비용 기록을 처리한다. 장애를 일부러 내서 평가 하네스로 성공률을 잰다.

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
cd agent
uv sync
uv run uvicorn taxsplit.main:app --reload --port 8080
```

테스트 · 린트 (`agent/`에서):

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
- **한 레포 두 폴더(agent/, gateway/)**: 빌드 도구를 분리하면서도 PR 하나로 양쪽을 고치고, compose 한 번으로 데모 전체를 띄움
- **LangChain 미사용**: 에이전트 루프, 도구 호출 검사, 평가를 SDK로 직접 구현해 설계를 직접 설명할 수 있게 함
