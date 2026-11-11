FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

WORKDIR /app

# 의존성 레이어(자주 안바뀜): 외부 라이브러리 설치. 의존성 파일이 바뀌지 않으면 캐시 재사용.
COPY pyproject.toml uv.lock ./
# 두 파일이 그대로면 캐시
RUN uv sync --frozen --no-dev --no-install-project 

# 소스 레이어(자주 바뀜): 코드를 패키지화.
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev

# 가상환경의 실행 파일(uvicorn 등)을 PATH에 추가
ENV PATH="/app/.venv/bin:$PATH"

# 0.0.0.0: 이걸 빼면 컨테이너 안의 localhost에서만 열려서 밖에서 접속이 안됨.
CMD ["uvicorn", "llm_backend_lab.main:app", "--host", "0.0.0.0", "--port", "8000"]
