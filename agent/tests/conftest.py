import os

# 테스트는 .env 없이도 돌아야 한다(CI 포함). 앱 import 전에 필수 설정값을 채워 둔다.
# 실제 DB에 붙는 테스트가 생기면 이 값을 테스트용 DB 주소로 바꾼다.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("APP_ENV", "test")
