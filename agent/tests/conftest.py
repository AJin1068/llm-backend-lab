import os
import socket

import pytest

# 테스트는 .env 없이도 돌아야 한다(CI 포함). 앱 import 전에 필수 설정값을 채워 둔다.
# 실제 DB에 붙는 테스트가 생기면 이 값을 테스트용 DB 주소로 바꾼다.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("APP_ENV", "test")

# 테스트 중 바깥 네트워크 연결을 막는다. 실수로 진짜 LLM API를 불러 돈을 쓰거나,
# 외부 응답에 따라 결과가 매번 달라지는 것을 막기 위해서다.
# 내 컴퓨터(localhost)와 유닉스 소켓은 허용한다.
# 로컬 DB에 붙는 테스트가 생겨도 그대로 쓰기 위해서다.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class NetworkBlockedError(RuntimeError):
    """테스트 중 바깥 네트워크 연결을 시도했을 때 난다."""


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    return host in _LOCAL_HOSTS or str(host).startswith("127.")


def _is_local_address(address) -> bool:
    # AF_UNIX 소켓은 주소가 파일 경로(str/bytes)다.
    if isinstance(address, (str, bytes)):
        return True
    return _is_local(address[0])


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if not _is_local_address(address):
            raise NetworkBlockedError(f"테스트 중 바깥 네트워크 연결 차단: {address!r}")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if not _is_local_address(address):
            raise NetworkBlockedError(f"테스트 중 바깥 네트워크 연결 차단: {address!r}")
        return original_connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_local(host):
            raise NetworkBlockedError(f"테스트 중 바깥 주소 조회 차단: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
