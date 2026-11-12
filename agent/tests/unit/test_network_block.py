import socket

import pytest
from conftest import NetworkBlockedError


def test_external_connect_is_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("93.184.216.34", 443))


def test_external_connect_ex_is_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex(("93.184.216.34", 443))


def test_external_name_lookup_is_blocked():
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("api.anthropic.com", 443)


def test_localhost_is_allowed():
    # 로컬 DB 테스트를 위해 localhost는 막지 않는다. 열린 포트가 없어도 차단 오류가 아니면 된다.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("localhost", port), timeout=1):
            pass
