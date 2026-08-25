import random
import socket
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def deterministic_random_seed() -> Iterator[None]:
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in default tests")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
