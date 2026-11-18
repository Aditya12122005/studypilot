from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a deterministic OpenAI client stub that never makes requests."""
    def install(replies):
        replies = iter(replies)
        calls = []

        class FakeOpenAI:
            def __init__(self, api_key):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            def create(self, **kwargs):
                calls.append(kwargs)
                reply = next(replies)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=reply)
                        )
                    ]
                )

        monkeypatch.setitem(
            sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI)
        )
        return calls

    return install
