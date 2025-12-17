import json
from types import SimpleNamespace

import pytest

from utils.gpt_helper.openai_api import chat_completion, run_sequential_requests, run_batch_requests


def test_chat_completion_uses_provided_client():
    class FakeChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = SimpleNamespace(content="hello")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions()))

    result = chat_completion(
        "sys",
        "user",
        client=fake_client,
        model="mymodel",
        temperature=0.7,
    )

    assert result == "hello"
    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "mymodel"
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert call["temperature"] == 0.7


def test_run_sequential_requests(monkeypatch):
    calls = []

    def fake_chat(system_prompt, user_prompt, **kwargs):
        calls.append((system_prompt, user_prompt, kwargs))
        return f"{system_prompt}|{user_prompt}"

    monkeypatch.setattr("utils.gpt_helper.openai_api.chat_completion", fake_chat)

    requests = [
        ("s1", "u1"),
        {"system": "s2", "user": "u2"},
    ]
    outputs = run_sequential_requests(requests, model="x")

    assert outputs == ["s1|u1", "s2|u2"]
    assert calls[0][2]["model"] == "x"


def test_run_batch_requests(monkeypatch):
    responses = [
        {
            "custom_id": "request-0",
            "response": {"body": {"choices": [{"message": {"content": "a"}}]}},
        },
        {
            "custom_id": "request-1",
            "response": {"body": {"choices": [{"message": {"content": "b"}}]}},
        },
    ]
    raw_output = "\n".join(json.dumps(record) for record in responses)

    class FakeFiles:
        def __init__(self):
            self.deleted = []

        def create(self, file, purpose):
            return SimpleNamespace(id="input-file")

        def content(self, file_id):
            assert file_id == "output-file"
            return SimpleNamespace(text=raw_output)

        def delete(self, file_id):
            self.deleted.append(file_id)
            return None

    class FakeBatches:
        def __init__(self):
            self.calls = []
            self.retrieves = 0

        def create(self, input_file_id, endpoint, completion_window):
            self.calls.append((input_file_id, endpoint, completion_window))
            return SimpleNamespace(id="batch-1", status="in_progress", output_file_id=None)

        def retrieve(self, batch_id):
            self.retrieves += 1
            if self.retrieves < 2:
                return SimpleNamespace(id=batch_id, status="in_progress", output_file_id=None)
            return SimpleNamespace(id=batch_id, status="completed", output_file_id="output-file")

    fake_client = SimpleNamespace(
        files=FakeFiles(),
        batches=FakeBatches(),
        chat=None,
    )
    monkeypatch.setattr("utils.gpt_helper.openai_api.time.sleep", lambda *_: None)

    outputs = run_batch_requests(
        [("sys1", "u1"), {"system": "sys2", "user": "u2"}],
        client=fake_client,
        model="test-model",
        poll_interval=0,
    )

    assert outputs == ["a", "b"]
    assert fake_client.files.deleted == ["input-file", "output-file"]
    assert fake_client.batches.calls[0][1] == "/v1/chat/completions"
