import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from src.agent.session import PI, Rpc, run_session

# emits an event every 50 ms forever, as pi does while a model streams tokens; exits on stdin EOF like pi
STREAM = """import json, sys, threading, time, os
threading.Thread(target=lambda: (sys.stdin.read(), os._exit(0)), daemon=True).start()
while True:
    print(json.dumps({"type": "message_update"}), flush=True)
    time.sleep(0.05)
"""


def test_deadline_holds_while_events_keep_streaming(tmp_path):
    rpc = Rpc([sys.executable, "-c", STREAM], tmp_path, None)
    start = time.time()
    try:
        with pytest.raises(queue.Empty):
            for _ in rpc.until("agent_settled", start + 0.5):
                pass
        assert time.time() - start < 2
    finally:
        rpc.close()


def test_close_is_safe_after_pi_died(tmp_path):
    rpc = Rpc([sys.executable, "-c", "pass"], tmp_path, None)
    rpc.p.wait()
    with pytest.raises(OSError):  # run_round turns this into a round error, which the evaluator retries
        for _ in range(1000):
            rpc.send(type="prompt", message="x" * 1000)
    rpc.close()


@pytest.mark.parametrize("missing", ["LLM_API_KEY", "LLM_API_BASE"])
def test_custom_provider_requires_shell_settings(tmp_path, monkeypatch, missing):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://example.invalid/v1")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        run_session(None, tmp_path, "system", "message", "nudge", model="custom/test", thinking="off")
    assert not list(tmp_path.iterdir())


def test_custom_provider_requires_model_id(tmp_path):
    with pytest.raises(ValueError, match="requires a model ID"):
        run_session(None, tmp_path, "system", "message", "nudge", model="custom/", thinking="off")


@pytest.mark.skipif(not PI.exists(), reason="pi runtime is not installed")
@pytest.mark.parametrize("thinking", ["off", "high"])
def test_custom_provider_uses_environment_without_saving_connection_settings(tmp_path, monkeypatch, thinking):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            chunk = {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]
            }
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("LLM_API_KEY", "test-secret-from-shell")
    monkeypatch.setenv("LLM_API_BASE", base)
    monkeypatch.setattr("src.agent.session.TIMEOUT", 20)
    tools = SimpleNamespace(names=[], submitted=None, calls=0)
    try:
        result = run_session(tools, tmp_path, "Reply OK", "ping", "ping", model="custom/test-chat", thinking=thinking)
    finally:
        server.shutdown()
        server.server_close()
    assert result["error"] is None, result
    assert len(requests) == 2  # initial prompt and the fixed nudge
    for path, authorization, body in requests:
        assert path == "/v1/chat/completions"
        assert authorization == "Bearer test-secret-from-shell"
        assert body["model"] == "test-chat"
        assert body["max_tokens"] == 384000
        assert "max_completion_tokens" not in body
        assert body["thinking"] == {"type": "enabled" if thinking == "high" else "disabled"}
        assert body.get("reasoning_effort") == ("high" if thinking == "high" else None)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            content = path.read_text()
            assert "test-secret-from-shell" not in content
            assert base not in content
