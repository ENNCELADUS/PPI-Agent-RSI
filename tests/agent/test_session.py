import queue
import sys
import time

import pytest

from src.agent.session import Rpc

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
    with pytest.raises(OSError):  # run_wave turns this into a wave error, which the evaluator retries
        for _ in range(1000):
            rpc.send(type="prompt", message="x" * 1000)
    rpc.close()
