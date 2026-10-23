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
