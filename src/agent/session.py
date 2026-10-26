"""Run one round of the agent: a fresh pi session (RPC mode) whose only tools are the round's tools.

pi runs from src/agent/pi (`npm ci` there first) with an isolated config directory: no built-in tools,
skills, prompt templates, context files, MCP or discovered extensions; only ppi_tools.ts, which
forwards calls to the Python tool server. If the session settles without a submission it gets one
fixed nudge. Returns what happened; the caller decides validity and results.
"""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from src.agent.tools import CAP, K, Sandbox, RoundTools, serve, specs

PI_DIR = Path(__file__).parent / "pi"
PI = PI_DIR / "node_modules" / ".bin" / "pi"
PROMPTS = Path(__file__).parent / "prompts"
TIMEOUT = 3600  # seconds per round


def prompt(name, **values):
    return (PROMPTS / f"{name}.md").read_text().format(**values)


def pi_version():
    return json.loads((PI_DIR / "node_modules/@earendil-works/pi-coding-agent/package.json").read_text())["version"]


class Rpc:
    """A pi --mode rpc process with a reader thread."""

    def __init__(self, cmd, cwd, env):
        self.stderr = open(Path(cwd) / "pi_stderr.log", "w")
        self.p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.put(json.loads(line))
        self.lines.put(None)

    def send(self, **command):
        self.p.stdin.write(json.dumps(command) + "\n")
        self.p.stdin.flush()

    def until(self, kind, deadline):
        """Events up to and including the first of the given type; raises on timeout or exit."""
        while True:
            left = deadline - time.time()
            if left <= 0:  # pi streams an event per token, so a get() timeout alone never fires
                raise queue.Empty
            event = self.lines.get(timeout=left)
            if event is None:
                raise RuntimeError(f"pi exited with code {self.p.wait()}")
            yield event
            if event.get("type") == kind:
                return

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:  # pi already exited
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.stderr.close()


def run_round(sandbox_dir, round_dir, tested, memory, *, t, rounds, model, thinking, prevalence):
    """One round in round_dir/agent. tested: canonical pairs already submitted; memory: agent-visible facts;
    prevalence: hit rate among candidate pairs, told to the agent as a prior."""
    assert PI.exists(), f"pi not installed: run `npm ci` in {PI_DIR}"
    sandbox = Sandbox(str(sandbox_dir))
    agent_dir = Path(round_dir) / "agent"
    tools = RoundTools(sandbox, agent_dir, tested, memory)
    (agent_dir / "tools.json").write_text(json.dumps(specs(), indent=1))
    config = agent_dir / "pi-config"  # empty: no user settings, models or extensions reach the agent
    config.mkdir(exist_ok=True)
    server, url = serve(tools)
    n = len(sandbox.candidates)
    system = prompt(
        "system", proteins=n, pairs=n * (n - 1) // 2, prevalence=prevalence, rounds=rounds, k=K, cap=CAP, window=5 * K
    )
    cmd = [
        str(PI),
        "--mode",
        "rpc",
        "--session-dir",
        str(agent_dir / "session"),
        "--model",
        model,
        "--thinking",
        thinking,
        "--system-prompt",
        system,
        "--no-builtin-tools",
        "--no-extensions",
        "-e",
        str(PI_DIR / "ppi_tools.ts"),
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-mcp",
        "--no-approve",
    ]
    env = {
        **os.environ,
        "PPI_TOOL_URL": url,
        "PPI_TOOLS_SPEC": str(agent_dir / "tools.json"),
        "PI_CODING_AGENT_DIR": str(config),
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
    }
    out = {"nudged": False, "aborted": False, "error": None, "stop_reasons": []}
    rpc = Rpc(cmd, agent_dir, env)
    deadline = time.time() + TIMEOUT
    try:
        rpc.send(type="set_auto_compaction", enabled=False)  # a fixed agent: context is never rewritten
        messages = [prompt("round", round=t, rounds=rounds, tested=len(tested), k=K)]
        while messages:
            rpc.send(type="prompt", message=messages.pop())
            last_error = None  # a provider error counts only if pi's own retries did not recover
            for event in rpc.until("agent_settled", deadline):
                if tools.submitted is not None and not out["aborted"]:
                    # pi stops on its own only if every call of the turn terminates; the batch is final, so stop here
                    out["aborted"] = True
                    rpc.send(type="abort")
                if event.get("type") == "response" and not event.get("success"):
                    raise RuntimeError(f"pi rejected {event.get('command')}: {event.get('error')}")
                if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                    reason = event["message"].get("stopReason")
                    out["stop_reasons"].append(reason)
                    last_error = event["message"].get("errorMessage", "provider error") if reason == "error" else None
                if event.get("type") == "extension_error":
                    out["error"] = f"extension: {event.get('error')}"
            out["error"] = out["error"] or (None if out["aborted"] else last_error)  # our abort is no error
            if tools.submitted is None and not out["nudged"] and not out["error"]:
                out["nudged"] = True
                messages.append(prompt("nudge", k=K))
        rpc.send(type="get_session_stats")
        while "stats" not in out:  # skip late replies, e.g. to the abort
            for event in rpc.until("response", deadline):
                if event.get("command") == "get_session_stats":
                    out["stats"] = event.get("data")
    except (queue.Empty, RuntimeError, OSError) as e:  # OSError: pi exited and the pipe broke
        out["error"] = f"timeout after {TIMEOUT} s" if isinstance(e, queue.Empty) else str(e)
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
    return out | {"submission": tools.submitted, "tool_calls": tools.calls}
