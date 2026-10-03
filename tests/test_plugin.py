"""Aegis assistant plugin: shared engine and per-assistant adapters."""

import json
import subprocess
import sys
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "aegis"
sys.path.insert(0, str(PLUGIN / "scripts"))

from aegis_plugin import adapters, engine  # noqa: E402
from aegis_plugin.adapters import ADAPTERS  # noqa: E402
from aegis_plugin.client import ApiError, State  # noqa: E402

API = "http://aegis.test"
CODE = "aegc_" + urlsafe_b64encode(f"{API}|secret-token".encode()).decode().rstrip("=")
SECRET_FILE = "/repo/." + "env"


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Record API calls and answer them from a queue of canned responses."""
    log = {"requests": [], "responses": []}

    def fake_request(api_url, path, body, key=None):
        log["requests"].append({"url": api_url, "path": path, "body": body, "key": key})
        response = log["responses"].pop(0) if log["responses"] else {"decision": "allow"}
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(engine, "request", fake_request)
    monkeypatch.setenv("AEGIS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(adapters.Path, "home", staticmethod(lambda: tmp_path))
    return log


def connected(tool, failure_mode="open"):
    state = State(tool)
    state.save({"api_url": API, "api_key": "aegk_test", "failure_mode": failure_mode})
    return engine.Engine(ADAPTERS[tool], state)


def handle(tool, hook, data, eng=None):
    adapter = ADAPTERS[tool]
    return (eng or engine.Engine(adapter)).handle(adapter.parse(hook, data))


@pytest.mark.parametrize("tool", ["claude-code", "codex"])
def test_connect_command_is_answered_without_reaching_the_model(calls, tool):
    calls["responses"].append(
        {
            "api_key": "aegk_new",
            "subscription_id": "s1",
            "email": "dev@corp",
            "organization": "Corp",
        }
    )
    out = handle(tool, "UserPromptSubmit", {"session_id": "s", "prompt": f"aegis connect {CODE}"})
    assert out.payload["decision"] == "block", "the prompt must not reach the model"
    assert "Aegis connected: dev@corp at Corp" in out.payload["reason"]
    sent = calls["requests"][0]
    assert sent["path"] == "/v1/subscriptions/connect" and sent["url"] == API
    assert sent["body"]["tool"] == tool and sent["body"]["code"] == CODE
    assert State(tool).credentials()["api_key"] == "aegk_new"


def test_status_and_disconnect_commands(calls):
    eng = connected("codex")
    calls["responses"].append({"status": "active", "failure_mode": "closed"})
    out = handle("codex", "UserPromptSubmit", {"prompt": "aegis status"}, eng)
    assert "status active" in out.payload["reason"] and out.payload["systemMessage"]
    assert State("codex").credentials()["failure_mode"] == "closed"
    calls["responses"].append({"status": "disconnected"})
    out = handle("codex", "UserPromptSubmit", {"prompt": "/aegis disconnect"}, eng)
    assert "disconnected" in out.payload["reason"]
    assert calls["requests"][-1]["body"]["event"] == "disconnect"
    assert State("codex").credentials() is None


def test_session_start_reports_liveness_or_asks_to_connect(calls):
    out = handle("codex", "SessionStart", {"session_id": "s"})
    assert "aegis connect" in out.payload["systemMessage"] and not calls["requests"]
    eng = connected("codex")
    calls["responses"].append({"status": "active"})
    assert handle("codex", "SessionStart", {"session_id": "s"}, eng).payload is None
    assert calls["requests"][0]["body"]["event"] == "session_start"


def test_codex_patch_is_described_in_canonical_terms(calls):
    eng = connected("codex")
    patch = f"*** Begin Patch\n*** Update File: {SECRET_FILE}\n+X=1\n*** End Patch\n"
    calls["responses"].append(
        {"decision": "deny", "violations": ["Secret files are off limits"], "audit_id": "a1"}
    )
    out = handle(
        "codex",
        "PreToolUse",
        {"session_id": "s", "tool_name": "apply_patch", "tool_input": {"command": patch}},
        eng,
    )
    meta = calls["requests"][0]["body"]["tool_args"]["_aegis"]
    assert meta == {"tool": "codex", "action": "file.write", "paths": [SECRET_FILE]}
    decision = out.payload["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "Secret files are off limits" in decision["permissionDecisionReason"]


def test_escalation_asks_in_claude_code_and_refuses_in_codex(calls):
    for tool, expected in (("claude-code", "ask"), ("codex", "deny")):
        eng = connected(tool)
        calls["responses"].append({"decision": "escalate", "violations": ["Needs review"]})
        out = handle(tool, "PreToolUse", {"tool_name": "Bash", "tool_input": {}}, eng)
        assert out.payload["hookSpecificOutput"]["permissionDecision"] == expected


def test_results_are_bound_to_their_tool_call(calls):
    eng = connected("claude-code")
    calls["responses"].append({"decision": "allow", "audit_id": "audit-1"})
    pre = {"session_id": "s", "tool_name": "Read", "tool_input": {"file_path": "/a"}}
    handle("claude-code", "PreToolUse", {**pre, "tool_use_id": "t1"}, eng)
    calls["responses"].append({"decision": "deny", "violations": ["Output contains credentials"]})
    post = {"session_id": "s", "tool_use_id": "t1", "tool_response": {"content": "x"}}
    out = handle("claude-code", "PostToolUse", post, eng)
    assert calls["requests"][-1]["body"]["audit_id"] == "audit-1"
    assert out.payload["decision"] == "block"


@pytest.mark.parametrize(
    ("mode", "blocked"), [("open", False), ("closed", True)], ids=["open", "closed"]
)
def test_outages_follow_the_organization_setting(calls, mode, blocked):
    eng = connected("codex", mode)
    calls["responses"].append(ApiError(0, "Aegis is unreachable"))
    out = handle("codex", "PreToolUse", {"tool_name": "Bash", "tool_input": {}}, eng)
    assert ("hookSpecificOutput" in out.payload) is blocked
    calls["responses"].append({"decision": "allow", "audit_id": "a"})
    handle("codex", "PreToolUse", {"tool_name": "Bash", "tool_use_id": "t", "tool_input": {}}, eng)
    calls["responses"].append(
        {"decision": "deny", "violations": ["Output classification unavailable -- fail-closed"]}
    )
    out = handle("codex", "PostToolUse", {"tool_use_id": "t", "tool_response": "ok"}, eng)
    assert (out.payload is not None) is blocked


def test_revoked_connections_always_block(calls):
    eng = connected("claude-code", "open")
    calls["responses"].append(ApiError(401, "Invalid API key"))
    out = handle("claude-code", "PreToolUse", {"tool_name": "Bash", "tool_input": {}}, eng)
    assert out.payload["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_codex_account_comes_from_the_id_token_claims(tmp_path, monkeypatch):
    claims = {"email": "dev@corp", "https://api.openai.com/auth": {"chatgpt_plan_type": "team"}}
    payload = urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "auth.json").write_text(
        json.dumps({"tokens": {"id_token": f"h.{payload}.sig"}})
    )
    monkeypatch.setattr(adapters.Path, "home", staticmethod(lambda: tmp_path))
    assert ADAPTERS["codex"].account() == {
        "account_email": "dev@corp",
        "account_organization": "ChatGPT team",
    }


def test_unconnected_assistants_are_left_alone_and_unknown_events_ignored(calls):
    assert ADAPTERS["codex"].parse("PermissionRequest", {}) is None
    out = handle("codex", "PreToolUse", {"tool_name": "Bash", "tool_input": {}})
    assert out.payload is None and not calls["requests"]


@pytest.mark.parametrize("manifest", [".claude-plugin/plugin.json", ".codex-plugin/plugin.json"])
def test_manifests_point_at_their_assistant_hooks(manifest):
    tool = "claude-code" if "claude" in manifest else "codex"
    data = json.loads((PLUGIN / manifest).read_text())
    hooks = json.loads((PLUGIN / data["hooks"]).read_text())["hooks"]
    assert set(hooks) == set(ADAPTERS[tool].events)
    for entries in hooks.values():
        handler = entries[0]["hooks"][0]
        command = " ".join([handler["command"], *handler.get("args", [])])
        assert f"hook {tool} " in command


def test_entry_point_runs_a_hook_end_to_end(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PLUGIN / "scripts" / "aegis.py"), "hook", "codex", "SessionStart"],
        input="{}",
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "AEGIS_STATE_DIR": str(tmp_path / "s"), "PATH": ""},
    )
    assert result.returncode == 0
    assert "aegis connect" in json.loads(result.stdout)["systemMessage"]
