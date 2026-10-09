"""Aegis assistant plugin: shared engine and per-assistant adapters."""

import json
import subprocess
import sys
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "aegis"
sys.path.insert(0, str(PLUGIN / "scripts"))

from aegis_plugin import adapters, engine, inventory  # noqa: E402
from aegis_plugin.adapters import ADAPTERS  # noqa: E402
from aegis_plugin.client import ApiError, State  # noqa: E402

API = "http://aegis.test"
CODE = "aegc_" + urlsafe_b64encode(f"{API}|secret-token".encode()).decode().rstrip("=")
SECRET_FILE = "/repo/." + "env"


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Record API calls and answer them from a queue of canned responses."""
    log = {"requests": [], "responses": []}

    def fake_request(api_url, path, body, key=None, method="POST"):
        log["requests"].append(
            {"url": api_url, "path": path, "body": body, "key": key, "method": method}
        )
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


@pytest.mark.parametrize(
    ("tool", "tool_name", "tool_input", "commands"),
    [
        ("claude-code", "Bash", {"command": "cat .env | grep OPENAI"}, ["cat .env | grep OPENAI"]),
        ("codex", "Bash", {"command": ["bash", "-lc", "cat .env"]}, ["cat .env"]),
        (
            "codex",
            "exec",
            {"input": 'await tools.exec_command({cmd: "cat .env", max_output_tokens: 2000});'},
            ["cat .env"],
        ),
    ],
)
def test_shell_commands_and_the_files_they_touch_reach_aegis(
    calls, tool, tool_name, tool_input, commands
):
    eng = connected(tool)
    handle(
        tool,
        "PreToolUse",
        {"session_id": "s", "tool_name": tool_name, "tool_input": tool_input},
        eng,
    )
    meta = calls["requests"][0]["body"]["tool_args"]["_aegis"]
    assert meta["action"] == "shell.exec"
    assert meta["commands"] == commands
    assert ".env" in meta["paths"]


def test_command_paths_keep_files_and_skip_flags_words_and_urls():
    found = adapters.command_paths(
        "curl -s https://x.dev/a > out.json && sed -n 1,5p ~/.ssh/config src/app.py README"
    )
    assert found == ["out.json", "~/.ssh/config", "src/app.py"]


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
    monkeypatch.delenv("CODEX_HOME", raising=False)
    expected = {"account_email": "dev@corp", "account_organization": "ChatGPT team"}
    assert ADAPTERS["codex"].account() == expected
    (tmp_path / "custom").mkdir()
    (tmp_path / ".codex" / "auth.json").rename(tmp_path / "custom" / "auth.json")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom"))
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


# ------------------------------------------------------------------------- inventory


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def skill(directory, name, description):
    write(
        directory / name / "SKILL.md", f"---\nname: {name}\ndescription: {description}\n---\nBody"
    )


def by_key(manifest):
    return {(c["kind"], c["name"]): c for c in manifest["components"]}


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "repo"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(inventory.Path, "home", staticmethod(lambda: home))
    plugin_root = home / ".claude/plugins/cache/acme/docs/1.2.0"
    write(plugin_root / ".claude-plugin/plugin.json", {"name": "docs", "version": "1.2.0"})
    write(
        plugin_root / ".mcp.json",
        {"mcpServers": {"search": {"command": "node srv.js --key s3cret"}}},
    )
    skill(plugin_root / "skills", "style-guide", "House style")
    write(plugin_root / "agents/writer.md", "---\nname: writer\ndescription: Drafts docs\n---\n")
    write(
        home / ".claude/plugins/installed_plugins.json",
        {"version": 2, "plugins": {"docs@acme": [{"installPath": str(plugin_root)}]}},
    )
    write(
        home / ".claude/settings.json",
        {"model": "claude-sonnet-4-5", "enabledPlugins": {"docs@acme": True, "off@acme": False}},
    )
    write(
        home / ".claude.json",
        {
            "mcpServers": {
                "linear": {
                    "type": "http",
                    "url": "https://mcp.linear.app/mcp?token=x",
                    "headers": {"Authorization": "Bearer y"},
                }
            },
            "projects": {
                str(project): {
                    "mcpServers": {
                        "pg": {
                            "command": "/Applications/My Tools/pg-mcp",
                            "env": {"PGPASSWORD": "z"},
                        }
                    }
                }
            },
        },
    )
    write(
        project / ".mcp.json",
        {"mcpServers": {"sentry": {"type": "sse", "url": "https://mcp.sentry.dev/sse"}}},
    )
    skill(home / ".claude/skills", "pdf", "Work with PDFs")
    write(project / ".claude/agents/reviewer.md", "Reviews diffs")
    return project


def test_claude_code_scanner_reports_what_the_assistant_loads(claude_home):
    manifest = inventory.ClaudeCodeScanner().manifest(str(claude_home))
    found = by_key(manifest)

    assert manifest["framework"] == "claude-code" and manifest["scope"].startswith("project:")
    assert found[("model", "claude-sonnet-4-5")]["provider"] == "anthropic"
    assert found[("mcp_server", "linear")]["locator"] == "https://mcp.linear.app/mcp"
    assert found[("mcp_server", "pg")]["attributes"] == {"scope": "local"}
    assert found[("mcp_server", "pg")]["locator"] == "pg-mcp"
    assert found[("mcp_server", "sentry")]["transport"] == "sse"
    plugin_server = found[("mcp_server", "plugin_docs_search")]
    assert (plugin_server["locator"], plugin_server["attributes"]) == ("node", {"plugin": "docs"})
    docs = found[("plugin", "docs")]
    assert docs["version"] == "1.2.0"
    assert {(c["kind"], c["name"]) for c in docs["children"]} == {
        ("skill", "style-guide"),
        ("subagent", "writer"),
    }
    assert found[("skill", "pdf")]["description"] == "Work with PDFs"
    assert ("subagent", "reviewer") in found
    assert ("plugin", "off") not in found
    serialized = json.dumps(manifest)
    for secret in ("s3cret", "token=x", "Bearer", "PGPASSWORD"):
        assert secret not in serialized


def test_codex_scanner_reads_config_toml_plugins_and_skills(tmp_path, monkeypatch):
    codex_home, project = tmp_path / "codex", tmp_path / "repo"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(inventory.Path, "home", staticmethod(lambda: tmp_path / "home"))
    write(
        codex_home / "config.toml",
        "\n".join(
            [
                'model = "gpt-5.1-codex"',
                '[plugins."github@openai-curated"]',
                "enabled = true",
                '[plugins."pets@openai-curated"]',
                "enabled = false",
                "[mcp_servers.docs]",
                'url = "https://developers.openai.com/mcp"',
                "[mcp_servers.repl]",
                'command = "/Applications/Tool.app/bin/node_repl"',
                "enabled = false",
                "[mcp_servers.repl.env]",
                'TOKEN = "never-sent"',
            ]
        ),
    )
    plugin_root = codex_home / "plugins/cache/openai-curated/github/0.1.12"
    write(plugin_root / ".codex-plugin/plugin.json", {"name": "github", "version": "0.1.12"})
    skill(plugin_root / "skills", "pr-review", "Review pull requests")
    # Frontmatter names may differ from the folder Codex opens the skill by.
    write(plugin_root / "skills/slides/SKILL.md", "---\nname: Slides\n---\nBody")
    write(plugin_root / ".mcp.json", {"mcpServers": {"github_app": {"command": "launch"}}})
    skill(project / ".agents/skills", "release", "Cut a release")

    manifest = inventory.CodexScanner().manifest(str(project), model="gpt-5.2-codex")
    found = by_key(manifest)

    assert ("model", "gpt-5.2-codex") in found and ("model", "gpt-5.1-codex") not in found
    assert found[("mcp_server", "docs")]["locator"] == "https://developers.openai.com/mcp"
    assert found[("mcp_server", "repl")]["attributes"] == {"scope": "user", "enabled": False}
    assert [c["name"] for c in found[("plugin", "github")]["children"]] == ["pr-review", "slides"]
    # Codex keeps plugin servers' own names in tool calls (mcp__github_app__...).
    assert found[("mcp_server", "github_app")]["attributes"] == {"plugin": "github"}
    assert ("plugin", "pets") not in found
    assert found[("skill", "release")]["attributes"] == {"scope": "project"}
    assert "never-sent" not in json.dumps(manifest)


def test_minimal_toml_parser_covers_the_keys_scanners_read():
    parsed = inventory._minimal_toml(
        'model = "m"\n[plugins."a@b"]\nenabled = true\n'
        '[mcp_servers.s]\nurl = "https://x"\nargs = []'
    )
    assert parsed == {
        "model": "m",
        "plugins": {"a@b": {"enabled": True}},
        "mcp_servers": {"s": {"url": "https://x"}},
    }


def test_session_start_sends_the_manifest_only_when_it_changes(calls, claude_home):
    eng = connected("claude-code")
    start = {"session_id": "s", "cwd": str(claude_home), "model": "claude-opus-4-6"}
    for _ in range(2):
        calls["responses"].append({"status": "active"})
        assert handle("claude-code", "SessionStart", start, eng).payload is None

    manifests = [r for r in calls["requests"] if r["path"] == "/v1/inventory/manifest"]
    assert len(manifests) == 1 and manifests[0]["method"] == "PUT"
    assert manifests[0]["key"] == "aegk_test"
    assert ("model", "claude-opus-4-6") in by_key(manifests[0]["body"])


def test_inventory_failures_never_block_the_session(calls, claude_home):
    eng = connected("claude-code")
    calls["responses"] += [{"status": "active"}, ApiError(500, "boom")]
    out = handle("claude-code", "SessionStart", {"session_id": "s", "cwd": str(claude_home)}, eng)
    assert out.payload is None
    # Not remembered, so the next session retries.
    calls["responses"] += [{"status": "active"}, {}]
    handle("claude-code", "SessionStart", {"session_id": "t", "cwd": str(claude_home)}, eng)
    assert sum(r["path"] == "/v1/inventory/manifest" for r in calls["requests"]) == 2


# --------------------------------------------------------------------------- usage

from aegis_plugin import usage  # noqa: E402


def claude_line(message_id, text, *, model="claude-sonnet-4-5", output=10, sidechain=False):
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-10-06T10:00:00Z",
            "isSidechain": sidechain,
            "message": {
                "id": message_id,
                "model": model,
                "content": [{"type": "text", "text": text}],
                "usage": {
                    "input_tokens": 12,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                    "output_tokens": output,
                },
            },
        }
    )


def codex_lines():
    return [
        json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}),
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.2-codex"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {"type": "message", "content": [{"text": "SECRET PROMPT"}]},
            }
        ),
        json.dumps(
            {
                "type": "token_usage_record",
                "timestamp": "2026-10-06T10:00:01Z",
                "payload": {
                    "response_id": "resp_1",
                    "usage": {
                        "input_tokens": 3000,
                        "cached_input_tokens": 2000,
                        "output_tokens": 90,
                        "reasoning_output_tokens": 40,
                    },
                },
            }
        ),
    ]


def test_claude_transcript_merges_blocks_holds_back_the_open_message_and_skips_synthetic():
    lines = [
        claude_line("msg_1", "thinking", output=3),
        claude_line("msg_1", "final block", output=25),
        json.dumps({"type": "user", "message": {"content": "user prompt text"}}),
        claude_line("msg_x", "", model="<synthetic>"),
        claude_line("msg_2", "subagent work", sidechain=True),
    ]
    data = ("\n".join(lines) + "\n").encode()

    parsed = usage.ClaudeTranscript().parse(data, "", final=False)
    (call,) = parsed.calls
    assert (call.call_id, call.output_tokens, call.input_tokens) == ("msg_1", 25, 1212)
    assert (call.cache_read_tokens, call.cache_write_tokens) == (1000, 200)
    # msg_2 may still be growing, so the next read starts at its first line.
    assert data[parsed.resume_at :].startswith(lines[4].encode())

    final = usage.ClaudeTranscript().parse(data, "", final=True)
    assert [c.call_id for c in final.calls] == ["msg_1", "msg_2"]
    assert final.calls[1].subagent


def test_codex_rollout_reads_usage_records_with_the_turn_model():
    data = ("\n".join(codex_lines()) + "\n").encode()
    parsed = usage.CodexRollout().parse(data, "", final=False)
    (call,) = parsed.calls
    assert (call.call_id, call.model, call.input_tokens, call.cache_read_tokens) == (
        "resp_1",
        "gpt-5.2-codex",
        3000,
        2000,
    )
    assert call.reasoning_tokens == 40 and parsed.model == "gpt-5.2-codex"


def test_older_codex_token_count_events_are_keyed_by_file_position():
    line = json.dumps(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"last_token_usage": {"input_tokens": 50, "output_tokens": 5}},
            },
        }
    )
    parsed = usage.CodexRollout().parse((line + "\n").encode(), "gpt-5", False, base_offset=700)
    assert [c.call_id for c in parsed.calls] == ["codex-offset-700"]


def token_count(timestamp, last, total):
    return json.dumps(
        {
            "type": "event_msg",
            "timestamp": timestamp,
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": last, "output_tokens": 1},
                    "total_token_usage": {"input_tokens": total, "output_tokens": 1},
                },
            },
        }
    )


def test_a_thread_resumed_on_newer_codex_counts_each_response_once(monkeypatch, tmp_path):
    # Months on an older Codex (token_count only, one of them a repeat), then resumed on
    # a newer one that writes a usage record and a token_count for the same response.
    monkeypatch.setattr(usage, "MAX_READ_BYTES", 300)  # the format changes between reads
    lines = [
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.6-sol"}}),
        token_count("2026-08-06T01:00:00Z", 100, 100),
        token_count("2026-08-06T01:00:01Z", 100, 100),  # repeated, not a new response
        token_count("2026-08-06T01:00:02Z", 200, 300),
        codex_lines()[3],
        token_count("2026-10-06T10:00:01Z", 3000, 3300),
    ]
    rollout = write(tmp_path / "rollout.jsonl", "\n".join(lines) + "\n")
    cursor, ids = {}, []
    while True:
        batch = usage.read_usage(usage.CodexRollout(), str(rollout), "t", cursor)
        ids += [(e["call_id"], e["usage"]["input_tokens"]) for e in batch.events]
        if batch.at_end:
            break
        cursor = {"offset": batch.offset, "model": batch.model, **batch.links}
    assert [tokens for _, tokens in ids] == [100, 200, 3000]
    assert ids[-1][0] == "resp_1"


def test_usage_from_before_connecting_is_not_reported(calls, tmp_path):
    eng = connected("codex")
    eng.state.save({**eng.state.credentials(), "connected_at": 1791280800})  # 2026-10-06T10:00Z
    lines = [
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.6-sol"}}),
        token_count("2026-08-06T01:00:00Z", 90_000_000, 90_000_000),
        token_count("2026-10-06T10:00:05Z", 4_000, 90_004_000),
    ]
    rollout = write(tmp_path / "rollout.jsonl", "\n".join(lines) + "\n")
    handle("codex", "Stop", {"session_id": "t-old", "transcript_path": str(rollout)}, eng)

    events = [e for r in calls["requests"] for e in r["body"].get("events", [])]
    assert [e["usage"]["input_tokens"] for e in events if e["type"] == "llm.call"] == [4_000]
    # The history is read past, not re-read on every turn.
    assert State("codex").usage_cursor("t-old")["offset"] == rollout.stat().st_size


def test_partial_last_line_is_left_for_the_next_read():
    data = (claude_line("msg_1", "done") + "\n" + claude_line("msg_2", "half")[:40]).encode()
    parsed = usage.ClaudeTranscript().parse(data, "", final=True)
    assert [c.call_id for c in parsed.calls] == ["msg_1"]
    assert parsed.resume_at == len(claude_line("msg_1", "done")) + 1


def test_a_session_reports_tokens_incrementally_and_never_sends_text(calls, tmp_path):
    eng = connected("claude-code")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(claude_line("msg_1", "SECRET ANSWER ONE") + "\n")
    hook = {"session_id": "s-9", "transcript_path": str(transcript), "cwd": str(tmp_path)}
    calls["responses"].append({"status": "active"})

    handle("claude-code", "SessionStart", hook, eng)
    with transcript.open("a") as f:
        f.write(claude_line("msg_2", "SECRET ANSWER TWO") + "\n")
    handle("claude-code", "Stop", hook, eng)  # msg_1 complete, msg_2 held back
    handle("claude-code", "Stop", hook, eng)  # nothing new
    handle("claude-code", "SessionEnd", hook, eng)  # releases msg_2 and closes

    telemetry = [r for r in calls["requests"] if r["path"] == "/v1/telemetry/events"]
    sent = [e for r in telemetry for e in r["body"]["events"]]
    assert [(e["type"], e.get("call_id")) for e in sent] == [
        ("session.start", None),
        ("llm.call", "msg_1"),
        ("llm.call", "msg_2"),
        ("session.end", None),
    ]
    assert all(e["session_id"] == "s-9" for e in sent)
    assert {r["key"] for r in telemetry} == {"aegk_test"}
    assert "SECRET" not in json.dumps(sent)
    assert State("claude-code").usage_cursor("s-9") == {}  # forgotten after the session


def test_failed_usage_reports_are_retried_from_the_same_place(calls, tmp_path):
    eng = connected("codex")
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text("\n".join(codex_lines()) + "\n")
    hook = {"session_id": "t-1", "transcript_path": str(transcript)}
    calls["responses"].append(ApiError(503, "busy"))
    handle("codex", "Stop", hook, eng)
    assert State("codex").usage_cursor("t-1") == {}
    handle("codex", "Stop", hook, eng)
    events = [e for r in calls["requests"] for e in r["body"].get("events", [])]
    assert [e["call_id"] for e in events if e["type"] == "llm.call"] == ["resp_1", "resp_1"]
    assert State("codex").usage_cursor("t-1")["model"] == "gpt-5.2-codex"


def test_usage_reporting_ignores_missing_transcripts(calls):
    eng = connected("claude-code")
    out = handle("claude-code", "Stop", {"session_id": "s", "transcript_path": "/nope.jsonl"}, eng)
    assert out.payload is None
    assert not [r for r in calls["requests"] if r["path"] == "/v1/telemetry/events"]


def test_long_sessions_are_read_to_the_end_in_one_report(calls, tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "MAX_READ_BYTES", 400)
    eng = connected("claude-code")
    transcript = tmp_path / "long.jsonl"
    transcript.write_text("".join(claude_line(f"msg_{i}", "x" * 50) + "\n" for i in range(12)))
    handle(
        "claude-code", "SessionEnd", {"session_id": "L", "transcript_path": str(transcript)}, eng
    )
    sent = [e for r in calls["requests"] for e in r["body"].get("events", [])]
    assert [e["call_id"] for e in sent if e["type"] == "llm.call"] == [
        f"msg_{i}" for i in range(12)
    ]


# ------------------------------------------------------------------- tool attribution


def claude_tool_turn(message_id, tool_id, tool_name, tool_input, *, output=10):
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-10-06T10:00:00Z",
            "message": {
                "id": message_id,
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
                ],
                "usage": {"input_tokens": 100, "output_tokens": output},
            },
        }
    )


def claude_tool_result(tool_id, text):
    return json.dumps(
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": text}]
            },
        }
    )


def test_claude_transcript_links_tool_calls_results_and_skill_targets():
    lines = [
        claude_tool_turn("msg_1", "tu_1", "mcp__github__search_code", {"q": "PRIVATE QUERY"}),
        claude_tool_result("tu_1", "PRIVATE RESULT " * 40),
        claude_tool_turn("msg_2", "tu_2", "Skill", {"skill": "docs:style-guide", "args": "x"}),
        claude_tool_result("tu_2", "skill body"),
        claude_line("msg_3", "final answer"),
    ]
    data = ("\n".join(lines) + "\n").encode()
    parsed = usage.ClaudeTranscript().parse(data, "", final=True)
    calls = {c.call_id: c for c in parsed.calls}
    assert calls["msg_1"].tool_calls == [{"id": "tu_1", "name": "mcp__github__search_code"}]
    assert calls["msg_2"].tool_results == [
        {"id": "tu_1", "tokens": 150, "name": "mcp__github__search_code"}
    ]
    assert calls["msg_2"].tool_calls[0]["target"] == "docs:style-guide"
    assert calls["msg_3"].tool_results[0]["target"] == "docs:style-guide"
    sent = json.dumps([c.event("s", "anthropic") for c in parsed.calls])
    assert "PRIVATE" not in sent and '"args"' not in sent


def test_held_back_messages_keep_their_tool_results_for_the_next_read(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        claude_tool_turn("msg_1", "tu_1", "Read", {"file_path": "/a"})
        + "\n"
        + claude_tool_result("tu_1", "x" * 400)
        + "\n"
        + claude_line("msg_2", "open")
        + "\n"
    )
    first = usage.read_usage(usage.ClaudeTranscript(), str(path), "s", {})
    assert [e["call_id"] for e in first.events] == ["msg_1"]  # msg_2 held back
    assert first.links["pending"] == [{"id": "tu_1", "tokens": 100, "name": "Read"}]
    cursor = {"offset": first.offset, "model": first.model, **first.links}
    second = usage.read_usage(usage.ClaudeTranscript(), str(path), "s", cursor, final=True)
    (event,) = second.events
    assert event["call_id"] == "msg_2" and event["tool_results"][0]["id"] == "tu_1"


def test_codex_results_are_read_by_the_next_response():
    def item(kind, call_id, **extra):
        return json.dumps(
            {"type": "response_item", "payload": {"type": kind, "call_id": call_id, **extra}}
        )

    def record(response_id):
        return json.dumps(
            {
                "type": "token_usage_record",
                "payload": {
                    "response_id": response_id,
                    "usage": {"input_tokens": 1000, "output_tokens": 10},
                },
            }
        )

    lines = [
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}}),
        item("custom_tool_call", "c1", name="exec", input="cat PRIVATE"),
        record("resp_1"),
        item("custom_tool_call_output", "c1", output="PRIVATE OUTPUT " * 30),
        item("function_call", "c2", name="mcp__docs__search", arguments="{}"),
        record("resp_2"),
        item("function_call_output", "c2", output="found"),
        record("resp_3"),
    ]
    parsed = usage.CodexRollout().parse(("\n".join(lines) + "\n").encode(), "", False)
    calls = {c.call_id: c for c in parsed.calls}
    assert [t["id"] for t in calls["resp_1"].tool_calls] == ["c1"]
    assert calls["resp_1"].tool_results == []
    assert calls["resp_2"].tool_results == [{"id": "c1", "tokens": 113, "name": "exec"}]
    assert calls["resp_2"].tool_calls == [{"id": "c2", "name": "mcp__docs__search"}]
    assert calls["resp_3"].tool_results[0]["name"] == "mcp__docs__search"
    assert "PRIVATE" not in json.dumps([c.event("s", "openai") for c in parsed.calls])
