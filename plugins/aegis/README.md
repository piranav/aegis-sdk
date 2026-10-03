# Aegis plugin for coding assistants

Connects a person's company-managed coding assistant to their organization's Aegis.
IT sees the subscription as connected, the person's team and personal rules apply to
every tool call and prompt, and IT is told when the plugin goes quiet.

Supported: **Claude Code** and **Codex**. One plugin directory serves both.

## Install

Your IT team sends a code.

**Claude Code**

```text
/plugin marketplace add piranav/aegis-sdk
/plugin install aegis@aegis
```

**Codex** (in a terminal, then in Codex)

```text
codex plugin marketplace add piranav/aegis-sdk
codex plugin add aegis@aegis
/hooks      review and trust the Aegis hooks
```

Then start a new session and send this as a message:

```text
aegis connect aegc_...
```

`aegis status` and `aegis disconnect` work the same way. The plugin answers these
itself, so the code never reaches the model and connecting works even without a
working model. From a terminal: `python3 scripts/aegis.py connect <code> --tool codex`.

## What it does

| Event | Behavior |
| --- | --- |
| Session start | Reports liveness; reminds the person to connect if they haven't |
| Prompt | Answers `aegis ...` commands; evaluates other prompts and blocks denied ones |
| Before a tool call | Allow, deny, or ask the person (Codex can't ask yet, so it refuses) |
| After a tool call | Evaluates the result against output rules |
| Session end | Reports the session ended |

Each tool call is sent with its canonical action (`file.read`, `file.write`,
`shell.exec`, `mcp.call`, `web.fetch`) and the files it touches, so one rule can hold
across assistants.

Keys are stored per assistant at `~/.aegis/<assistant>/credentials.json` (mode 0600).
Requires `python3` (standard library only). If Aegis is unreachable, the plugin follows
the organization's setting: allow with a warning, or block. A suspended or revoked
subscription always blocks.

To make the plugin mandatory, deploy it through managed configuration: Claude Code
managed settings (`enabledPlugins`, `strictKnownMarketplaces`) or Codex
`requirements.toml`.

## Adding an assistant

Add an adapter in `scripts/aegis_plugin/adapters.py` (event names, canonical tool
actions, account details, decision formats), a hooks file that calls
`aegis.py hook <assistant> <Event>`, and the assistant's manifest. The engine,
connection handling, and API client are shared.
