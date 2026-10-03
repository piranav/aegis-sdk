# Aegis plugin for Claude Code

Connects a person's company-managed Claude Code to their organization's Aegis. IT sees
the subscription as connected, the person's team and personal rules apply to every tool
call and prompt, and IT is told when the plugin goes quiet.

## Install

Your IT team sends a code. Inside Claude Code:

```text
/plugin marketplace add piranav/aegis-sdk
/plugin install aegis@aegis
/aegis:connect aegc_...
```

`/aegis:status` shows the connection, and `/aegis:disconnect` disconnects (IT is notified).

## What it does

| Hook | Behavior |
| --- | --- |
| `SessionStart` | Reports liveness; reminds the person to connect if they haven't |
| `UserPromptSubmit` | Evaluates the prompt; blocked prompts never reach Claude |
| `PreToolUse` | Evaluates every tool call: allow, deny, or ask the person (escalations) |
| `PostToolUse` | Evaluates the tool result against output rules |
| `SessionEnd` | Reports the session ended |

The key is stored at `~/.aegis/claude-code/credentials.json` (mode 0600). Requires
`python3` (standard library only). If Aegis is unreachable, the plugin follows the
organization's setting: allow with a warning, or block. A suspended or revoked
subscription always blocks.

To make the plugin mandatory, deploy it through Claude Code managed settings
(`enabledPlugins` and `strictKnownMarketplaces`) so people can't disable it.
