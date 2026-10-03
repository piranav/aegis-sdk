---
name: connect
description: Connect this Claude Code to your organization's Aegis with the code from your IT invite.
argument-hint: <code>
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

!`python3 "${CLAUDE_PLUGIN_ROOT}/scripts/aegis.py" connect $ARGUMENTS`

Report the result above to the user in one or two sentences. Do not run anything else.
