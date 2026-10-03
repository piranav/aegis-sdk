---
name: disconnect
description: Disconnect this Claude Code from Aegis. Your IT team is notified.
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

!`python3 "${CLAUDE_PLUGIN_ROOT}/scripts/aegis.py" disconnect`

Relay the result above to the user briefly. Do not run anything else.
