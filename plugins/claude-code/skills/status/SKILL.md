---
name: status
description: Show whether this Claude Code is connected to Aegis and which organization manages it.
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

!`python3 "${CLAUDE_PLUGIN_ROOT}/scripts/aegis.py" status`

Relay the status above to the user briefly. Do not run anything else.
