"""What a coding assistant is set up with on this machine, as an Aegis manifest.

At session start each assistant's scanner reads the configuration files the assistant
itself reads (MCP servers, plugins, skills, subagents, the default model) and reports
them, so IT sees each person's tooling. What the assistant then actually *uses* needs
no scanning: every tool call is evaluated by Aegis, which resolves MCP, skill, and
subagent calls from their names.

Only names and locations leave the machine. Server environment variables, headers,
arguments, and tokens are never read into the manifest, and URLs are reduced to
scheme, host, and path. Files that are missing or malformed are skipped silently: a
scanner must never get in the way of starting a session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - older system Pythons
    tomllib = None

MAX_COMPONENTS = 1500
MAX_SKILLS_PER_DIR = 200
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)


# ----------------------------------------------------------------- component helpers


def component(kind, name, children=None, **fields):
    node = {"kind": kind, "name": str(name)[:200]}
    for key, value in fields.items():
        if value not in (None, "", {}):
            node[key] = value
    if children:
        node["children"] = children
    return node


def locator(value):
    """URLs keep scheme, host, and path; commands keep only the executable name."""
    if not value or not isinstance(value, str):
        return None
    parts = urlsplit(value.strip())
    if parts.scheme and parts.netloc:
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))
    executable = value.strip().split()[0]
    return executable.rstrip("/").rsplit("/", 1)[-1] or executable


def mcp_server(name, config, **attributes):
    config = config if isinstance(config, dict) else {}
    url = config.get("url") or config.get("serverUrl")
    transport = config.get("type") or config.get("transport") or ("http" if url else "stdio")
    if config.get("enabled") is False or config.get("disabled") is True:
        attributes["enabled"] = False
    return component(
        "mcp_server",
        mcp_key(name),
        transport=str(transport),
        # ``command`` is a path without arguments (those live in ``args``), so its
        # file name is safe to take even when the install path contains spaces.
        locator=locator(url) if url else _executable(config.get("command")),
        attributes={k: v for k, v in attributes.items() if v is not None},
    )


def _executable(command):
    if not isinstance(command, str) or not command.strip():
        return None
    return command.strip().rstrip("/").rsplit("/", 1)[-1].split()[0] or None


def mcp_key(name):
    """Assistants name MCP tools ``mcp__<server>__<tool>`` with this normalization."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))


def read_json(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_toml(path):
    try:
        text = Path(path).read_text()
    except OSError:
        return {}
    if tomllib is not None:
        try:
            return tomllib.loads(text)
        except ValueError:
            return {}
    return _minimal_toml(text)


def _minimal_toml(text):
    """Tables and simple ``key = value`` lines: enough for MCP, plugin, and model keys."""
    data, table = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        header = re.match(r"^\[([^\[\]]+)\]$", line)
        if header:
            table = data
            for part in re.findall(r'"([^"]+)"|([^.\s]+)', header.group(1)):
                table = table.setdefault(part[0] or part[1], {})
            continue
        pair = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*(.+)$", line)
        if pair:
            value = pair.group(2).strip()
            if value in ("true", "false"):
                parsed = value == "true"
            elif value.startswith('"') and value.endswith('"'):
                parsed = value[1:-1]
            else:
                continue
            (table if table is not None else data)[pair.group(1)] = parsed
    return data


def frontmatter(path):
    try:
        text = Path(path).read_text(errors="ignore")[:8000]
    except OSError:
        return {}
    match = FRONTMATTER.match(text)
    fields = {}
    for line in (match.group(1) if match else "").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in ("name", "description", "model"):
            fields[key.strip()] = value.strip().strip("\"'")
    return fields


def skills_in(directory, by_folder=False, **attributes):
    """Each ``<dir>/<skill>/SKILL.md`` is a skill named by its frontmatter or folder.

    ``by_folder`` names it by folder only: Codex has no skill tool and opens a skill by
    reading its ``SKILL.md``, so the folder is the name Aegis sees it used by.
    """
    found = []
    try:
        entries = sorted(Path(directory).iterdir())[:MAX_SKILLS_PER_DIR]
    except OSError:
        return found
    for entry in entries:
        manifest = entry / "SKILL.md"
        if not manifest.is_file():
            continue
        meta = frontmatter(manifest)
        found.append(
            component(
                "skill",
                entry.name if by_folder else meta.get("name") or entry.name,
                description=meta.get("description"),
                fingerprint=_digest(manifest),
                attributes=attributes or None,
            )
        )
    return found


def agents_in(directory, **attributes):
    """Each ``<dir>/*.md`` is a subagent definition."""
    found = []
    try:
        entries = sorted(Path(directory).glob("*.md"))[:MAX_SKILLS_PER_DIR]
    except OSError:
        return found
    for entry in entries:
        meta = frontmatter(entry)
        found.append(
            component(
                "subagent",
                meta.get("name") or entry.stem,
                description=meta.get("description"),
                fingerprint=_digest(entry),
                attributes=attributes or None,
            )
        )
    return found


def plugin_contents(name, root, manifest_dir, server_name, skills_by_folder=False):
    """A plugin's own skills and subagents, plus the MCP servers it starts.

    Skills and subagents nest under the plugin because assistants invoke them as
    ``<plugin>:<name>``. Plugin MCP servers are reported top-level under the name the
    assistant gives their tools (``server_name(plugin, server)``) so observed calls match.
    """
    root = Path(root)
    meta = read_json(root / manifest_dir / "plugin.json")
    children = skills_in(root / "skills", by_folder=skills_by_folder) + agents_in(root / "agents")
    servers = read_json(root / ".mcp.json")
    servers = servers.get("mcpServers", servers) if isinstance(servers, dict) else {}
    mcp = [
        mcp_server(server_name(name, server), config, plugin=name)
        for server, config in servers.items()
        if isinstance(config, dict)
    ]
    plugin = component(
        "plugin",
        name,
        children,
        description=meta.get("description"),
        version=meta.get("version") and str(meta.get("version")),
    )
    return [plugin, *mcp]


def _digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]
    except OSError:
        return None


# ------------------------------------------------------------------------- scanners


class Scanner:
    """Reads one assistant's configuration. Subclasses implement ``components``."""

    framework = ""

    @staticmethod
    def plugin_server_name(plugin, server):
        """The server name an assistant uses in ``mcp__<server>__<tool>`` for plugin servers."""
        return server

    def manifest(self, cwd="", model=""):
        cwd = cwd or os.getcwd()
        components = self.components(cwd)
        if model:
            components = [c for c in components if c["kind"] != "model"]
            components.append(component("model", model))
        return {
            # Each project is reported separately so one project's manifest never
            # undeclares the servers another project configured.
            "scope": "project:" + hashlib.sha256(str(cwd).encode()).hexdigest()[:16],
            "framework": self.framework,
            "components": _dedupe(components)[:MAX_COMPONENTS],
        }

    def components(self, cwd):
        raise NotImplementedError


class ClaudeCodeScanner(Scanner):
    framework = "claude-code"

    @staticmethod
    def plugin_server_name(plugin, server):
        # Claude Code namespaces plugin servers: mcp__plugin_<plugin>_<server>__<tool>.
        return f"plugin_{plugin}_{server}"

    def components(self, cwd):
        home = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home())
        claude_dir = home / ".claude" if not os.environ.get("CLAUDE_CONFIG_DIR") else home
        project = Path(cwd)
        global_config = read_json(home / ".claude.json")
        settings = [
            read_json(claude_dir / "settings.json"),
            read_json(project / ".claude" / "settings.json"),
            read_json(project / ".claude" / "settings.local.json"),
        ]
        found = []

        model = next((s.get("model") for s in reversed(settings) if s.get("model")), None)
        if model:
            found.append(component("model", model, provider="anthropic"))

        project_entry = (global_config.get("projects") or {}).get(str(project)) or {}
        for scope, servers in (
            ("user", global_config.get("mcpServers")),
            ("local", project_entry.get("mcpServers")),
            ("project", read_json(project / ".mcp.json").get("mcpServers")),
        ):
            for name, config in (servers or {}).items():
                found.append(mcp_server(name, config, scope=scope))

        enabled = {}
        for s in settings:
            enabled.update(s.get("enabledPlugins") or {})
        installed = (
            read_json(claude_dir / "plugins" / "installed_plugins.json").get("plugins") or {}
        )
        for key, on in enabled.items():
            if not on:
                continue
            installs = installed.get(key) or []
            path = installs[-1].get("installPath") if installs else None
            name = key.split("@", 1)[0]
            if path:
                found.extend(
                    plugin_contents(name, path, ".claude-plugin", self.plugin_server_name)
                )
            else:
                found.append(component("plugin", name))

        for base, scope in ((claude_dir, "user"), (project / ".claude", "project")):
            found.extend(skills_in(base / "skills", scope=scope))
            found.extend(agents_in(base / "agents", scope=scope))
        return found


class CodexScanner(Scanner):
    framework = "codex"

    def components(self, cwd):
        codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        project = Path(cwd)
        config = read_toml(codex_home / "config.toml")
        project_config = read_toml(project / ".codex" / "config.toml")
        found = []

        model = project_config.get("model") or config.get("model")
        if model:
            found.append(component("model", model, provider="openai"))

        for scope, conf in (("user", config), ("project", project_config)):
            for name, server in (conf.get("mcp_servers") or {}).items():
                found.append(mcp_server(name, server, scope=scope))

        plugins = {**(config.get("plugins") or {}), **(project_config.get("plugins") or {})}
        for key, settings in plugins.items():
            if isinstance(settings, dict) and settings.get("enabled") is False:
                continue
            name, _, marketplace = key.partition("@")
            path = _latest_version(codex_home / "plugins" / "cache" / marketplace / name)
            if path is not None:
                # Codex keeps a plugin server's own name (mcp__codex_app__...).
                found.extend(
                    plugin_contents(
                        name, path, ".codex-plugin", self.plugin_server_name, skills_by_folder=True
                    )
                )
            else:
                found.append(component("plugin", name))

        for base, scope in (
            (codex_home / "skills", "user"),
            (Path.home() / ".agents" / "skills", "user"),
            (project / ".agents" / "skills", "project"),
            (project / ".codex" / "skills", "project"),
        ):
            found.extend(skills_in(base, by_folder=True, scope=scope))
        return found


def _latest_version(plugin_dir):
    try:
        versions = [p for p in Path(plugin_dir).iterdir() if p.is_dir()]
    except OSError:
        return None
    return max(versions, key=lambda p: p.stat().st_mtime) if versions else None


def _dedupe(components):
    """One entry per (kind, name); later scopes (project) refine earlier ones (user)."""
    merged = {}
    for item in components:
        merged[(item["kind"], item["name"])] = item
    return sorted(merged.values(), key=lambda c: (c["kind"], c["name"]))


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
