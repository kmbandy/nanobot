"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.nvidia_escalate import NvidiaEscalateTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.sysmon import SysmonTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.config.schema import CfCrawlConfig
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService

# ── Discord fleet commands (!status, !tasks) ───────────────────────────────────

async def _cmd_status() -> str:
    """Collect and format fleet status for Discord."""
    import sys
    collector = Path.home() / "mad-lab-scripts" / "data_collector.py"
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(collector),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        data = json.loads(stdout)
    except Exception as e:
        return f"❌ Status unavailable: {e}"

    lines = ["🤖 **mad-lab fleet status**"]

    gpus = data.get("gpu", [])
    if gpus:
        gpu_parts = []
        for g in gpus:
            gpu_parts.append(
                f"{g['name']}: {g['utilization']}% | {g['vram_used']}/{g['vram_total']}MB | {g['temperature']}°C"
            )
        lines.append("**GPU** — " + " | ".join(gpu_parts))

    sys_ = data.get("system", {})
    lines.append(
        f"**System** — CPU: {sys_.get('cpu_usage', 0):.1f}% | "
        f"RAM: {sys_.get('ram_used', 0) // 1024:.1f}/{sys_.get('ram_total', 0) // 1024:.0f}GB | "
        f"Up: {sys_.get('uptime_human', '?')}"
    )

    agents = data.get("agents", [])
    if agents:
        agent_parts = []
        for a in agents:
            icon = "✅" if a["status"] == "running" else "⏸"
            agent_parts.append(f"{icon} {a['name']}")
        lines.append("**Agents** — " + " | ".join(agent_parts))

    return "\n".join(lines)


def _build_reddit_prompt(raw: str) -> str:
    """Parse '!reddit <subreddit> <subject...>' and return a scoped LLM prompt."""
    # Strip the command prefix — handle both plain and mention-prefixed variants
    # e.g. "!reddit localllama best coding model" or "@bot !reddit localllama ..."
    text = raw.strip()
    # Find the !reddit token and take everything after it
    idx = text.lower().find("!reddit ")
    after = text[idx + len("!reddit "):].strip() if idx != -1 else text

    parts = after.split(None, 1)
    if len(parts) < 2:
        subreddit = parts[0].lstrip("r/") if parts else "unknown"
        subject = "recent top posts"
    else:
        subreddit = parts[0].lstrip("r/")
        subject = parts[1]

    return (
        f'Search r/{subreddit} for posts related to "{subject}". '
        f"Find the 5 most relevant or popular posts. "
        f"For each one write a 2-3 sentence summary. "
        f"Rules: use web_search with site:reddit.com/r/{subreddit} — "
        f"do at most 3 searches, do not follow external links, "
        f"do not write to memory. "
        f"Once you have 5 summaries, stop immediately and present them."
    )


def _build_explain_prompt(raw: str) -> str:
    """Parse '!explain <code or description>' and return a focused explanation prompt."""
    idx = raw.lower().find("!explain ")
    content = raw[idx + len("!explain "):].strip() if idx != -1 else raw.strip()
    return (
        f"Explain the following clearly and concisely:\n\n{content}\n\n"
        f"If it's code: describe what it does, how it works, and any gotchas. "
        f"If it's a concept: give a plain-English explanation with a brief example. "
        f"Keep it focused — no tangents. Do not write to memory."
    )


async def _cmd_mem(raw: str) -> str:
    """Search ChromaDB and format top results for Discord."""
    idx = raw.lower().find("!mem ")
    query = raw[idx + len("!mem "):].strip() if idx != -1 else raw.strip()
    if not query:
        return "Usage: `!mem <search query>`"
    try:
        import chromadb
        from pathlib import Path as _Path
        client = chromadb.PersistentClient(path=str(_Path.home() / ".mad-lab-mcp" / "chromadb"))
        col = client.get_collection("memory")
        r = col.query(
            query_texts=[query],
            n_results=min(5, col.count() or 1),
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        return f"❌ Memory search failed: {e}"

    ids       = r["ids"][0]
    metas     = r["metadatas"][0]
    docs      = r["documents"][0]
    distances = r["distances"][0]

    if not ids:
        return f"🧠 No results for `{query}`"

    lines = [f"🧠 **Memory search:** `{query}`\n"]
    for doc_id, meta, doc, dist in zip(ids, metas, docs, distances):
        score = max(0.0, 1.0 - dist)
        dtype = meta.get("type", "?")
        ts    = meta.get("timestamp", "")[:10]
        preview = doc.replace("\n", " ").strip()[:120]
        lines.append(f"`{doc_id[:8]}` [{dtype}] {score:.2f}  *{ts}*\n{preview}\n")

    return "\n".join(lines)[:2000]


def _cmd_remind(raw: str) -> str:
    """Schedule a Discord reminder. Format: !remind <time> <message>
    Time: 5m, 30m, 1h, 2h, etc."""
    import re, json
    from pathlib import Path as _Path

    idx = raw.lower().find("!remind ")
    after = raw[idx + len("!remind "):].strip() if idx != -1 else ""

    # Parse time token: 5m, 30m, 1h, 2h30m, etc.
    m = re.match(r'^((?:\d+h)?(?:\d+m)?)\s+(.+)', after, re.IGNORECASE)
    if not m or not m.group(1):
        return (
            "Usage: `!remind <time> <message>`\n"
            "Examples: `!remind 20m check on arch-1`  `!remind 1h30m restart scraper`"
        )

    time_str = m.group(1).lower()
    reminder_msg = m.group(2).strip()

    # Parse total minutes
    hours   = int(re.search(r'(\d+)h', time_str).group(1)) if 'h' in time_str else 0
    minutes = int(re.search(r'(\d+)m', time_str).group(1)) if 'm' in time_str else 0
    total_minutes = hours * 60 + minutes
    if total_minutes < 1:
        return "❌ Minimum remind time is 1 minute."
    if total_minutes > 1440:
        return "❌ Maximum remind time is 24 hours."

    # Write a one-shot reminder script
    reminders_dir = _Path.home() / ".mad-lab-mcp" / "reminders"
    reminders_dir.mkdir(exist_ok=True)

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    fire_at = _dt.now(_tz.utc) + _td(minutes=total_minutes)
    stamp   = fire_at.strftime("%Y%m%d_%H%M%S")
    script  = reminders_dir / f"remind_{stamp}.py"

    # Load Discord config
    conf_path = _Path.home() / ".mad-lab-mcp" / "supervisor.conf"
    nanobot_conf = _Path.home() / ".nanobot" / "config.json"
    conf = {}
    if conf_path.exists():
        for line in conf_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                conf[k.strip()] = v.strip()
    token = conf.get("DISCORD_TOKEN", "")
    if not token:
        try:
            c = json.loads(nanobot_conf.read_text())
            token = c.get("channels", {}).get("discord", {}).get("token", "")
        except Exception:
            pass
    channel = conf.get("DISCORD_ALERT_CHANNEL", "")

    script.write_text(
        f'#!/usr/bin/env python3\n'
        f'import httpx, os\n'
        f'from pathlib import Path\n'
        f'msg = "⏰ **Reminder:** {reminder_msg}"\n'
        f'r = httpx.post("https://discord.com/api/v10/channels/{channel}/messages",\n'
        f'    headers={{"Authorization": "Bot {token}"}},\n'
        f'    json={{"content": msg}}, timeout=10)\n'
        f'# Self-destruct\n'
        f'Path(__file__).unlink(missing_ok=True)\n'
        f'# Remove own cron entry\n'
        f'import subprocess\n'
        f'cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout\n'
        f'cron = "\\n".join(l for l in cron.splitlines() if "{script.name}" not in l)\n'
        f'subprocess.run(["crontab", "-"], input=cron, text=True)\n'
    )
    script.chmod(0o755)

    # Add cron entry at the right time
    cron_min  = fire_at.minute
    cron_hour = fire_at.hour
    cron_dom  = fire_at.day
    cron_mon  = fire_at.month
    import subprocess as _sp
    current = _sp.run(["crontab", "-l"], capture_output=True, text=True).stdout
    new_line = f"{cron_min} {cron_hour} {cron_dom} {cron_mon} * /usr/bin/python3 {script} >> /home/kmbandy/.mad-lab-mcp/reminders/remind.log 2>&1"
    _sp.run(["crontab", "-"], input=current.rstrip() + "\n" + new_line + "\n", text=True)

    human = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    fire_local = fire_at.strftime("%H:%M UTC")
    return f"⏰ Reminder set for **{human}** from now ({fire_local}): *{reminder_msg}*"


def _build_python_prompt(raw: str) -> str:
    """Parse '!python <description>' and return a focused code generation prompt."""
    idx = raw.lower().find("!python ")
    description = raw[idx + len("!python "):].strip() if idx != -1 else raw.strip()
    return (
        f"Write a Python script or function for the following: {description}\n\n"
        f"Requirements:\n"
        f"- Include a brief docstring explaining what it does\n"
        f"- Add minimal inline comments only where the logic isn't obvious\n"
        f"- Make it runnable as-is (include example usage or a __main__ block if appropriate)\n"
        f"- Use only the standard library unless a third-party package is clearly necessary\n"
        f"- Keep it concise — no unnecessary boilerplate\n"
        f"Do not write to memory. Just output the code."
    )


def _build_search_prompt(raw: str) -> str:
    """Parse '!search <query>' and return a guardrailed web search prompt."""
    idx = raw.lower().find("!search ")
    query = raw[idx + len("!search "):].strip() if idx != -1 else raw.strip()
    return (
        f'Search the web for "{query}". '
        f"Find the 5 most relevant results. "
        f"For each one write a 2-3 sentence summary including the source. "
        f"Rules: do at most 3 searches, do not write to memory. "
        f"Once you have 5 summaries, stop immediately and present them."
    )


def _build_arxiv_prompt(raw: str) -> str:
    """Parse '!arxiv <query>' and return an arxiv search prompt."""
    idx = raw.lower().find("!arxiv ")
    query = raw[idx + len("!arxiv "):].strip() if idx != -1 else raw.strip()
    return (
        f'Search arxiv.org for papers related to "{query}". '
        f"Use web_search with site:arxiv.org. "
        f"Find the 5 most relevant recent papers. "
        f"For each paper give: title, one-sentence summary, and URL. "
        f"Rules: do at most 3 searches, do not write to memory. "
        f"Once you have 5 papers, stop immediately and present them."
    )


def _build_summarize_prompt(raw: str) -> str | None:
    """Fetch URL and build a summarize prompt. Returns None on fetch error."""
    import httpx

    idx = raw.lower().find("!summarize ")
    url = raw[idx + len("!summarize "):].strip() if idx != -1 else raw.strip()
    url = url.split()[0] if url else ""
    if not url:
        return None

    try:
        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return f"Please summarize the following web content from {url}:\n\n{text[:8000]}"


async def _cmd_gh(raw: str) -> str:
    """Fetch GitHub repo info and latest release."""
    import httpx

    idx = raw.lower().find("!gh ")
    slug = raw[idx + len("!gh "):].strip() if idx != -1 else raw.strip()
    slug = slug.split()[0] if slug else ""
    if not slug:
        return "Usage: `!gh <owner/repo>`"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            repo_resp = await client.get(
                f"https://api.github.com/repos/{slug}",
                headers={"Accept": "application/vnd.github+json"},
            )
            repo_resp.raise_for_status()
            repo = repo_resp.json()

            release_tag = "none"
            release_date = ""
            try:
                rel_resp = await client.get(
                    f"https://api.github.com/repos/{slug}/releases/latest",
                    headers={"Accept": "application/vnd.github+json"},
                )
                if rel_resp.status_code == 200:
                    rel = rel_resp.json()
                    release_tag = rel.get("tag_name", "?")
                    release_date = (rel.get("published_at", "") or "")[:10]
            except Exception:
                pass
    except Exception:
        return f"❌ Repo not found or API error: `{slug}`"

    description = repo.get("description") or "No description"
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    issues = repo.get("open_issues_count", 0)
    html_url = repo.get("html_url", f"https://github.com/{slug}")
    full_name = repo.get("full_name", slug)
    release_line = f"📅 Latest: {release_tag} ({release_date})" if release_date else f"📅 Latest: {release_tag}"
    return (
        f"📦 **{full_name}** — {description}\n"
        f"⭐ {stars}  🍴 {forks}  🐛 {issues} open issues\n"
        f"{release_line}\n"
        f"🔗 {html_url}"
    )


def _cmd_diff() -> str:
    """Show nanobot changes since upstream/main."""
    import subprocess

    nanobot_root = Path(__file__).parent.parent.parent

    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""

    log_out = _run(["git", "-C", str(nanobot_root), "log", "--oneline", "upstream/main..HEAD"])
    if log_out:
        diff_stat = _run(["git", "-C", str(nanobot_root), "diff", "--stat", "upstream/main..HEAD"])
        header = "📋 **nanobot changes since upstream/main:**"
    else:
        log_out = _run(["git", "-C", str(nanobot_root), "log", "--oneline", "-10"])
        diff_stat = ""
        header = "📋 **nanobot log** (last 10 — upstream/main not found):"

    result = f"{header}\n```\n{log_out[:1800]}\n```"
    if diff_stat:
        result += f"\n```\n{diff_stat[:400]}\n```"
    return result


def _build_code_summary_prompt(raw: str) -> str | None:
    """Read a file from mad-lab repos and build an LLM summary prompt."""
    idx = raw.lower().find("!mad-code-summary ")
    path_str = raw[idx + len("!mad-code-summary "):].strip() if idx != -1 else raw.strip()
    path_str = path_str.split()[0] if path_str else ""
    if not path_str:
        return None

    # Resolve path — check as-is, then relative to ~/mad-lab-mcp and ~/nanobot
    candidates = [
        Path(path_str),
        Path.home() / path_str,
        Path.home() / "mad-lab-mcp" / path_str,
        Path.home() / "nanobot" / path_str,
    ]
    resolved: Path | None = None
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                resolved = c
                break
        except Exception:
            continue

    if resolved is None:
        return f"__NOT_FOUND__:{path_str}"

    try:
        content = resolved.read_text(errors="replace")[:6000]
    except Exception:
        return None

    return (
        f"Please summarize the purpose and functionality of this file: `{resolved}`\n\n"
        f"Focus on: what it does, key functions/classes, how it fits in the project.\n\n"
        f"```\n{content}\n```"
    )


_RESTART_TARGETS: dict[str, tuple[str, str]] = {
    "eng-1":  ("nanobot-eng-1",  "nanobot.service"),
    "eng-2":  ("nanobot-eng-2",  "nanobot-8b.service"),
    "arch-1": ("nanobot-arch-1", "nanobot-27b.service"),
}


def _cmd_restart(raw: str) -> str:
    """Handle '!restart <bot>' — restart a fleet service."""
    import subprocess, os
    from pathlib import Path as _Path

    idx = raw.lower().find("!restart ")
    target = raw[idx + len("!restart "):].strip().lower() if idx != -1 else ""

    if target not in _RESTART_TARGETS:
        valid = ", ".join(_RESTART_TARGETS)
        return f"❌ Unknown bot `{target}`. Valid targets: {valid}"

    friendly, service = _RESTART_TARGETS[target]

    # Fix D-Bus so systemctl --user works
    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        candidate = f"/run/user/{os.getuid()}/bus"
        if _Path(candidate).exists():
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={candidate}"

    # Clear session files before restart to prevent loop resume
    session_dirs = {
        "nanobot.service":     _Path.home() / ".nanobot"     / "workspace" / "sessions",
        "nanobot-8b.service":  _Path.home() / ".nanobot-8b"  / "workspace" / "sessions",
        "nanobot-27b.service": _Path.home() / ".nanobot-27b" / "workspace" / "sessions",
    }
    sessions_dir = session_dirs.get(service)
    cleared = 0
    if sessions_dir and sessions_dir.exists():
        for f in sessions_dir.glob("*.json*"):
            try:
                f.unlink()
                cleared += 1
            except Exception:
                pass

    r = subprocess.run(
        ["systemctl", "--user", "restart", service],
        capture_output=True,
    )
    if r.returncode != 0:
        err = r.stderr.decode().strip()
        return f"❌ Failed to restart `{friendly}`: {err}"

    session_note = f" (cleared {cleared} session file(s))" if cleared else ""
    return f"♻️ `{friendly}` restarted{session_note}."


# ── !new-nanobot wizard ────────────────────────────────────────────────────────

_WIZARD_PATH = Path.home() / "mad-lab-mcp" / "shared" / "new-nanobot-wizard.json"

_WIZARD_LLAMA_CUDA = """\
[Unit]
Description=llama-server: {model_name} CUDA (mad-lab-{bot_name})
After=network.target

[Service]
ExecStart=/home/kmbandy/llama.cpp/build/bin/llama-server \\
    --model {model_path} \\
    --n-gpu-layers 999 \\
    --ctx-size 32768 \\
    --cache-type-k q4_0 \\
    --cache-type-v q4_0 \\
    --parallel 1 \\
    --host 127.0.0.1 \\
    --port {llama_port}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

_WIZARD_LLAMA_ROCM = """\
[Unit]
Description=llama-server: {model_name} ROCm RX480 (mad-lab-{bot_name})
After=network.target

[Service]
Environment=HIP_VISIBLE_DEVICES=0
Environment=ROCBLAS_TENSILE_LIBPATH=/tmp/fake-tensile
ExecStartPre=/bin/bash -c "mkdir -p /tmp/fake-tensile && cp /opt/rocm-6.2.4/lib/rocblas/library/TensileLibrary_lazy_gfx900.dat /tmp/fake-tensile/TensileLibrary_lazy_gfx803.dat"
ExecStart=/home/kmbandy/llama.cpp/build-rocm/bin/llama-server \\
    --model {model_path} \\
    --n-gpu-layers 999 \\
    --ctx-size 24576 \\
    --cache-type-k q8_0 \\
    --cache-type-v q8_0 \\
    --parallel 1 \\
    --host 127.0.0.1 \\
    --port {llama_port} \\
    --no-warmup
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

_WIZARD_LLAMA_CPU = """\
[Unit]
Description=llama-server: {model_name} CPU (mad-lab-{bot_name})
After=network.target

[Service]
Environment=CUDA_VISIBLE_DEVICES=
ExecStart=/home/kmbandy/llama.cpp/build/bin/llama-server \\
    --model {model_path} \\
    --n-gpu-layers 0 \\
    --ctx-size 16384 \\
    --parallel 1 \\
    --host 127.0.0.1 \\
    --port {llama_port}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

_WIZARD_NANOBOT_SVC = """\
[Unit]
Description=mad-lab-{bot_name} Discord agent
After=network.target llama-server-{bot_name}.service
Requires=llama-server-{bot_name}.service

[Service]
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/python3 /home/kmbandy/.local/bin/nanobot gateway --config /home/kmbandy/.nanobot-{bot_name}/config.json
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _wizard_load() -> dict | None:
    try:
        if _WIZARD_PATH.exists():
            return json.loads(_WIZARD_PATH.read_text())
    except Exception:
        pass
    return None


def _wizard_save(state: dict) -> None:
    _WIZARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WIZARD_PATH.write_text(json.dumps(state, indent=2))


def _wizard_clear() -> None:
    try:
        _WIZARD_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _wizard_find_model(query: str) -> Path | None:
    """Reuse mad-hot-swap fuzzy model matching."""
    import re as _re
    models_dir = Path.home() / "models"
    candidates = sorted(models_dir.glob("*.gguf"), key=lambda p: p.name.lower())
    query_lower = query.lower()
    for m in candidates:
        if m.name.lower() == query_lower:
            return m
    tokens = _re.split(r'[\s\-_\.]+', query_lower)
    scored = [m for m in candidates if all(t in m.name.lower() for t in tokens if t)]
    if len(scored) == 1:
        return scored[0]
    if len(scored) > 1:
        return min(scored, key=lambda p: len(p.name))
    return None


def _wizard_list_models() -> str:
    models_dir = Path.home() / "models"
    models = sorted(models_dir.glob("*.gguf"), key=lambda p: p.name.lower())
    if not models:
        return "No models found in ~/models/"
    lines = ["Available models:"]
    for m in models:
        size_mb = m.stat().st_size // (1024 * 1024)
        lines.append(f"  `{m.stem}` ({size_mb} MB)")
    return "\n".join(lines)


def _wizard_generate(state: dict) -> str:
    """Generate all config/service files for the new bot. Returns summary message."""
    bot_name: str = state["name"]
    model_path: str = state["model_path"]
    hardware: str = state["hardware"]
    llama_port: int = int(state["llama_port"])
    gateway_port: int = int(state["gateway_port"])
    group_policy: str = state["group_policy"]

    model_stem = Path(model_path).stem.lower()
    model_name = Path(model_path).name

    # Dirs
    conf_dir = Path.home() / f".nanobot-{bot_name}"
    workspace_dir = conf_dir / "workspace"
    sessions_dir = workspace_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # 1. nanobot config.json — copy eng-1 config, patch fields
    src_conf = Path.home() / ".nanobot" / "config.json"
    try:
        nc = json.loads(src_conf.read_text())
    except Exception:
        nc = {}

    # Patch model + workspace
    nc.setdefault("agents", {}).setdefault("defaults", {})["model"] = model_stem
    nc["agents"]["defaults"]["workspace"] = str(workspace_dir)

    # Patch HTTP port
    nc.setdefault("channels", {}).setdefault("http", {})["port"] = gateway_port
    nc["channels"]["http"]["enabled"] = True

    # Patch gateway port
    nc.setdefault("gateway", {})["port"] = gateway_port

    # Patch Discord group policy
    nc["channels"].setdefault("discord", {})["groupPolicy"] = group_policy

    conf_file = conf_dir / "config.json"
    conf_file.write_text(json.dumps(nc, indent=2))

    # 2. llama-server service file
    llama_templates = {"cuda": _WIZARD_LLAMA_CUDA, "rocm": _WIZARD_LLAMA_ROCM, "cpu": _WIZARD_LLAMA_CPU}
    llama_svc_content = llama_templates[hardware].format(
        bot_name=bot_name,
        model_name=model_name,
        model_path=model_path,
        llama_port=llama_port,
    )
    llama_svc_file = Path.home() / f".config/systemd/user/llama-server-{bot_name}.service"
    llama_svc_file.write_text(llama_svc_content)

    # 3. nanobot service file
    nanobot_svc_content = _WIZARD_NANOBOT_SVC.format(bot_name=bot_name)
    nanobot_svc_file = Path.home() / f".config/systemd/user/nanobot-{bot_name}.service"
    nanobot_svc_file.write_text(nanobot_svc_content)

    return (
        f"✅ **New bot `{bot_name}` created!**\n\n"
        f"  Model: `{model_name}` ({hardware.upper()})\n"
        f"  llama-server port: `{llama_port}`\n"
        f"  Gateway port: `{gateway_port}`\n"
        f"  Config: `~/.nanobot-{bot_name}/config.json`\n\n"
        f"**To start:**\n"
        f"```\n"
        f"systemctl --user daemon-reload\n"
        f"systemctl --user enable --now llama-server-{bot_name}.service\n"
        f"systemctl --user enable --now nanobot-{bot_name}.service\n"
        f"```\n"
        f"_(Edit `~/.nanobot-{bot_name}/config.json` to add MCP servers, Discord token, etc.)_"
    )


def _wizard_step(state: dict, user_input: str) -> tuple[dict | None, str]:
    """Process one wizard step. Returns (new_state, response). new_state=None means done/cancel."""
    step = state.get("step", 1)

    if user_input.lower() in ("cancel", "abort", "quit", "exit"):
        return None, "❌ New bot wizard cancelled."

    if step == 1:
        # Expecting: bot name
        name = user_input.strip().lower()
        if not re.match(r'^[a-z0-9][a-z0-9\-]{1,20}$', name):
            return state, "⚠️ Name must be 2–21 chars, lowercase letters/numbers/hyphens, starting with a letter or number. Try again:"
        if (Path.home() / f".nanobot-{name}").exists():
            return state, f"⚠️ Bot `{name}` already exists (`~/.nanobot-{name}/` found). Choose a different name:"
        new_state = {**state, "step": 2, "name": name}
        model_list = _wizard_list_models()
        return new_state, (
            f"✅ Bot name: `{name}`\n\n"
            f"**Step 2/6** — Which model?\n{model_list}\n\n"
            f"Type the model name or a partial match (e.g. `ministral`, `omnicoder`):"
        )

    if step == 2:
        # Expecting: model query
        model_path = _wizard_find_model(user_input.strip())
        if model_path is None:
            return state, f"⚠️ No model matching `{user_input.strip()}`. Try again (or `!models` to see list):"
        new_state = {**state, "step": 3, "model_path": str(model_path)}
        return new_state, (
            f"✅ Model: `{model_path.name}`\n\n"
            f"**Step 3/6** — Hardware:\n"
            f"  `cuda` — GTX 1070 (CUDA, port 8080)\n"
            f"  `rocm` — RX 480 (ROCm, pure transformers only)\n"
            f"  `cpu`  — CPU only (slow, for small models)\n\n"
            f"Type `cuda`, `rocm`, or `cpu`:"
        )

    if step == 3:
        # Expecting: hardware
        hw = user_input.strip().lower()
        if hw not in ("cuda", "rocm", "cpu"):
            return state, "⚠️ Type `cuda`, `rocm`, or `cpu`:"
        if hw == "rocm":
            model_name = Path(state.get("model_path", "")).name.lower()
            for kw in ("mamba", "rwkv", "jamba", "zamba", "falcon-mamba"):
                if kw in model_name:
                    return state, f"⚠️ `{model_name}` may be SSM/Mamba which crashes on RX 480. Choose `cuda` or `cpu`, or pick a different model. Re-enter hardware:"
        new_state = {**state, "step": 4, "hardware": hw}
        return new_state, (
            f"✅ Hardware: `{hw}`\n\n"
            f"**Step 4/6** — llama-server port? (default: `8084`)\n"
            f"Current ports in use: 8080 (eng-1), 8083 (eng-2), 8082 (lore)\n\n"
            f"Type a port number or press Enter for `8084`:"
        )

    if step == 4:
        # Expecting: llama port
        raw = user_input.strip()
        port = 8084 if not raw or raw.lower() in ("enter", "default", "") else None
        try:
            if raw:
                port = int(raw)
        except ValueError:
            pass
        if port is None or not (1024 <= port <= 65535):
            return state, "⚠️ Enter a valid port number (1024–65535) or leave blank for `8084`:"
        new_state = {**state, "step": 5, "llama_port": port}
        return new_state, (
            f"✅ llama-server port: `{port}`\n\n"
            f"**Step 5/6** — nanobot gateway port? (default: `18794`)\n"
            f"Current: 18790 (eng-1), 18792 (eng-2), 18793 (arch-1)\n\n"
            f"Type a port number or press Enter for `18794`:"
        )

    if step == 5:
        # Expecting: gateway port
        raw = user_input.strip()
        port = 18794 if not raw or raw.lower() in ("enter", "default", "") else None
        try:
            if raw:
                port = int(raw)
        except ValueError:
            pass
        if port is None or not (1024 <= port <= 65535):
            return state, "⚠️ Enter a valid port number (1024–65535) or leave blank for `18794`:"
        new_state = {**state, "step": 6, "gateway_port": port}
        return new_state, (
            f"✅ Gateway port: `{port}`\n\n"
            f"**Step 6/6** — Discord group policy?\n"
            f"  `mention` — bot only responds when @mentioned (default)\n"
            f"  `open`    — bot responds to all messages in channel\n\n"
            f"Type `mention` or `open`:"
        )

    if step == 6:
        # Expecting: group policy
        policy = user_input.strip().lower()
        if policy not in ("mention", "open"):
            return state, "⚠️ Type `mention` or `open`:"
        state["group_policy"] = policy
        try:
            result = _wizard_generate(state)
        except Exception as e:
            return None, f"❌ Failed to generate bot files: {e}"
        return None, result

    return None, "❌ Unknown wizard state. Type `!new-nanobot` to restart."


def _cmd_models() -> str:
    """List available .gguf models in ~/models/."""
    import subprocess
    swap_bin = Path.home() / "mad-lab-mcp" / "bin" / "mad-hot-swap"
    try:
        r = subprocess.run(
            ["python3", str(swap_bin), "--list"],
            capture_output=True, text=True, timeout=10,
        )
        out = r.stdout.strip()
        return f"```\n{out}\n```" if out else "No models found in ~/models/"
    except Exception as e:
        return f"❌ Failed to list models: {e}"


def _cmd_hotswap(raw: str) -> str:
    """Handle '!mad-hot-swap <bot> <model>' — swap model on a bot."""
    import subprocess, os
    from pathlib import Path as _Path

    idx = raw.lower().find("!mad-hot-swap ")
    after = raw[idx + len("!mad-hot-swap "):].strip() if idx != -1 else ""
    parts = after.split(None, 1)
    if len(parts) < 2:
        return "Usage: `!mad-hot-swap <bot> <model>`\nExample: `!mad-hot-swap eng-1 qwen3.5-9b`\nUse `!models` to list available models."

    bot, model_query = parts[0].lower(), parts[1]

    if bot not in ("eng-1", "eng-2"):
        return f"❌ Unknown bot `{bot}`. Valid: eng-1, eng-2"

    # Fix D-Bus so systemctl --user works inside subprocess
    env = os.environ.copy()
    if "DBUS_SESSION_BUS_ADDRESS" not in env:
        candidate = f"/run/user/{os.getuid()}/bus"
        if _Path(candidate).exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={candidate}"

    swap_bin = _Path.home() / "mad-lab-mcp" / "bin" / "mad-hot-swap"
    try:
        r = subprocess.run(
            ["python3", str(swap_bin), bot, model_query],
            capture_output=True, text=True, timeout=60, env=env,
        )
        out = (r.stdout + r.stderr).strip()
        return out if out else ("✅ Swap complete." if r.returncode == 0 else "❌ Swap failed (no output).")
    except Exception as e:
        return f"❌ Hot-swap failed: {e}"


async def _cmd_brief() -> str:
    """Run morning-brief --stdout and return the brief text."""
    import sys
    brief_script = Path.home() / "mad-lab-mcp" / "bin" / "morning-brief"
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(brief_script), "--stdout",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        text = stdout.decode().strip()
        return text or "📨 Brief generated but was empty."
    except Exception as e:
        return f"❌ Brief failed: {e}"


async def _cmd_tasks() -> str:
    """Query ChromaDB for pending tasks and format for Discord."""
    try:
        import chromadb
        from pathlib import Path as _Path
        db_path = str(_Path.home() / ".mad-lab-mcp" / "chromadb")
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection("memory")
        results = col.get(
            where={"$and": [{"type": "task"}, {"status": "pending"}]},
            include=["metadatas", "documents"],
        )
    except Exception as e:
        return f"❌ Tasks unavailable: {e}"

    ids = results.get("ids", [])
    if not ids:
        return "📋 **Pending tasks** — none"

    lines = [f"📋 **Pending tasks** ({len(ids)})"]
    for task_id, meta, doc in zip(ids, results["metadatas"], results["documents"]):
        assignee = meta.get("assignee", "unassigned")
        priority = meta.get("priority", "normal")
        complexity = meta.get("complexity", "?")
        short_id = task_id[:8]
        desc = (doc[:60] + "…") if len(doc) > 60 else doc
        lines.append(f"`{short_id}` [{complexity}/{priority}] → {assignee}: {desc}")

    return "\n".join(lines)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        searxng_url: str | None = None,
        exec_config: ExecToolConfig | None = None,
        nvidia_api_key: str | None = None,
        nvidia_default_model: str | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        cf_crawl_config: CfCrawlConfig | None = None,
        memory_max_chars: int | None = None,
        memory_max_tokens: int | None = None,
        memory_compaction_enabled: bool | None = None,
        sysmon: bool = True,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.searxng_url = searxng_url
        self.exec_config = exec_config or ExecToolConfig()
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_default_model = nvidia_default_model
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.sysmon = sysmon

        self.context = ContextBuilder(
            workspace=workspace,
            provider=provider,
            model=self.model,
            memory_max_chars=memory_max_chars or 8000,
            memory_max_tokens=memory_max_tokens or 2000,
            memory_compaction_enabled=memory_compaction_enabled if memory_compaction_enabled is not None else True,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()

        # Use custom memory settings if provided (stored for potential future use)
        self.memory_max_chars = memory_max_chars or 8000
        self.memory_max_tokens = memory_max_tokens or 2000
        self.memory_compaction_enabled = memory_compaction_enabled if memory_compaction_enabled is not None else True
        self.cf_crawl_config = cf_crawl_config

        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )

        self._register_default_tools()

        # Register CfCrawlTool if config provided
        if cf_crawl_config:
            from nanobot.agent.tools.cf_crawl import CfCrawlTool
            self.tools.register(CfCrawlTool(config=cf_crawl_config))

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy, searxng_url=self.searxng_url))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        if self.sysmon:
            self.tools.register(SysmonTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(NvidiaEscalateTool(
            api_key=self.nvidia_api_key or '',
            default_model=self.nvidia_default_model or 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
        ))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    None,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = await self.context.build_messages_full(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd in ("/new", "/reset", "!reset"):
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        # ── New-nanobot wizard (intercepts all messages while active) ──────────
        if cmd == "!new-nanobot cancel":
            _wizard_clear()
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="❌ New bot wizard cancelled.")

        if cmd == "!new-nanobot":
            _wizard_clear()
            _wizard_save({"step": 1})
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=(
                                      "🤖 **New nanobot wizard** — Step 1/6\n\n"
                                      "What should this bot be called?\n"
                                      "Use lowercase letters, numbers, hyphens (e.g. `researcher`, `eng-3`)\n\n"
                                      "_(Type `cancel` at any step to abort)_"
                                  ))

        _wizard_state = _wizard_load()
        if _wizard_state is not None:
            new_wiz_state, wiz_response = _wizard_step(_wizard_state, msg.content)
            if new_wiz_state is None:
                _wizard_clear()
            else:
                _wizard_save(new_wiz_state)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=wiz_response)

        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new or !reset — Clear session history\n/stop — Stop current task\n/help — Show this\n!status — Fleet GPU/RAM/agent status\n!tasks — Pending pipeline tasks\n!ping — Bot liveness + model + uptime\n!brief — On-demand overnight summary\n!restart <bot> — Restart eng-1 / eng-2 / arch-1\n!models — List models in ~/models/\n!mad-hot-swap <bot> <model> — Swap model (e.g. !mad-hot-swap eng-1 qwen3.5-9b)\n!new-nanobot — Interactive wizard to create a new bot config + service files\n!summarize <url> — Fetch and summarize any webpage\n!gh <owner/repo> — GitHub repo info + latest release\n!diff — Show nanobot changes since upstream sync\n!mad-code-summary <path> — Summarize a file's purpose\n!search <query> — Web search, 5 results, stops\n!arxiv <query> — arxiv search, 5 papers, stops\n!reddit <sub> <topic> — Subreddit search, 5 posts, stops\n!python <description> — Generate a Python script or function\n!explain <code/concept> — Plain-English explanation\n!mem <query> — Search your ChromaDB knowledge base\n!remind <time> <msg> — e.g. !remind 20m check arch-1")

        if cmd == "!status":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=await _cmd_status(),
            )

        if cmd == "!tasks":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=await _cmd_tasks(),
            )

        if cmd == "!ping":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=self._cmd_ping(msg.session_key),
            )

        if cmd == "!brief":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=await _cmd_brief(),
            )

        if cmd.startswith("!reddit "):
            msg.content = _build_reddit_prompt(msg.content)

        elif cmd.startswith("!search "):
            msg.content = _build_search_prompt(msg.content)

        elif cmd.startswith("!arxiv "):
            msg.content = _build_arxiv_prompt(msg.content)

        elif cmd.startswith("!python "):
            msg.content = _build_python_prompt(msg.content)

        elif cmd.startswith("!explain "):
            msg.content = _build_explain_prompt(msg.content)

        if cmd.startswith("!mem "):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=await _cmd_mem(msg.content),
            )

        if cmd == "!diff":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=_cmd_diff(),
            )

        if cmd.startswith("!gh "):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=await _cmd_gh(msg.content),
            )

        if cmd.startswith("!summarize "):
            result = _build_summarize_prompt(msg.content)
            if result is None:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="❌ Failed to fetch URL. Check the address and try again.",
                )
            msg.content = result

        if cmd.startswith("!mad-code-summary "):
            result = _build_code_summary_prompt(msg.content)
            if result is None:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="❌ Failed to read file.",
                )
            if isinstance(result, str) and result.startswith("__NOT_FOUND__:"):
                fname = result.split(":", 1)[1]
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"❌ File not found: `{fname}`\nTry a path relative to ~/mad-lab-mcp/ or ~/nanobot/",
                )
            msg.content = result

        if cmd == "!models":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=_cmd_models(),
            )

        if cmd.startswith("!mad-hot-swap "):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=_cmd_hotswap(msg.content),
            )

        if cmd.startswith("!remind "):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=_cmd_remind(msg.content),
            )

        elif cmd.startswith("!restart "):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=_cmd_restart(msg.content),
            )

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = await self.context.build_messages_full(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _cmd_ping(self, session_key: str) -> str:
        """Return a one-liner pong for !ping, identifying this bot instance."""
        import subprocess

        # Derive friendly bot name from workspace path
        ws = str(self.workspace)
        if ".nanobot-27b" in ws:
            bot_name = "nanobot-arch-1"
            service  = "nanobot-27b.service"
        elif ".nanobot-8b" in ws:
            bot_name = "nanobot-eng-2"
            service  = "nanobot-8b.service"
        else:
            bot_name = "nanobot-eng-1"
            service  = "nanobot.service"

        # State: any active tasks for this session?
        busy = bool(self._active_tasks.get(session_key))
        state = "🔄 working" if busy else "💤 idle"

        # Service uptime via systemctl
        uptime_str = ""
        try:
            out = subprocess.check_output(
                ["systemctl", "--user", "show", service,
                 "--property=ActiveEnterTimestamp"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            # "ActiveEnterTimestamp=Thu 2026-03-19 00:49:42 EDT"
            val = out.split("=", 1)[-1].strip()
            if val and val != "n/a":
                from datetime import datetime, timezone
                # Parse systemd timestamp — try common formats
                for fmt in ("%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S"):
                    try:
                        started = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
                        secs = int((datetime.now(timezone.utc) - started).total_seconds())
                        if secs < 3600:
                            uptime_str = f" | up {secs // 60}m"
                        elif secs < 86400:
                            uptime_str = f" | up {secs // 3600}h {(secs % 3600) // 60}m"
                        else:
                            uptime_str = f" | up {secs // 86400}d {(secs % 86400) // 3600}h"
                        break
                    except ValueError:
                        continue
        except Exception:
            pass

        return f"🏓 **{bot_name}** — {state} | `{self.model}`{uptime_str}"

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            # Skip the tool call hint message added for context
            if role == "user" and isinstance(content, str) and content.startswith("[Tool Call]"):
                continue
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
