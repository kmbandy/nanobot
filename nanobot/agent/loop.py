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
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new or !reset — Clear session history\n/stop — Stop current task\n/help — Show this\n!status — Fleet GPU/RAM/agent status\n!tasks — Pending pipeline tasks\n!ping — Bot liveness + model + uptime\n!brief — On-demand overnight summary\n!restart <bot> — Restart eng-1 / eng-2 / arch-1\n!search <query> — Web search, 5 results, stops\n!arxiv <query> — arxiv search, 5 papers, stops\n!reddit <sub> <topic> — Subreddit search, 5 posts, stops")

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
