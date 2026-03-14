"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class MCPServerConnection:
    """Manages a single MCP server connection with reconnect support."""

    def __init__(self, name: str, cfg):
        self._name = name
        self._cfg = cfg
        self._stack: AsyncExitStack | None = None
        self.session = None

    async def connect(self) -> list:
        """Open transport + session. Returns list of tool defs."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        cfg = self._cfg
        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            transport_type = cfg.type
            if not transport_type:
                if cfg.command:
                    transport_type = "stdio"
                elif cfg.url:
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    raise ValueError(f"MCP server '{self._name}': no command or url configured")

            if transport_type == "stdio":
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                raise ValueError(
                    f"MCP server '{self._name}': unknown transport type '{transport_type}'"
                )

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = await session.list_tools()

        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self.session = session
        return tools.tools

    async def reconnect(self) -> None:
        """Tear down the current connection and re-establish it."""
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
        self._stack = None
        self.session = None
        await self.connect()
        logger.info("MCP server '{}': reconnected successfully", self._name)

    async def aclose(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self.session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(
        self,
        connection: MCPServerConnection,
        server_name: str,
        tool_def,
        tool_timeout: int = 30,
    ):
        self._connection = connection
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import McpError, types

        async def _call():
            session = self._connection.session
            if session is None:
                raise RuntimeError("MCP session is not connected")
            return await asyncio.wait_for(
                session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )

        try:
            result = await _call()
        except McpError as exc:
            code = getattr(getattr(exc, "error", None), "code", None)
            if code == 32600:
                logger.warning(
                    "MCP server session terminated for '{}', reconnecting...", self._name
                )
                try:
                    await self._connection.reconnect()
                    result = await _call()
                except Exception as retry_exc:
                    logger.exception(
                        "MCP tool '{}' failed after reconnect: {}: {}",
                        self._name,
                        type(retry_exc).__name__,
                        retry_exc,
                    )
                    return f"(MCP tool call failed after reconnect: {type(retry_exc).__name__})"
            else:
                logger.exception(
                    "MCP tool '{}' failed: {}: {}", self._name, type(exc).__name__, exc
                )
                return f"(MCP tool call failed: {type(exc).__name__})"
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return f"(MCP tool call timed out after {self._tool_timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
            # Re-raise only if our task was externally cancelled (e.g. /stop).
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            logger.exception(
                "MCP tool '{}' failed: {}: {}", self._name, type(exc).__name__, exc
            )
            return f"(MCP tool call failed: {type(exc).__name__})"

        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    for name, cfg in mcp_servers.items():
        try:
            conn = MCPServerConnection(name, cfg)
            tools = await conn.connect()
            await stack.enter_async_context(conn)

            for tool_def in tools:
                wrapper = MCPToolWrapper(conn, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)

            logger.info("MCP server '{}': connected, {} tools registered", name, len(tools))
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
