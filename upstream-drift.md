# Upstream Drift Log

## Last sync: 2026-03-20

**Branch**: feature-forksync-v3  
**Upstream ref**: upstream/main @ 5424551

### Changes pulled in
~130 commits. Notable upstream changes:
- Multi-provider WebSearchTool (duckduckgo/tavily/searxng/jina/brave)
- SSRF protection (`security/network.py`)
- LangSmith integration in litellm_provider
- MCP `enabledTools` filtering (replaces `allowTools`)
- Channel plugin architecture (`extra="allow"` in ChannelsConfig)
- Async background memory consolidation
- onboard wizard (`--wizard` flag)
- ExecTool `enable` flag
- MCP nullable JSON schema params fix
- `build_messages` sync API (we kept `build_messages_full` alongside)
- `_TOOL_ARG_MAX_CHARS` removed upstream (we kept it — Qwen Jinja compat)
- Memory compaction removed upstream (we kept it — optional, uses ChromaDB)

### Custom patches preserved
- `suppressToolsParam` + `_extract_text_tool_calls` + EOS strip (litellm_provider)
- `_TOOL_ARG_MAX_CHARS` truncation (context.py)
- `build_messages_full` async variant (context.py)
- SysmonTool + NvidiaEscalateTool (loop.py)
- @mention stripping + all !commands (loop.py)
- MCPServerConnection + auto-reconnect McpError 32600 (mcp.py)
- allow_bots in DiscordConfig (discord.py)
- searxng_url fallback in _search_searxng (web.py)
- HttpConfig in schema (http.py still uses it)

### Schema breaking changes
- `allowTools` in mcpServers config → `enabledTools` (default: `["*"]` = all)
  Bot configs using `"allowTools"` need to be renamed to `"enabledTools"`
- Channel configs no longer in schema (all define their own) — no action needed
