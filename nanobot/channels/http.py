"""HTTP channel — accepts POST /v1/chat/completions for direct bot messaging.

Allows external processes (task pollers, other bots, scripts) to send messages
into the agent loop via a simple HTTP API without going through Discord.
Auth is via a shared Bearer token set in config; bind host controls exposure.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import HttpConfig

_HTTP_STATUS = {
    200: "OK",
    202: "Accepted",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    408: "Request Timeout",
    500: "Internal Server Error",
    504: "Gateway Timeout",
}


class HttpChannel(BaseChannel):
    """HTTP channel that exposes a minimal /v1/chat/completions endpoint."""

    name = "http"
    display_name = "HTTP"

    def __init__(self, config: HttpConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: HttpConfig = config
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.config.host,
            self.config.port,
        )
        logger.info(
            "HTTP channel listening on {}:{}", self.config.host, self.config.port
        )
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def send(self, msg: OutboundMessage) -> None:
        """Resolve the waiting HTTP request with the bot's response."""
        future = self._pending.get(msg.chat_id)
        if future and not future.done():
            future.set_result(msg.content or "")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await self._process_request(reader, writer)
        except Exception as e:
            logger.warning("HTTP channel connection error: {}", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Read request line
        try:
            request_line = (
                await asyncio.wait_for(reader.readline(), timeout=10)
            ).decode(errors="replace")
        except asyncio.TimeoutError:
            await self._write_response(writer, 408, "Request timeout")
            return

        parts = request_line.strip().split()
        if len(parts) < 2:
            await self._write_response(writer, 400, "Bad request")
            return
        method, path = parts[0], parts[1]

        # Read headers
        headers: dict[str, str] = {}
        while True:
            try:
                line = (
                    await asyncio.wait_for(reader.readline(), timeout=10)
                ).decode(errors="replace")
            except asyncio.TimeoutError:
                await self._write_response(writer, 408, "Request timeout")
                return
            if line in ("\r\n", "\n", ""):
                break
            if ":" in line:
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()

        if method != "POST" or path != "/v1/chat/completions":
            await self._write_response(writer, 404, "Not found")
            return

        # API key auth
        if self.config.api_key:
            auth = headers.get("authorization", "")
            if auth != f"Bearer {self.config.api_key}":
                await self._write_response(writer, 401, "Unauthorized")
                return

        # Read body
        content_length = int(headers.get("content-length", 0))
        if not content_length:
            await self._write_response(writer, 400, "Missing content-length")
            return
        try:
            raw_body = await asyncio.wait_for(
                reader.readexactly(content_length), timeout=30
            )
        except asyncio.TimeoutError:
            await self._write_response(writer, 408, "Body read timeout")
            return

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            await self._write_response(writer, 400, "Invalid JSON")
            return

        messages = body.get("messages", [])
        if not messages:
            await self._write_response(writer, 400, "No messages")
            return

        content = messages[-1].get("content", "")
        chat_id = str(uuid.uuid4())
        wait = body.get("wait", True)  # False = fire-and-forget, return 202 immediately
        timeout = float(body.get("timeout", 300))

        if not wait:
            # Fire-and-forget: dispatch to agent and return immediately.
            # Caller doesn't need the response — agent completes async on its own.
            await self._write_response(writer, 202, "Accepted")
            await self._handle_message(
                sender_id="http",
                chat_id=chat_id,
                content=content,
            )
            return

        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[chat_id] = future

        try:
            await self._handle_message(
                sender_id="http",
                chat_id=chat_id,
                content=content,
            )
            result = await asyncio.wait_for(future, timeout=timeout)
            response_body = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": result}}]}
            )
            await self._write_response(
                writer, 200, response_body, content_type="application/json"
            )
        except asyncio.TimeoutError:
            await self._write_response(writer, 504, "Agent timeout")
        except asyncio.CancelledError:
            pass
        finally:
            self._pending.pop(chat_id, None)

    @staticmethod
    async def _write_response(
        writer: asyncio.StreamWriter,
        status: int,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        status_text = _HTTP_STATUS.get(status, "Unknown")
        body_bytes = body.encode()
        header = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode()
        writer.write(header + body_bytes)
        await writer.drain()
