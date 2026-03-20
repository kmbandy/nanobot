"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    CHUNK_SIZE = 1900

    def _split_content(self, content: str) -> list[str]:
        """Split content into chunks no larger than CHUNK_SIZE, breaking on newlines where possible."""
        if len(content) <= self.CHUNK_SIZE:
            return [content]
        chunks = []
        while content:
            if len(content) <= self.CHUNK_SIZE:
                chunks.append(content)
                break
            split_at = content.rfind("\n", 0, self.CHUNK_SIZE)
            if split_at == -1:
                split_at = self.CHUNK_SIZE
            chunks.append(content[:split_at].rstrip())
            content = content[split_at:].lstrip()
        return chunks

    async def execute(self, **kwargs: Any) -> str:
        content: str = kwargs["content"]
        channel: str | None = kwargs.get("channel") or self._default_channel
        chat_id: str | None = kwargs.get("chat_id") or self._default_chat_id
        message_id: str | None = kwargs.get("message_id") or self._default_message_id
        media: list[str] | None = kwargs.get("media")

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        chunks = self._split_content(content)
        errors = []
        for i, chunk in enumerate(chunks):
            chunk_text = f"{chunk}\n*(part {i+1}/{len(chunks)})*" if len(chunks) > 1 else chunk
            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=chunk_text,
                media=(media or []) if i == 0 else [],
                metadata={"message_id": message_id},
            )
            try:
                await self._send_callback(msg)
            except Exception as e:
                errors.append(f"chunk {i+1}: {e}")

        if errors:
            return f"Error sending message: {'; '.join(errors)}"

        if channel == self._default_channel and chat_id == self._default_chat_id:
            self._sent_in_turn = True
        media_info = f" with {len(media)} attachments" if media else ""
        parts_info = f" ({len(chunks)} parts)" if len(chunks) > 1 else ""
        return f"Message sent to {channel}:{chat_id}{media_info}{parts_info}"
