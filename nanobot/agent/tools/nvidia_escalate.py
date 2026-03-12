import httpx
import os
import re
from loguru import logger
from nanobot.agent.tools.base import Tool


class NvidiaEscalateTool(Tool):
    """Escalate complex tasks to NVIDIA's hosted LLM API for heavyweight reasoning.
    
    Use ONLY for tasks exceeding local model capability: architecture critique,
    independent validation, or second opinions on important decisions.
    Do not use for routine queries or simple tasks.
    """

    @property
    def name(self) -> str:
        return "nvidia_escalate"

    @property
    def description(self) -> str:
        return self.__doc__

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The complex task to escalate (required)"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (optional, defaults to nvidia/llama-3.1-nemotron-ultra-253b-v1)"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt for context"
                }
            },
            "required": ["task"]
        }

    def __init__(self, api_key: str = '', default_model: str = 'nvidia/llama-3.1-nemotron-ultra-253b-v1'):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", '')
        self.default_model = default_model

    async def execute(self, task: str, model: str = None, system_prompt: str = None) -> str:
        """Execute escalation to NVIDIA API."""
        api_key = self.api_key
        
        if not api_key:
            return (
                "ERROR: NVIDIA API key not configured. "
                "Add 'nvidiaApiKey' to config.json or set NVIDIA_API_KEY environment variable."
            )

        selected_model = model or self.default_model

        # Qwen-style models embed <tool_call> XML in task strings.
        # Nemotron's vLLM endpoint chokes on these tags during tokenization.
        # 1. Strip complete closed blocks
        task = re.sub(r"<tool_call>.*?</tool_call>", "", task, flags=re.DOTALL)
        # 2. Strip any unclosed <tool_call> block (runs to end of string)
        task = re.sub(r"<tool_call>.*$", "", task, flags=re.DOTALL).strip()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})

        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }

        logger.debug("nvidia_escalate: task after sanitization (first 200 chars): {}", task[:200])
        logger.info("nvidia_escalate: calling {} (task length: {} chars)", selected_model, len(task))
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                logger.info("nvidia_escalate: HTTP {} from NVIDIA API", response.status_code)
                response.raise_for_status()
                result = response.json()
                msg = result["choices"][0]["message"]
                content = msg.get("content")
                # Nemotron Ultra (reasoning model) may return content=None with
                # the answer in reasoning_content or reasoning fields.
                if not content:
                    content = (msg.get("reasoning_content")
                               or msg.get("reasoning")
                               or "")
                if not content:
                    logger.warning("nvidia_escalate: empty content in response, keys: {}", list(msg.keys()))
                    return "nvidia_escalate: received empty response from model."
                logger.info("nvidia_escalate: received {} chars from {}", len(content), selected_model)
                return content
        except httpx.HTTPStatusError as e:
            # Don't return response.text — NVIDIA's error body echoes back the input,
            # which may contain <tool_call> XML that will poison the next LLM call.
            logger.error("nvidia_escalate: HTTP error {} - {}", response.status_code, response.text[:500])
            return f"nvidia_escalate failed: HTTP {response.status_code} from NVIDIA API."
        except Exception as e:
            logger.error("nvidia_escalate: unexpected error: {}", e)
            return f"nvidia_escalate failed: {type(e).__name__}"