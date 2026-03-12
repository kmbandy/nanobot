import httpx
import os
import re
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
                    "description": "Model to use (optional, defaults to 253B)",
                    "enum": [
                        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
                        "meta/llama-3.1-405b-instruct",
                        "mistralai/mixtral-8x22b-instruct-v0.1",
                        "nvidia/llama-3.3-nemotron-super-49b-v1"
                    ]
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

        # Strip XML tool call blocks that Qwen-style models may embed in task strings.
        # Nemotron's vLLM endpoint processes <tool_call> tags during tokenization and will 500.
        task = re.sub(r"<tool_call>.*?</tool_call>", "", task, flags=re.DOTALL).strip()

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

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"API error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"