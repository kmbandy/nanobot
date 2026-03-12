import asyncio
import re
from datetime import datetime, timezone
from typing import Literal

import httpx

from nanobot.agent.tools.base import Tool
from nanobot.config.schema import CfCrawlConfig

from loguru import logger


class CfCrawlTool(Tool):
    """Cloudflare Browser Rendering / Crawl API tool."""

    name = "cf_crawl"
    description = "Crawl web pages using Cloudflare Browser Rendering API"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Starting URL to crawl",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages to crawl (default: 20)",
                "default": 20,
            },
            "output_format": {
                "type": "string",
                "description": "Output format for crawled content",
                "enum": ["markdown", "html", "json"],
                "default": "markdown",
            },
            "modified_since": {
                "type": "string",
                "description": "ISO 8601 timestamp to filter pages modified after this time",
            },
            "static_mode": {
                "type": "boolean",
                "description": "Use static mode (no JavaScript execution)",
                "default": False,
            },
        },
        "required": ["url"],
    }

    def __init__(self, config: CfCrawlConfig | None = None):
        super().__init__()
        self.config = config

    async def execute(self, **kwargs) -> str:
        """Execute the cf_crawl tool."""
        url = kwargs.get("url")
        if not url:
            return "Error: 'url' parameter is required"

        max_pages = kwargs.get("max_pages", 20)
        output_format = kwargs.get("output_format", "markdown")
        modified_since = kwargs.get("modified_since")
        static_mode = kwargs.get("static_mode", False)

        # Validate configuration
        if not self.config:
            return "Error: cf_crawl tool is not configured. Please set api_token and account_id."

        if not self.config.api_token:
            return "Error: Cloudflare API token is not configured."

        if not self.config.account_id:
            return "Error: Cloudflare account ID is not configured."

        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
        }

        # Build crawl configuration
        crawl_config = {
            "start_url": url,
            "max_pages": max_pages,
            "output_format": output_format,
            "static_mode": static_mode,
        }

        if modified_since:
            # Validate ISO 8601 format
            try:
                datetime.fromisoformat(modified_since.replace("Z", "+00:00"))
            except ValueError:
                return "Error: 'modified_since' must be a valid ISO 8601 timestamp"

            crawl_config["modified_since"] = modified_since

        # Initiate crawl job
        endpoint = f"{self.config.base_url}/accounts/{self.config.account_id}/browser-rendering/crawl"

        try:
            logger.debug(f"Initiating crawl job for {url}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=crawl_config,
                    timeout=30,
                )

                if response.status_code != 200:
                    return f"Error: Failed to initiate crawl job. Status: {response.status_code}, Response: {response.text}"

                job_data = response.json()
                job_id = job_data.get("result", {}).get("id")

                if not job_id:
                    return "Error: No job ID returned from crawl initiation"

                logger.debug(f"Crawl job started: {job_id}")

        except httpx.TimeoutException:
            return "Error: Timeout while initiating crawl job"
        except httpx.RequestError as e:
            return f"Error: Network error while initiating crawl job: {str(e)}"

        # Poll for job completion
        poll_endpoint = f"{self.config.base_url}/accounts/{self.config.account_id}/browser-rendering/crawl/{job_id}"
        start_time = asyncio.get_event_loop().time()
        poll_interval = 3  # seconds
        timeout = 120  # seconds

        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    response = await client.get(poll_endpoint, headers=headers, timeout=10)

                    if response.status_code != 200:
                        return f"Error: Failed to poll job status. Status: {response.status_code}, Response: {response.text}"

                    job_data = response.json()
                    result = job_data.get("result", {})
                    status = result.get("status")

                    if status == "complete":
                        logger.debug(f"Crawl job completed: {job_id}")
                        return self._format_response(result)

                    elif status == "failed":
                        error_message = result.get("error", "Unknown error")
                        return f"Error: Crawl job failed: {error_message}"

                    elif status in ["queued", "running"]:
                        await asyncio.sleep(poll_interval)
                        logger.debug(f"Crawl job still {status}, polling again...")
                        continue

                    else:
                        return f"Error: Unexpected job status: {status}"

                except httpx.TimeoutException:
                    logger.debug("Poll request timed out, retrying...")
                    await asyncio.sleep(poll_interval)
                except httpx.RequestError as e:
                    logger.debug(f"Poll request failed: {str(e)}")
                    await asyncio.sleep(poll_interval)

        return "Error: Crawl job timed out (120s limit exceeded)"

    def _format_response(self, result: dict) -> str:
        """Format crawl result as structured text."""
        crawled_pages = result.get("crawled_pages", [])

        if not crawled_pages:
            return "No pages were crawled."

        output_parts = []

        for page in crawled_pages:
            url = page.get("url", "Unknown URL")
            content = page.get("content", "")
            output_parts.append(f"## {url}\n{content}\n")

        response = "\n".join(output_parts)

        # Truncate to 50000 chars if needed
        if len(response) > 50000:
            response = response[:50000]
            response += "\n\n[Content truncated due to size limit]"

        return response
