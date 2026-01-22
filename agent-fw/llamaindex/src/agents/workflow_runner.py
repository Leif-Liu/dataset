from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.agent import AgentWorkflow


def run_workflow_sync(
    workflow: AgentWorkflow, prompt: str, max_iterations: int | None = None
) -> str:
    async def _run() -> Any:
        handler = workflow.run(user_msg=prompt, max_iterations=max_iterations)
        return await handler

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError("Cannot run AgentWorkflow in a running event loop.")

    result = asyncio.run(_run())
    if hasattr(result, "response") and hasattr(result.response, "content"):
        return result.response.content or ""
    return str(result)

