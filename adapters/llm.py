"""LLM client adapter for KiraAI.

Lightning calls `client.chat([{role, content}])` directly with a message list.
KiraAI requires `client.chat(LLMRequest)`. This wrapper bridges the two so the
extractor and persona engine can stay close to upstream.
"""

from __future__ import annotations

from typing import Optional

from core.provider import LLMRequest, LLMModelClient
from core.prompt_manager import Prompt
from core.logging_manager import get_logger

logger = get_logger("hippocampus.llm", "cyan")


async def chat_text(
    client: LLMModelClient,
    prompt: str,
    *,
    system: Optional[str] = None,
) -> str:
    """Call an LLMModelClient with a single user prompt, return the text response.

    Returns empty string on failure.
    """
    if client is None:
        logger.warning("chat_text called with client=None")
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    req = LLMRequest(messages=messages)
    try:
        resp = await client.chat(req)
    except Exception as e:
        logger.error(f"chat_text failed: {e}")
        return ""

    text = getattr(resp, "text_response", "") or ""
    return text.strip()


def append_to_prompt_section(system_prompts, name: str, addition: str) -> bool:
    """Append `addition` to the system prompt section identified by `name`.

    Returns True if the section existed.
    """
    for p in system_prompts:
        if isinstance(p, Prompt) and p.name == name:
            p.content = (p.content or "") + addition
            return True
    return False
