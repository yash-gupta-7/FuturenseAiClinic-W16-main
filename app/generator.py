"""Generation step: assemble a grounded prompt and call the LLM.

Provider-flexible: OpenAI (spec default) or Anthropic. The prompt forces the
model to answer ONLY from the retrieved context, cite sources by filename, and
abstain with a fixed sentinel when the context does not contain the answer.
"""
import logging
from typing import List, Dict, Any, Tuple

from app import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the AI Admission Copilot. You answer questions about university "
    "admission policies using ONLY the provided context excerpts.\n"
    "Rules:\n"
    "1. Use only facts found in the context. Never use outside knowledge.\n"
    "2. Cite the source filename(s) you used in square brackets, e.g. "
    "[transfer_credit_policy.md].\n"
    f"3. If the context does not contain the answer, reply EXACTLY: "
    f"\"{config.ABSTAIN_MESSAGE}\" and nothing else.\n"
    "4. Be concise and precise with numbers, deadlines, and conditions."
)


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered, source-labelled block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['content']}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite the source filename(s)."
    )


def _generate_openai(system: str, user: str) -> Tuple[str, Dict[str, int]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return resp.choices[0].message.content.strip(), usage


def _generate_anthropic(system: str, user: str) -> Tuple[str, Dict[str, int]]:
    from anthropic import Anthropic

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_CHAT_MODEL,
        max_tokens=1024,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return resp.content[0].text.strip(), usage


def generate(question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    """Generate a grounded answer from the question and retrieved chunks.

    Returns (answer_text, token_usage).
    """
    if not chunks:
        return config.ABSTAIN_MESSAGE, {"input_tokens": 0, "output_tokens": 0}

    context = format_context(chunks)
    user = build_user_prompt(question, context)

    if config.GENERATION_PROVIDER == "anthropic":
        return _generate_anthropic(SYSTEM_PROMPT, user)
    return _generate_openai(SYSTEM_PROMPT, user)
