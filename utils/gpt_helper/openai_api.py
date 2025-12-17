"""Helpers for working with OpenAI chat and batch APIs."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.2"
RequestLike = Union[Mapping[str, str], Tuple[str, str]]


def _normalize_request(request: RequestLike) -> Tuple[str, str]:
    """Extract (system, user) from a mapping or 2-tuple."""
    if isinstance(request, Mapping) and "system" in request and "user" in request:
        return str(request["system"]), str(request["user"])
    if isinstance(request, tuple) and len(request) == 2:
        return str(request[0]), str(request[1])
    raise ValueError("Each request must provide system and user messages.")


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
    **kwargs: object,
) -> str:
    """
    Execute a single chat completion and return the assistant content.
    Additional OpenAI parameters (e.g., temperature) can be passed via kwargs.
    """
    client = client or OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    message = response.choices[0].message
    return message.content or ""


def run_sequential_requests(
    requests: Sequence[RequestLike],
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
    **kwargs: object,
) -> List[str]:
    """Process a list of (system, user) prompts one at a time."""
    outputs: List[str] = []
    for request in requests:
        system_prompt, user_prompt = _normalize_request(request)
        outputs.append(
            chat_completion(
                system_prompt,
                user_prompt,
                model=model,
                client=client,
                **kwargs,
            )
        )
    return outputs


def run_batch_requests(
    requests: Sequence[RequestLike],
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
    poll_interval: float = 5.0,
    completion_window: str = "24h",
) -> List[Optional[str]]:
    """
    Submit prompts using the OpenAI batch API and wait for completion.

    Returns a list of responses aligned to the input order.
    """
    if not requests:
        return []

    client = client or OpenAI()
    tmp_path: Optional[Path] = None
    input_file_id: Optional[str] = None
    output_file_id: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            for idx, request in enumerate(requests):
                system_prompt, user_prompt = _normalize_request(request)
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                batch_line = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                tmp_file.write(json.dumps(batch_line))
                tmp_file.write("\n")

        with tmp_path.open("rb") as input_file:
            uploaded = client.files.create(file=input_file, purpose="batch")
        input_file_id = uploaded.id
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )

        terminal_states = {"completed", "failed", "canceled", "expired"}
        while batch.status not in terminal_states:
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch.id)

        output_file_id = batch.output_file_id
        if batch.status != "completed" or not output_file_id:
            raise RuntimeError(
                f"Batch {batch.id} did not complete successfully (status={batch.status})."
            )

        output_content = client.files.content(output_file_id)
        if hasattr(output_content, "text"):
            raw_output = output_content.text
        elif hasattr(output_content, "content"):
            raw_output = output_content.content.decode("utf-8")
        else:
            raw_output = str(output_content)

        results: List[Optional[str]] = [None] * len(requests)
        for line in raw_output.splitlines():
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if not custom_id:
                continue

            response_body = ((record.get("response") or {}).get("body") or {})
            choices = response_body.get("choices") or []
            content = None
            if choices:
                content = choices[0].get("message", {}).get("content")

            idx = int(custom_id.split("-", 1)[-1]) if "-" in custom_id else None
            if idx is not None and 0 <= idx < len(results):
                results[idx] = content

        return results
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
        if input_file_id:
            try:
                client.files.delete(input_file_id)
            except Exception:
                pass
        if output_file_id:
            try:
                client.files.delete(output_file_id)
            except Exception:
                pass
