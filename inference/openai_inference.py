"""Shared Azure OpenAI inference helper for the GPT-5.4 and GPT-4o local
entry points.

Uses the official `openai` Python SDK's `AzureOpenAI` client class. This class
ONLY sends requests to the `azure_endpoint` you configure; it never contacts
api.openai.com. Microsoft's Azure OpenAI docs recommend this SDK.

Credentials are loaded from a `.env` file in the repo root (gitignored), or
from process env vars. Required:
    AZURE_OPENAI_ENDPOINT   e.g. https://my-resource.cognitiveservices.azure.com
    AZURE_OPENAI_API_KEY    the Azure OpenAI key
"""

import base64
import io
import json
import os
import sys
import time
from typing import Any

_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from data.chexpert_dataset import PATHOLOGIES, iter_batched, load_sample_ids


def _load_dotenv_if_present() -> None:
    """Load .env from the repo root, if it exists. No-op otherwise."""
    env_path = os.path.join(_repo_root, ".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Fallback minimal parser (KEY=VALUE, ignoring comments/blank lines).
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv_if_present()

DEFAULT_API_VERSION = "2025-04-01-preview"

PROMPT = (
    "You are an expert radiologist reviewing a single chest radiograph.\n"
    "For each of the 14 thoracic conditions listed below, decide whether the\n"
    "condition is PRESENT (1) or ABSENT (0) in this image.\n\n"
    "Conditions (use these exact keys, in this order):\n"
    + "\n".join(f"  - {p}" for p in PATHOLOGIES)
    + "\n\n"
    "Respond with ONLY a single valid JSON object. The keys must be the\n"
    "exact condition names above and the values must be the integers 0 or 1.\n"
    "Do not include any prose, markdown fences, units, or explanations.\n"
    'Example: {"Atelectasis": 0, "Cardiomegaly": 1, ...}'
)


def make_azure_client(api_version: str | None = None):
    """Build an Azure-only OpenAI client. Never contacts api.openai.com."""
    try:
        from openai import AzureOpenAI
    except ImportError as e:
        raise ImportError(
            "The `openai` package is required (it provides the AzureOpenAI "
            "client; this code never targets api.openai.com). "
            "Install with: pip install openai"
        ) from e

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set "
            "(in .env at repo root, or as process env vars)."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
        or os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )


def image_to_data_url(image) -> str:
    """Convert a PIL.Image to a base64 JPEG data URL."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_response(raw: str) -> dict[str, int]:
    """Parse a model JSON response into {pathology: 0|1}.

    Tolerates markdown code fences, surrounding prose, and bool/str values.
    Returns an empty dict if no valid object can be extracted.
    """
    if not raw:
        return {}

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = text[start : end + 1]

    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return {}

    out: dict[str, int] = {}
    for p in PATHOLOGIES:
        if p not in obj:
            continue
        v = obj[p]
        if isinstance(v, bool):
            out[p] = int(v)
        elif isinstance(v, (int, float)):
            out[p] = 1 if v >= 0.5 else 0
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "present", "positive"}:
                out[p] = 1
            elif s in {"0", "false", "no", "absent", "negative"}:
                out[p] = 0
    return out


def _binarize_label(raw_label, uncertain_policy: str = "zero") -> int:
    """CheXpert label -> 0/1. Uncertainty (-1) and missing default to U-Zeros."""
    if raw_label is None:
        return 0
    try:
        v = float(raw_label)
    except (TypeError, ValueError):
        return 0
    if v == 1.0:
        return 1
    if v == -1.0:
        return {"zero": 0, "one": 1}.get(uncertain_policy, 0)
    return 0


def score(predictions: dict[str, int], labels: dict[str, Any]) -> dict[str, Any]:
    """Return per-label correctness and a single 'correct' flag (all-labels-match)."""
    per_label: dict[str, int] = {}
    bin_labels: dict[str, int] = {}
    for p in PATHOLOGIES:
        gt = _binarize_label(labels.get(p))
        bin_labels[p] = gt
        pred = predictions.get(p)
        per_label[p] = int(pred is not None and int(pred) == gt)
    all_correct = int(all(per_label[p] for p in PATHOLOGIES))
    return {
        "correct": all_correct,
        "per_label_correct": per_label,
        "binarized_labels": bin_labels,
    }


def _load_existing(output_path: str) -> dict:
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _flagged_only(cfr) -> dict | None:
    """Extract only flagged/detected categories from a content_filter dict."""
    if not cfr or not isinstance(cfr, dict):
        return None
    flagged = {
        k: v for k, v in cfr.items()
        if isinstance(v, dict) and (v.get("filtered") or v.get("detected"))
    }
    return flagged or None


def call_chat_completions(
    client,
    deployment: str,
    image,
    extra_kwargs: dict | None = None,
) -> dict:
    """Azure Chat Completions API (used by GPT-4o).

    Returns {'text': str, 'diagnostics': dict}. Diagnostics capture
    finish_reason, model refusal, content-filter flags, and token usage so we
    can distinguish refusals/filtering from genuine model predictions.
    """
    data_url = image_to_data_url(image)
    kwargs = {
        "model": deployment,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    text = choice.message.content or ""
    full = resp.model_dump()
    diagnostics = {
        "finish_reason": choice.finish_reason,
        "refusal": getattr(choice.message, "refusal", None),
        "completion_content_filter": _flagged_only(
            full["choices"][0].get("content_filter_results")
        ),
        "prompt_content_filter": [
            _flagged_only(p.get("content_filter_results"))
            for p in (full.get("prompt_filter_results") or [])
        ],
        "usage": full.get("usage"),
        "model": full.get("model"),
    }
    return {"text": text, "diagnostics": diagnostics}


def call_responses(
    client,
    deployment: str,
    image,
    extra_kwargs: dict | None = None,
) -> dict:
    """Azure Responses API (used by GPT-5.4).

    Returns {'text': str, 'diagnostics': dict}. Diagnostics capture status,
    incomplete reason, refusal, reasoning summary, and reasoning-token usage.
    """
    data_url = image_to_data_url(image)
    kwargs = {
        "model": deployment,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    resp = client.responses.create(**kwargs)

    text = getattr(resp, "output_text", None) or ""
    full = resp.model_dump()

    refusal: str | None = None
    reasoning_summary_parts: list[str] = []
    if not text:
        parts: list[str] = []
        for item in full.get("output") or []:
            if item.get("type") == "reasoning":
                for s in item.get("summary") or []:
                    t = s.get("text")
                    if t:
                        reasoning_summary_parts.append(t)
            for content in item.get("content") or []:
                ctype = content.get("type")
                if ctype in ("output_text", "text") and content.get("text"):
                    parts.append(content["text"])
                elif ctype == "refusal" and content.get("refusal"):
                    refusal = content["refusal"]
        if parts:
            text = "".join(parts)
    else:
        for item in full.get("output") or []:
            if item.get("type") == "reasoning":
                for s in item.get("summary") or []:
                    t = s.get("text")
                    if t:
                        reasoning_summary_parts.append(t)

    usage = full.get("usage") or {}
    reasoning_tokens = (
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    )

    diagnostics = {
        "status": full.get("status"),
        "incomplete_details": full.get("incomplete_details"),
        "refusal": refusal,
        "reasoning_summary": "\n".join(reasoning_summary_parts) or None,
        "reasoning_tokens": reasoning_tokens,
        "usage": usage or None,
        "model": full.get("model"),
    }
    return {"text": text, "diagnostics": diagnostics}


_CALLERS = {
    "chat": call_chat_completions,
    "responses": call_responses,
}


def run_inference(
    deployment: str,
    output_path: str,
    api_type: str = "chat",
    api_version: str | None = None,
    extra_kwargs: dict | None = None,
    batch_size: int = 4,
    split: str = "validation",
    sleep_seconds: float = 0.0,
    limit: int | None = None,
) -> None:
    """Run inference for one Azure deployment, with resumable JSON output.

    api_type: "chat" (Chat Completions) or "responses" (Responses API).
    api_version: overrides AZURE_OPENAI_API_VERSION env var; use the version
        appropriate for the chosen api_type.
    limit: if set, only process this many new items (useful for smoke tests).
    """
    if api_type not in _CALLERS:
        raise ValueError(f"api_type must be one of {list(_CALLERS)}; got {api_type!r}")
    caller = _CALLERS[api_type]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    client = make_azure_client(api_version=api_version)

    sample_ids = load_sample_ids()
    results = _load_existing(output_path)
    done_ids = set(results.keys())
    total = len(sample_ids)
    print(
        f"[{deployment}] api={api_type} version={api_version} "
        f"{len(done_ids)}/{total} already done; resuming.",
        flush=True,
    )

    processed = 0
    for batch in iter_batched(batch_size=batch_size, sample_ids=sample_ids, split=split):
        for item in batch:
            item_id = item["item_id"]
            if item_id in done_ids:
                continue
            if limit is not None and processed >= limit:
                return

            try:
                out = caller(client, deployment, item["image"], extra_kwargs)
                text = out["text"]
                diagnostics = out["diagnostics"]
                preds = parse_response(text)

                # Distinguish refusals / empty / incomplete from genuine predictions.
                # Missing items get correct=None so the response matrix can treat
                # them as MISSING (not WRONG) under IRT.
                missing_reason: str | None = None
                if diagnostics.get("refusal"):
                    missing_reason = "model_refusal"
                elif diagnostics.get("incomplete_details"):
                    missing_reason = "incomplete"
                elif diagnostics.get("finish_reason") == "content_filter":
                    missing_reason = "azure_content_filter"
                elif not text:
                    missing_reason = "empty_response"
                elif not preds:
                    missing_reason = "unparseable"

                if missing_reason:
                    results[item_id] = {
                        "deployment": deployment,
                        "api_type": api_type,
                        "predictions": None,
                        "labels": item["labels"],
                        "raw": text,
                        "diagnostics": diagnostics,
                        "missing_reason": missing_reason,
                        "correct": None,
                        "per_label_correct": None,
                        "binarized_labels": None,
                    }
                else:
                    scored = score(preds, item["labels"])
                    results[item_id] = {
                        "deployment": deployment,
                        "api_type": api_type,
                        "predictions": preds,
                        "labels": item["labels"],
                        "raw": text,
                        "diagnostics": diagnostics,
                        "missing_reason": None,
                        **scored,
                    }
            except Exception as e:  # noqa: BLE001
                results[item_id] = {
                    "deployment": deployment,
                    "api_type": api_type,
                    "error": f"{type(e).__name__}: {e}",
                    "missing_reason": "exception",
                    "correct": None,
                }

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            processed += 1
            if processed % 10 == 0:
                print(
                    f"[{deployment}] {len(results)}/{total} items written",
                    flush=True,
                )
            if sleep_seconds:
                time.sleep(sleep_seconds)

    print(f"[{deployment}] Done. {len(results)} items in {output_path}", flush=True)

