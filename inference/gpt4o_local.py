"""Azure OpenAI GPT-4o inference for CheXpert (uses Azure Chat Completions API).

Endpoint, key, and deployment are loaded from `.env` at the repo root.
Run:
    pip install openai python-dotenv datasets pillow
    python inference/gpt4o_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from openai_inference import run_inference

DEPLOYMENT = os.environ.get("GPT4O_DEPLOYMENT", "gpt-4o")
OUTPUT_PATH = os.environ.get("GPT4O_OUTPUT", "data/inference/gpt-4o.json")
API_VERSION = os.environ.get("GPT4O_API_VERSION", "2025-01-01-preview")
MAX_TOKENS = int(os.environ.get("GPT4O_MAX_TOKENS", "400"))
TEMPERATURE = float(os.environ.get("GPT4O_TEMPERATURE", "0"))


def main():
    run_inference(
        deployment=DEPLOYMENT,
        output_path=OUTPUT_PATH,
        api_type="chat",
        api_version=API_VERSION,
        extra_kwargs={
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": 42,
        },
    )


if __name__ == "__main__":
    from openai_inference import _load_dotenv_if_present
    _load_dotenv_if_present()
    main()

