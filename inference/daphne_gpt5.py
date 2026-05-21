"""Azure OpenAI GPT-5.4 inference for CheXpert (uses Azure Responses API).

Endpoint, key, and deployment are loaded from `.env` at the repo root.
Run:
    pip install openai python-dotenv datasets pillow
    python inference/daphne_gpt5.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from openai_inference import run_inference

DEPLOYMENT = os.environ.get("GPT5_DEPLOYMENT", "gpt-5.4")
OUTPUT_PATH = os.environ.get("GPT5_OUTPUT", "outputs/daphne_gpt5_outputs.json")
API_VERSION = os.environ.get("GPT5_API_VERSION", "2025-04-01-preview")
MAX_OUTPUT_TOKENS = int(os.environ.get("GPT5_MAX_OUTPUT_TOKENS", "2000"))


def main():
    run_inference(
        deployment=DEPLOYMENT,
        output_path=OUTPUT_PATH,
        api_type="responses",
        api_version=API_VERSION,
        extra_kwargs={"max_output_tokens": MAX_OUTPUT_TOKENS},
    )


if __name__ == "__main__":
    # Ensure .env is loaded even when this is the entry point
    from openai_inference import _load_dotenv_if_present
    _load_dotenv_if_present()
    main()

