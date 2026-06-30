"""
Push the locally trained PitchCoach BERT model to HuggingFace Hub.

Run this once after train_bert.py so your Vercel deployment can call it
via the HuggingFace Inference API.

Usage:
    hf auth login                 # paste your HF write token
    python push_to_hub.py
    python push_to_hub.py --repo-id YOUR_HF_USERNAME/pitch-coach-bert

Then set these env vars in your Vercel project dashboard:
    HF_API_KEY   = your HuggingFace read token
    HF_MODEL_ID  = YOUR_HF_USERNAME/pitch-coach-bert
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer, DistilBertForSequenceClassification

DEFAULT_REPO_NAME = "pitch-coach-bert"


def default_repo_id() -> str:
    """Use the logged-in HF account as the repo namespace."""
    try:
        username = HfApi().whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            "Not logged in to Hugging Face. Run:\n"
            "  hf auth login\n"
            "Then retry, or pass --repo-id YOUR_USERNAME/pitch-coach-bert"
        ) from exc
    return f"{username}/{DEFAULT_REPO_NAME}"


def check_repo_access(repo_id: str) -> None:
    namespace = repo_id.split("/", 1)[0]
    try:
        username = HfApi().whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            "Not logged in to Hugging Face. Run: hf auth login"
        ) from exc
    if namespace != username:
        raise SystemExit(
            f"You are logged in as '{username}' but --repo-id targets '{namespace}'.\n"
            f"Use your own namespace, e.g.:\n"
            f"  python push_to_hub.py --repo-id {username}/{DEFAULT_REPO_NAME}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"HuggingFace repo id (default: <your-username>/{DEFAULT_REPO_NAME})",
    )
    parser.add_argument(
        "--model-path",
        default="pitch_coach_model",
        help="Local directory with saved model (default: pitch_coach_model)",
    )
    args = parser.parse_args()

    repo_id = args.repo_id or default_repo_id()
    check_repo_access(repo_id)

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        raise SystemExit(f"Model not found at '{model_dir}'. Run train_bert.py first.")

    print(f"Loading model from '{model_dir}' …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))

    # Ensure the HF Inference API returns human-readable labels ("good"/"bad")
    # rather than generic "LABEL_0"/"LABEL_1".
    model.config.id2label = {0: "bad", 1: "good"}
    model.config.label2id = {"bad": 0, "good": 1}

    print(f"Pushing to https://huggingface.co/{repo_id} …")
    try:
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
    except Exception as exc:
        print(f"\nPush failed: {exc}", file=sys.stderr)
        raise SystemExit(
            "\nIf you see 403 Forbidden, check:\n"
            "  1. hf auth login (token needs Write access)\n"
            "  2. --repo-id uses YOUR username, not someone else's\n"
            f"  3. Try: python push_to_hub.py --repo-id {repo_id}"
        ) from exc

    print(
        f"\nDone! Set these in your Vercel dashboard:\n"
        f"  HF_MODEL_ID = {repo_id}\n"
        f"  HF_API_KEY  = <your HuggingFace read token>\n"
        f"\nInference API endpoint:\n"
        f"  https://api-inference.huggingface.co/models/{repo_id}"
    )


if __name__ == "__main__":
    main()
