"""Publish the native SAI inference code and checkpoint to Hugging Face.

Hugging Face's SDK handles authentication and file transport only. Inference
continues to use the project's own EmbeddingModel and EmbeddingTokenizer.
"""

import argparse
from pathlib import Path
import shutil
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = (
    "README.md",
    "requirements-inference.txt",
    "config/base.json",
    "config/100M.json",
    "config/embedding.json",
    "model/SAI-Embedding_100M.pt",
    "src/__init__.py",
    "src/model/__init__.py",
    "src/model/EmbeddingModel.py",
    "src/model/TransformerModel.py",
    "src/model/DecoderBlock.py",
    "src/model/GroupedQueryAttention.py",
    "src/model/RotaryPositionalEmbedding.py",
    "src/model/SwiGLU.py",
    "src/data/__init__.py",
    "src/data/embedding_data.py",
    "src/tokenizer/__init__.py",
    "src/tokenizer/tokenizer.model",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/eval/__init__.py",
    "src/eval/evaluate_embedding.py",
    "results/benchmark/sai_eval.json",
)


def prepare_release(destination):
    """Copy an explicit release manifest; exclude private docs and training data."""
    missing = [name for name in RELEASE_FILES if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")
    for name in RELEASE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, target)
    # Evaluation needs SciPy; native inference does not need Transformers.
    (destination / "requirements.txt").write_text(
        "-r requirements-inference.txt\nscipy\nhuggingface_hub\n",
        encoding="utf-8",
    )


def publish_release(repo_id, token=None):
    from huggingface_hub import HfApi, get_token

    token = token or get_token()
    if not token:
        raise RuntimeError("Run hf auth login with a Write token before publishing.")
    api = HfApi(token=token)
    account = api.whoami()
    print(f"Authenticated as {account['name']}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sai-hf-release-") as folder:
        prepare_release(Path(folder))
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            commit_message="Publish SAI-Embedding_100M checkpoint and native inference code",
        )
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        missing = set(RELEASE_FILES) - remote_files
        if missing:
            raise RuntimeError(f"Release incomplete on Hub: {sorted(missing)}")
        print(f"Published: https://huggingface.co/{repo_id}")
        print(f"Commit: {commit.commit_url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="thongbuind/SAI-Embedding_100M")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="sai-hf-release-") as folder:
            prepare_release(Path(folder))
            for name in RELEASE_FILES:
                print(name)
            print("requirements.txt")
            print(f"Ready: {len(RELEASE_FILES) + 1} files for {args.repo_id}")
    else:
        publish_release(args.repo_id)


if __name__ == "__main__":
    main()
