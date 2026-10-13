#!/usr/bin/env python3
"""Upload the model and the dataset to the Hugging Face Hub. NOT run during the build (no token yet).

  hf auth login                 # or export HF_TOKEN=... (write token, never commit it)
  python push_to_hub.py <namespace> [--model-name saudi-tender-tagger] [--dataset-name etimad-tender-titles] [--private] [--dry-run]

Creates <namespace>/<model-name> (model card = README.md, weights, tokenizer, labels, predict.py, scripts)
and <namespace>/<dataset-name> (data/published/*, DATASET-CARD.md as the dataset README)."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("namespace"); ap.add_argument("--model-name", default="saudi-tender-tagger")
ap.add_argument("--dataset-name", default="etimad-tender-titles"); ap.add_argument("--private", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
model_repo, data_repo = f"{a.namespace}/{a.model_name}", f"{a.namespace}/{a.dataset_name}"
model_dir = os.path.join(HERE, "models", "saudi-tender-tagger")
print("model  ->", model_repo, "\ndataset ->", data_repo)
if a.dry_run: sys.exit(0)
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to the token stored by `hf auth login`
if api.whoami().get("name") is None: sys.exit("Not logged in. Run `hf auth login` or set HF_TOKEN. Nothing was uploaded.")
api.create_repo(model_repo, repo_type="model", private=a.private, exist_ok=True)
api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=model_dir)
for f in ("README.md", "LICENSE", "predict.py"):
    api.upload_file(path_or_fileobj=os.path.join(HERE, f), path_in_repo=f, repo_id=model_repo, repo_type="model")
api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=os.path.join(HERE, "scripts"), path_in_repo="scripts")
api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=os.path.join(HERE, "labels"), path_in_repo="labels")
api.create_repo(data_repo, repo_type="dataset", private=a.private, exist_ok=True)
api.upload_folder(repo_id=data_repo, repo_type="dataset", folder_path=os.path.join(HERE, "data", "published"))
api.upload_file(path_or_fileobj=os.path.join(HERE, "DATASET-CARD.md"), path_in_repo="README.md", repo_id=data_repo, repo_type="dataset")
print("done:", f"https://huggingface.co/{model_repo}", f"https://huggingface.co/datasets/{data_repo}")
