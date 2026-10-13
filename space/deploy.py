#!/usr/bin/env python3
"""Create the Space chaliasif/saudi-tender-tagger and upload space/.   python space/deploy.py [--private]"""
import argparse, os
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument("--private", action="store_true"); a = ap.parse_args()
REPO = "chaliasif/saudi-tender-tagger"
api = HfApi()
api.create_repo(REPO, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
api.upload_folder(repo_id=REPO, repo_type="space", folder_path=os.path.dirname(os.path.abspath(__file__)),
                  ignore_patterns=["deploy.py", "__pycache__/*"])
print("https://huggingface.co/spaces/" + REPO)
