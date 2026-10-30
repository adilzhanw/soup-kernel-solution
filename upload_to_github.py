#!/usr/bin/env python3
"""Upload the solution to GitHub via the REST API (no git CLI needed).

Usage: python upload_to_github.py <TOKEN>
Files uploaded: solution/, harness/, task/, tests/, README.md, SUBMISSION.md,
kaggle_notebook.ipynb, .gitignore, LICENSE, NOTICE, data/ (train+eval jsonl).
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_TOKEN", "")
OWNER = "adilzhanw"
REPO = "soup-kernel-solution"
BRANCH = "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"

if not TOKEN:
    sys.exit("No token. Pass it as arg 1 or set GITHUB_TOKEN.")

def api(method, path, body=None):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode()) if r.headers.get("Content-Type", "").startswith("application/json") else None
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
        raise

def file_path_in_repo(repo_path):
    """Return the repo's default branch SHA (or None if empty)."""
    try:
        ref = api("GET", f"/git/ref/heads/{BRANCH}")
        return ref["object"]["sha"]
    except Exception:
        return None

def upload_file(repo_path, local_path, branch, sha):
    """Create or update a file via the contents API."""
    content_b64 = base64.b64encode(local_path.read_bytes()).decode()
    body = {"message": f"add {repo_path}", "content": content_b64, "branch": branch}
    if sha:
        body["sha"] = sha
    # get existing sha if file exists
    try:
        existing = api("GET", f"/contents/{repo_path}?ref={branch}")
        if isinstance(existing, dict) and "sha" in existing:
            body["sha"] = existing["sha"]
    except Exception:
        pass
    api("PUT", f"/contents/{repo_path}", body)
    print(f"  uploaded {repo_path}")

def main():
    root = Path(__file__).resolve().parent
    # collect files to upload (skip __pycache__, .pytest_cache, runs, data is optional)
    skip_dirs = {"__pycache__", ".pytest_cache", "runs", "runs_local", "runs_base", ".soup-crashes", "dashboard.html"}
    files = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if any(part in skip_dirs for part in parts):
            continue
        if p.suffix in {".pyc", ".log"}:
            continue
        files.append(rel)
    files.sort()
    print(f"Uploading {len(files)} files to {OWNER}/{REPO}...")

    # get default branch sha (create main if empty)
    sha = file_path_in_repo(None)
    if sha is None:
        # repo is empty: create an initial commit via the API (create README first)
        readme = root / "README.md"
        content_b64 = base64.b64encode(readme.read_bytes()).decode()
        body = {"message": "initial commit (README)", "content": content_b64, "branch": BRANCH}
        api("PUT", "/contents/README.md", body)
        sha = file_path_in_repo(None)
        print(f"  created initial commit, sha={sha[:8]}")
        # remove README from the list since we just uploaded it
        files = [f for f in files if f != Path("README.md")]

    for rel in files:
        local = root / rel
        repo_path = str(rel)
        upload_file(repo_path, local, BRANCH, sha)

    print(f"\nDone. View: https://github.com/{OWNER}/{REPO}")
    # get the latest commit sha
    commits = api("GET", f"/commits?per_page=1")
    if commits:
        print(f"Latest commit: {commits[0]['sha']}")
        print(f"Commit URL: https://github.com/{OWNER}/{REPO}/commit/{commits[0]['sha']}")

if __name__ == "__main__":
    main()
