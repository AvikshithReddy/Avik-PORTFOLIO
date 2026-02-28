#!/usr/bin/env python3
"""
Refresh chatbot data files from source content and optionally trigger ingest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        if tag in ("p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "div"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth:
            self._skip_depth -= 1
        if tag in ("p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "div"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def get_text(self) -> str:
        text = "".join(self._chunks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def extract_title(html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title


def build_portfolio_json(index_html: Path, out_json: Path) -> None:
    html = index_html.read_text(encoding="utf-8", errors="ignore")
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    content = extractor.get_text()
    payload = {
        "source": index_html.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": extract_title(html),
        "content": content,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def copy_file(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    if src.resolve() == dest.resolve():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def load_env(path: Path) -> dict:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def trigger_ingest(url: str, sources: list[str], admin_key: str | None) -> None:
    payload = json.dumps({"sources": sources}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if admin_key:
        req.add_header("x-admin-key", admin_key)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        print(f"Ingest response: {body}")
    except Exception as exc:
        print(f"Ingest failed: {exc}")
        print("Make sure the backend is running on http://localhost:8000 before using --ingest.")


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def refresh_once(args) -> None:
    repo_root = resolve_repo_root()
    data_dir = Path(args.data_dir).resolve()

    index_html = Path(args.index_html).resolve()
    if not index_html.exists():
        print(f"index.html not found: {index_html}", file=sys.stderr)
        return

    portfolio_json = data_dir / "portfolio_data.json"
    build_portfolio_json(index_html, portfolio_json)
    print(f"Updated {portfolio_json}")

    resume_src = Path(args.resume_src).resolve()
    resume_dest = data_dir / "avikshithReddy_resume.pdf"
    if copy_file(resume_src, resume_dest):
        print(f"Synced resume to {resume_dest}")
    else:
        print(f"Resume source not found: {resume_src}")

    linkedin_src = Path(args.linkedin_src).resolve()
    linkedin_dest = data_dir / "Profile.pdf"
    if copy_file(linkedin_src, linkedin_dest):
        print(f"Synced LinkedIn to {linkedin_dest}")
    else:
        print(f"LinkedIn source not found: {linkedin_src}")

    if args.ingest:
        env = load_env(Path(args.env_path))
        admin_key = env.get("ADMIN_INGEST_KEY", "")
        openai_key = env.get("OPENAI_API_KEY", "")
        if not openai_key:
            print("Skipping ingest: OPENAI_API_KEY is empty in chatbot-backend/.env.")
            return
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        trigger_ingest(args.ingest_url, sources, admin_key)


def watch(args) -> None:
    watch_paths = [
        Path(args.index_html).resolve(),
        Path(args.resume_src).resolve(),
        Path(args.linkedin_src).resolve(),
    ]
    last_mtimes: dict[Path, float] = {}
    for p in watch_paths:
        last_mtimes[p] = p.stat().st_mtime if p.exists() else 0.0

    print("Watching for changes...")
    while True:
        changed = False
        for p in watch_paths:
            mtime = p.stat().st_mtime if p.exists() else 0.0
            if mtime != last_mtimes.get(p, 0.0):
                last_mtimes[p] = mtime
                changed = True
        if changed:
            refresh_once(args)
        time.sleep(args.interval)


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = resolve_repo_root()
    default_data_dir = repo_root / "chatbot-backend" / "data"
    default_index_html = repo_root / "index.html"
    default_resume_src = repo_root / "avikshithReddy_resume.pdf"
    default_linkedin_src = repo_root / "Profile.pdf"
    if not default_linkedin_src.exists():
        default_linkedin_src = default_data_dir / "Profile.pdf"

    parser = argparse.ArgumentParser(description="Refresh chatbot data files")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--index-html", default=str(default_index_html))
    parser.add_argument("--resume-src", default=str(default_resume_src))
    parser.add_argument("--linkedin-src", default=str(default_linkedin_src))
    parser.add_argument("--env-path", default=str(repo_root / "chatbot-backend" / ".env"))
    parser.add_argument("--ingest", action="store_true", help="Trigger /api/ingest after refresh")
    parser.add_argument("--ingest-url", default="http://localhost:8000/api/ingest")
    parser.add_argument("--sources", default="resume,portfolio,github,linkedin")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and auto-refresh")
    parser.add_argument("--interval", type=float, default=2.0, help="Watch poll interval (seconds)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.watch:
        watch(args)
        return 0

    refresh_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
