from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

STAMPS_DIR = Path(__file__).parent.parent / "data" / "stamps"
EMOJI_CACHE = Path(__file__).parent.parent / "data" / "emoji_list.json"


def get_token() -> str:
    token = os.environ.get("SLACK_TOKEN", "")
    if not token or token == "xoxb-your-token-here":
        raise RuntimeError("SLACK_TOKEN が .env に設定されていません")
    return token


def fetch_emoji_list(token: str | None = None) -> dict[str, str]:
    """emoji.list API でカスタム絵文字の名前→画像URL辞書を返す。エイリアスは除外。"""
    token = token or get_token()
    resp = requests.get(
        "https://slack.com/api/emoji.list",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok"):
        raise RuntimeError(f"Slack API エラー: {data.get('error', 'unknown')}")

    emoji = data.get("emoji", {})
    # エイリアス（alias:xxx）を除外し、実画像URLのみ
    return {name: url for name, url in emoji.items() if not url.startswith("alias:")}


def download_stamps(emoji_dict: dict[str, str], force: bool = False) -> list[Path]:
    """絵文字画像をダウンロードして data/stamps/ に保存。キャッシュ済みはスキップ。"""
    STAMPS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    total = len(emoji_dict)

    for i, (name, url) in enumerate(emoji_dict.items(), 1):
        ext = _guess_extension(url)
        path = STAMPS_DIR / f"{name}{ext}"

        if path.exists() and not force:
            downloaded.append(path)
            continue

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            downloaded.append(path)
        except requests.RequestException as e:
            print(f"  [{i}/{total}] スキップ: {name} ({e})")
            continue

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] ダウンロード中...")

    return downloaded


def _guess_extension(url: str) -> str:
    """URLから拡張子を推測。"""
    lower = url.lower().split("?")[0]
    if lower.endswith(".gif"):
        return ".gif"
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return ".jpg"
    return ".png"


def save_emoji_cache(emoji_dict: dict[str, str]) -> None:
    """絵文字リストをJSONキャッシュに保存。"""
    EMOJI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMOJI_CACHE.write_text(json.dumps(emoji_dict, ensure_ascii=False, indent=2))


def load_emoji_cache() -> dict[str, str] | None:
    """キャッシュから絵文字リストを読み込み。なければNone。"""
    if EMOJI_CACHE.exists():
        return json.loads(EMOJI_CACHE.read_text())
    return None
