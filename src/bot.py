from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from .image_processor import load_pixel_cache, split_input_image_pixels
from .mosaic import find_best_matches_pixel, compose_mosaic, generate_slack_text

load_dotenv()


def _download_slack_file(url: str, suffix: str) -> str:
    import urllib.parse
    token = os.environ['SLACK_BOT_TOKEN']
    # リダイレクト先が *.slack.com なら認証ヘッダーを継続送信、外部CDN(S3等)なら除去する。
    # geniee.slack.com → files.slack.com のような同一組織内リダイレクトに対応するため手動追跡。
    for _ in range(5):
        is_slack = urllib.parse.urlparse(url).netloc.endswith("slack.com")
        headers = {"Authorization": f"Bearer {token}"} if is_slack else {}
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308):
            url = resp.headers["Location"]
            continue
        break
    resp.raise_for_status()
    content = resp.content
    content_type = resp.headers.get('Content-Type', '?')
    print(f"[bot] ダウンロードサイズ: {len(content)} bytes, Content-Type: {content_type}")
    if 'text/html' in content_type:
        print(f"[bot] HTMLレスポンス（先頭300字）: {content[:300].decode('utf-8', errors='replace')}")
        raise RuntimeError("画像ではなくHTMLが返されました（認証エラーの可能性）")
    if len(content) == 0:
        raise RuntimeError("ダウンロードしたファイルが空です")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def _generate_mosaic(input_path: str) -> tuple[str, str]:
    """モザイク画像(.png)とSlackテキスト(.txt)を生成して一時ファイルパスを返す。"""
    result = load_pixel_cache(min_opacity=0.5, exclude_animated=True)
    if not result:
        raise RuntimeError("キャッシュが見つかりません。process-stamps を実行してください。")

    names, stamp_matrix, pixel_size = result
    grid_pixels, rows, cols = split_input_image_pixels(input_path, 30, pixel_size)
    matches = find_best_matches_pixel(
        grid_pixels, names, stamp_matrix, rows, cols, no_adjacent_duplicate=True
    )

    img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img_tmp.close()
    txt_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    txt_tmp.close()

    compose_mosaic(matches, cell_size=32, output_path=img_tmp.name)
    generate_slack_text(matches, output_path=txt_tmp.name)
    return img_tmp.name, txt_tmp.name


def _get_upload_ts(upload_resp, channel_id: str, client) -> str | None:
    """アップロードしたファイルのメッセージ ts をチャンネル履歴から取得する。
    files_upload_v2 はレスポンス返却後にチャンネル投稿が完了するため、リトライで待つ。"""
    import time
    data = upload_resp.data if hasattr(upload_resp, "data") else upload_resp
    uploaded = data.get("file") or (data.get("files") or [{}])[0]
    file_id = uploaded.get("id")
    print(f"[bot] uploaded file_id: {file_id}")
    if not file_id:
        return None
    for attempt in range(5):
        history = client.conversations_history(channel=channel_id, limit=10)
        for msg in history.get("messages", []):
            if file_id in [f.get("id") for f in msg.get("files", [])]:
                return msg["ts"]
        print(f"[bot] 履歴にファイル未確認 (attempt {attempt + 1}/5), 1秒後にリトライ...")
        time.sleep(1)
    print(f"[bot] file_id={file_id} を含むメッセージが見つかりませんでした")
    return None


def handle_mention(event, client, logger):
    print(f"[bot] メンション受信: channel={event['channel']} ts={event.get('ts')}")

    files = event.get("files", [])
    image_files = [f for f in files if f.get("mimetype", "").startswith("image/")]
    if not image_files:
        print("[bot] 画像が添付されていないためスキップ")
        return

    channel_id = event["channel"]
    file_info = image_files[0]
    suffix = "." + file_info.get("mimetype", "image/jpeg").split("/")[-1]
    print(f"[bot] 画像検出: {file_info.get('name', '(unknown)')}")

    input_path = img_path = txt_path = None
    try:
        download_url = file_info.get("url_private_download") or file_info.get("url_private")
        print(f"[bot] 画像をダウンロード中... ({download_url})")
        input_path = _download_slack_file(download_url, suffix)
        print(f"[bot] ダウンロード完了: {input_path}")

        print("[bot] モザイク生成中...")
        img_path, txt_path = _generate_mosaic(input_path)
        print(f"[bot] 生成完了: {img_path}")

        print("[bot] 画像をSlackにアップロード中...")
        upload_resp = client.files_upload_v2(
            channel=channel_id,
            file=img_path,
            filename="mosaic.png",
            title="Stamp Art",
        )
        print(f"[bot] アップロード完了")

        message_ts = _get_upload_ts(upload_resp, channel_id, client)
        print(f"[bot] message_ts: {message_ts}")
        if message_ts:
            print(f"[bot] スレッドにテキストを送信中: thread_ts={message_ts}")
            stamp_text = Path(txt_path).read_text()
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,
                text=stamp_text,
            )
            print("[bot] 完了!")
        else:
            print("[bot] 警告: message_ts が取得できなかったためスレッド返信をスキップ")

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"[bot] エラー: {e}")
    finally:
        for path in (input_path, img_path, txt_path):
            if path:
                Path(path).unlink(missing_ok=True)


def start() -> None:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    app_token = os.environ.get("SLACK_APP_TOKEN", "")
    if not bot_token:
        raise RuntimeError("SLACK_BOT_TOKEN が .env に設定されていません")
    if not app_token:
        raise RuntimeError("SLACK_APP_TOKEN が .env に設定されていません")

    app = App(token=bot_token)
    app.event("app_mention")(handle_mention)

    @app.event("message")
    def debug_message(event):
        print(f"[debug] message event: {event}")

    SocketModeHandler(app, app_token).start()
