from __future__ import annotations

import argparse
import sys

from .slack_client import (
    fetch_emoji_list,
    download_stamps,
    save_emoji_cache,
    STAMPS_DIR,
)
from .image_processor import (
    process_all_stamps,
    split_input_image,
    load_color_cache,
)
from .mosaic import find_best_matches, compose_mosaic, generate_slack_text


def cmd_fetch_stamps(args: argparse.Namespace) -> None:
    """Slackからスタンプを取得してダウンロード。"""
    print("スタンプ一覧を取得中...")
    emoji_dict = fetch_emoji_list()
    save_emoji_cache(emoji_dict)
    print(f"  {len(emoji_dict)} 個のカスタム絵文字を取得")

    print("画像をダウンロード中...")
    paths = download_stamps(emoji_dict, force=args.force)
    print(f"  {len(paths)} 個の画像を保存済み → {STAMPS_DIR}")


def cmd_process_stamps(args: argparse.Namespace) -> None:
    """ダウンロード済みスタンプの4象限色情報を計算。"""
    print("スタンプの4象限平均色を計算中...")
    color_map = process_all_stamps(STAMPS_DIR, size=args.size)
    print(f"  完了: {len(color_map)} スタンプ")


def cmd_generate(args: argparse.Namespace) -> None:
    """モザイクアートを生成。"""
    color_map = load_color_cache(min_opacity=args.min_opacity)
    if not color_map:
        print("色情報がありません。先に process-stamps を実行してください。")
        sys.exit(1)
    print(f"  不透明率 {args.min_opacity:.0%} 以上のスタンプ: {len(color_map)} 個")

    print(f"入力画像を {args.grid}x グリッドに分割中...")
    grid_colors, rows, cols = split_input_image(args.input, args.grid)
    print(f"  グリッド: {rows}行 × {cols}列")

    print("最適なスタンプをマッチング中（4象限比較）...")
    matches = find_best_matches(
        grid_colors, color_map, no_adjacent_duplicate=args.no_duplicate
    )

    output_base = args.output or f"output/mosaic_{args.grid}x{args.grid}"
    if output_base.endswith(".png") or output_base.endswith(".txt"):
        output_base = output_base.rsplit(".", 1)[0]

    print("モザイクを合成中...")
    compose_mosaic(matches, cell_size=args.cell_size, output_path=f"{output_base}.png")

    if args.slack:
        print("Slack用テキストを生成中...")
        generate_slack_text(matches, output_path=f"{output_base}.txt")

    print("完了!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slack スタンプでモザイクアートを生成する CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch-stamps
    p_fetch = sub.add_parser("fetch-stamps", help="Slackからスタンプ画像を取得")
    p_fetch.add_argument("--force", action="store_true", help="既存画像も再ダウンロード")
    p_fetch.set_defaults(func=cmd_fetch_stamps)

    # process-stamps
    p_proc = sub.add_parser("process-stamps", help="スタンプの4象限色情報を計算")
    p_proc.add_argument("--size", type=int, default=64, help="分析時のリサイズサイズ (px)")
    p_proc.set_defaults(func=cmd_process_stamps)

    # generate
    p_gen = sub.add_parser("generate", help="モザイクアートを生成")
    p_gen.add_argument("--input", "-i", required=True, help="入力画像パス")
    p_gen.add_argument("--grid", "-g", type=int, default=50, help="グリッドサイズ (1辺のマス数)")
    p_gen.add_argument("--output", "-o", help="出力ファイルのベース名")
    p_gen.add_argument("--cell-size", type=int, default=32, help="出力画像のセルサイズ (px)")
    p_gen.add_argument("--no-duplicate", action="store_true", help="隣接セルでの同一スタンプ使用を抑制")
    p_gen.add_argument("--slack", action="store_true", help="Slack貼り付け用の .txt も出力")
    p_gen.add_argument("--min-opacity", type=float, default=0.5, help="スタンプの最低不透明率 0.0〜1.0 (デフォルト: 0.5)")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
