from __future__ import annotations

import argparse
import sys
from datetime import datetime

from .slack_client import (
    fetch_emoji_list,
    download_stamps,
    save_emoji_cache,
    STAMPS_DIR,
)
from .image_processor import (
    process_all_stamps,
    split_input_image,
    split_input_image_pixels,
    split_input_image_kmeans,
    load_color_cache,
    load_pixel_cache,
    load_kmeans_cache,
)
from .mosaic import find_best_matches, find_best_matches_pixel, compose_mosaic, generate_slack_text


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
    """ダウンロード済みスタンプの色情報を計算。"""
    print("スタンプの色情報を計算中...")
    color_map = process_all_stamps(STAMPS_DIR, size=args.size, pixel_size=args.pixel_size)
    print(f"  完了: {len(color_map)} スタンプ")


def cmd_generate(args: argparse.Namespace) -> None:
    """モザイクアートを生成。"""
    filters = []
    if args.min_opacity > 0:
        filters.append(f"不透明率{args.min_opacity:.0%}以上")
    if args.no_animated:
        filters.append("アニメーション除外")
    label = "、".join(filters) if filters else "フィルタなし"

    if args.method == "pixel":
        result = load_pixel_cache(min_opacity=args.min_opacity, exclude_animated=args.no_animated)
        if not result:
            print("ピクセルデータがありません。先に process-stamps を実行してください。")
            sys.exit(1)
        names, stamp_matrix, pixel_size = result
        print(f"  対象スタンプ: {len(names)} 個 ({label})")

        print(f"入力画像を {args.grid}x グリッドに分割中...")
        grid_pixels, rows, cols = split_input_image_pixels(args.input, args.grid, pixel_size)
        print(f"  グリッド: {rows}行 × {cols}列")

        print("最適なスタンプをマッチング中（全画素比較）...")
        matches = find_best_matches_pixel(
            grid_pixels, names, stamp_matrix, rows, cols, no_adjacent_duplicate=args.no_duplicate
        )
    elif args.method == "kmeans":
        color_map = load_kmeans_cache(min_opacity=args.min_opacity, exclude_animated=args.no_animated)
        if not color_map:
            print("K-meansデータがありません。先に process-stamps を実行してください。")
            sys.exit(1)
        print(f"  対象スタンプ: {len(color_map)} 個 ({label})")

        print(f"入力画像を {args.grid}x グリッドに分割中...")
        grid_colors, rows, cols = split_input_image_kmeans(args.input, args.grid)
        print(f"  グリッド: {rows}行 × {cols}列")

        print("最適なスタンプをマッチング中（K-means比較）...")
        matches = find_best_matches(
            grid_colors, color_map, no_adjacent_duplicate=args.no_duplicate
        )
    else:
        color_map = load_color_cache(min_opacity=args.min_opacity, exclude_animated=args.no_animated)
        if not color_map:
            print("色情報がありません。先に process-stamps を実行してください。")
            sys.exit(1)
        print(f"  対象スタンプ: {len(color_map)} 個 ({label})")

        print(f"入力画像を {args.grid}x グリッドに分割中...")
        grid_colors, rows, cols = split_input_image(args.input, args.grid)
        print(f"  グリッド: {rows}行 × {cols}列")

        print("最適なスタンプをマッチング中（4象限比較）...")
        matches = find_best_matches(
            grid_colors, color_map, no_adjacent_duplicate=args.no_duplicate
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output or f"output/{ts}_mosaic_{args.grid}x{args.grid}"
    if output_base.endswith(".png") or output_base.endswith(".txt"):
        output_base = output_base.rsplit(".", 1)[0]

    print("モザイクを合成中...")
    compose_mosaic(matches, cell_size=args.cell_size, output_path=f"{output_base}.png")

    if args.slack:
        print("Slack用テキストを生成中...")
        generate_slack_text(matches, output_path=f"{output_base}.txt")

    print("完了!")


def cmd_bot(args: argparse.Namespace) -> None:
    """Slack Bot を Socket Mode で起動。"""
    from .bot import start
    print("Slack Bot を起動中... (Ctrl+C で停止)")
    start()


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
    p_proc = sub.add_parser("process-stamps", help="スタンプの色情報を計算")
    p_proc.add_argument("--size", type=int, default=64, help="4象限比較用のリサイズサイズ (px)")
    p_proc.add_argument("--pixel-size", type=int, default=32, help="全画素比較用のリサイズサイズ (px, デフォルト: 32)")
    p_proc.set_defaults(func=cmd_process_stamps)

    # generate
    p_gen = sub.add_parser("generate", help="モザイクアートを生成")
    p_gen.add_argument("--input", "-i", required=True, help="入力画像パス")
    p_gen.add_argument("--grid", "-g", type=int, default=30, help="グリッドサイズ (長辺のマス数, デフォルト: 30)")
    p_gen.add_argument("--output", "-o", help="出力ファイルのベース名 (デフォルト: output/{timestamp}_mosaic_{grid}x{grid})")
    p_gen.add_argument("--cell-size", type=int, default=32, help="出力画像の1セルのサイズ (px, デフォルト: 32)")
    p_gen.add_argument("--allow-duplicate", dest="no_duplicate", action="store_false", default=True,
                       help="隣接セルで同じスタンプを許可 (デフォルト: 不許可)")
    p_gen.add_argument("--no-slack", dest="slack", action="store_false", default=True,
                       help="Slack用テキスト出力を無効化 (デフォルト: 出力する)")
    p_gen.add_argument("--min-opacity", type=float, default=0.5, help="スタンプの最低不透明率 0.0〜1.0 (デフォルト: 0.5)")
    p_gen.add_argument("--no-animated", action="store_true", default=False,
                       help="アニメーションスタンプを除外 (デフォルト: 含む)")
    p_gen.add_argument("--method", choices=["pixel", "quad", "kmeans"], default="pixel",
                       help="マッチング手法: pixel=全画素比較, quad=4象限比較, kmeans=K-means支配色比較 (デフォルト: pixel)")
    p_gen.set_defaults(func=cmd_generate)

    # bot
    p_bot = sub.add_parser("bot", help="Slack Bot を起動 (Socket Mode)")
    p_bot.set_defaults(func=cmd_bot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
