from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .image_processor import STAMPS_DIR


def find_best_matches(
    grid_colors: np.ndarray,
    stamp_colors: dict[str, list[list[float]]],
    no_adjacent_duplicate: bool = False,
) -> list[list[str]]:
    """各グリッドセルに最も色が近いスタンプを割り当てる。
    4象限それぞれの色差の合計で比較する。
    Returns: [row][col] = スタンプ名（拡張子なし）
    """
    rows, cols = grid_colors.shape[:2]
    stamp_names = list(stamp_colors.keys())
    # [num_stamps, 4, 3] → [num_stamps, 12] に平坦化して距離計算を高速化
    stamp_flat = np.array(
        [sum(quads, []) for quads in stamp_colors.values()], dtype=np.float64
    )

    result: list[list[str]] = []

    for r in range(rows):
        row_result: list[str] = []
        for c in range(cols):
            # [4, 3] → [12]
            target = grid_colors[r, c].flatten()
            distances = np.sqrt(((stamp_flat - target) ** 2).sum(axis=1))
            sorted_indices = np.argsort(distances)

            if no_adjacent_duplicate:
                chosen = _pick_non_duplicate(sorted_indices, stamp_names, row_result, result, c)
            else:
                chosen = stamp_names[sorted_indices[0]]

            row_result.append(chosen)
        result.append(row_result)

    return result


def _pick_non_duplicate(
    sorted_indices: np.ndarray,
    names: list[str],
    current_row: list[str],
    prev_rows: list[list[str]],
    c: int,
) -> str:
    """隣接セルと同じスタンプを避けて選ぶ。"""
    neighbors = set()
    if current_row:
        neighbors.add(current_row[-1])
    if prev_rows and c < len(prev_rows[-1]):
        neighbors.add(prev_rows[-1][c])

    for idx in sorted_indices[:20]:
        candidate = names[idx]
        if candidate not in neighbors:
            return candidate
    return names[sorted_indices[0]]


def find_best_matches_pixel(
    grid_pixels: np.ndarray,
    stamp_names: list[str],
    stamp_matrix: np.ndarray,
    rows: int,
    cols: int,
    no_adjacent_duplicate: bool = False,
) -> list[list[str]]:
    """全画素比較で各グリッドセルに最も近いスタンプを割り当てる。
    grid_pixels: [rows*cols, D] uint8
    stamp_matrix: [N, D] uint8
    """
    grid_f = grid_pixels.astype(np.float32)
    stamp_f = stamp_matrix.astype(np.float32)

    result: list[list[str]] = []
    idx = 0
    for r in range(rows):
        row_result: list[str] = []
        for c in range(cols):
            cell = grid_f[idx]
            diffs = stamp_f - cell
            distances = (diffs * diffs).sum(axis=1)
            sorted_indices = np.argsort(distances)

            if no_adjacent_duplicate:
                chosen = _pick_non_duplicate(sorted_indices, stamp_names, row_result, result, c)
            else:
                chosen = stamp_names[sorted_indices[0]]

            row_result.append(chosen)
            idx += 1
        result.append(row_result)

    return result


def compose_mosaic(
    matches: list[list[str]],
    cell_size: int = 32,
    output_path: str = "output/mosaic.png",
) -> Path:
    """スタンプ画像を丸ごと並べてモザイク画像を合成。"""
    rows = len(matches)
    cols = len(matches[0])

    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), (255, 255, 255))

    for r, row in enumerate(matches):
        for c, stamp_name in enumerate(row):
            stamp_path = _find_stamp_file(stamp_name)
            if not stamp_path:
                continue
            img = Image.open(stamp_path).convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img)
            rgb = bg.convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
            canvas.paste(rgb, (c * cell_size, r * cell_size))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"  モザイク画像を保存: {out} ({canvas.size[0]}x{canvas.size[1]})")
    return out


def generate_slack_text(matches: list[list[str]], output_path: str = "output/mosaic.txt") -> Path:
    """スタンプ名を :emoji: 形式のテキストファイルに出力。"""
    lines = []
    for row in matches:
        line = "".join(f":{name}:" for name in row)
        lines.append(line)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  Slackテキストを保存: {out} ({len(matches)}行 × {len(matches[0])}列)")
    return out


def _find_stamp_file(name: str) -> Path | None:
    """スタンプ名からファイルパスを探す。"""
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        p = STAMPS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None
