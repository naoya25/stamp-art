from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

STAMPS_DIR = Path(__file__).parent.parent / "data" / "stamps"
COLOR_CACHE = Path(__file__).parent.parent / "data" / "stamp_quad_colors.json"


def _load_rgba(img_path: Path, size: int = 64) -> Image.Image:
    """画像を読み込み RGBA にしてリサイズ。"""
    img = Image.open(img_path).convert("RGBA")
    return img.resize((size, size), Image.LANCZOS)


def _to_rgb_array(img_path: Path, size: int = 64) -> np.ndarray:
    """画像を読み込み、透過→白背景にして RGB の numpy 配列を返す。"""
    img = _load_rgba(img_path, size)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    return np.array(bg.convert("RGB"))


def calc_opacity(img_path: Path, size: int = 64) -> float:
    """画像の不透明率(0.0〜1.0)を返す。アルファ値128以上のピクセルの割合。"""
    img = _load_rgba(img_path, size)
    alpha = np.array(img)[:, :, 3]
    return float((alpha >= 128).sum()) / alpha.size


def calc_quad_colors(arr: np.ndarray) -> list[list[float]]:
    """画像配列を2×2に分割し、各象限の平均RGB (計4×3=12値) を返す。
    順序: [tl, tr, bl, br]  各要素は [R, G, B]
    """
    h, w = arr.shape[:2]
    mh, mw = h // 2, w // 2
    quads = [
        arr[:mh, :mw],       # top-left
        arr[:mh, mw:],       # top-right
        arr[mh:, :mw],       # bottom-left
        arr[mh:, mw:],       # bottom-right
    ]
    return [q.mean(axis=(0, 1)).tolist() for q in quads]


def process_all_stamps(stamps_dir: Path, size: int = 64) -> dict[str, dict]:
    """全スタンプの4象限平均色と不透明率を計算してキャッシュに保存。
    Returns: {スタンプ名: {"colors": [[R,G,B]x4], "opacity": float}, ...}
    """
    stamp_map: dict[str, dict] = {}
    stamp_files = sorted(stamps_dir.glob("*"))
    stamp_files = [f for f in stamp_files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")]
    total = len(stamp_files)

    for i, stamp_path in enumerate(stamp_files, 1):
        try:
            arr = _to_rgb_array(stamp_path, size)
            opacity = calc_opacity(stamp_path, size)
            stamp_map[stamp_path.stem] = {
                "colors": calc_quad_colors(arr),
                "opacity": round(opacity, 3),
            }
        except Exception as e:
            print(f"  [{i}/{total}] スキップ: {stamp_path.name} ({e})")
            continue

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] 処理中...")

    _save_cache(stamp_map)
    print(f"  {len(stamp_map)} スタンプの色情報を計算完了")
    return stamp_map


def split_input_image(image_path: str, grid_size: int) -> tuple[np.ndarray, int, int]:
    """入力画像をグリッドに分割し、各セルの4象限平均色を計算。
    長辺基準で中央クロップし、正方形セルにする。
    Returns: (色配列 [rows, cols, 4, 3], rows, cols)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    long = max(w, h)
    cell_size = long // grid_size
    cols = w // cell_size
    rows = h // cell_size

    # 中央クロップ（余白を左右 or 上下均等にカット）
    crop_w = cols * cell_size
    crop_h = rows * cell_size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))
    arr = np.array(img)

    cell_w = cell_size
    cell_h = cell_size

    # [rows, cols, 4象限, 3(RGB)]
    colors = np.zeros((rows, cols, 4, 3), dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            cell = arr[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            colors[r, c] = calc_quad_colors(cell)

    return colors, rows, cols


def _save_cache(stamp_map: dict[str, dict]) -> None:
    COLOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COLOR_CACHE.write_text(json.dumps(stamp_map, ensure_ascii=False, indent=2))


def load_color_cache(min_opacity: float = 0.0) -> Optional[dict[str, list[list[float]]]]:
    """キャッシュを読み込み、min_opacity 未満のスタンプを除外して色情報のみ返す。"""
    if not COLOR_CACHE.exists():
        return None
    raw = json.loads(COLOR_CACHE.read_text())
    filtered = {}
    for name, data in raw.items():
        if isinstance(data, dict):
            if data.get("opacity", 1.0) >= min_opacity:
                filtered[name] = data["colors"]
        else:
            filtered[name] = data
    return filtered
