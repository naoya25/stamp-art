from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

STAMPS_DIR = Path(__file__).parent.parent / "data" / "stamps"
COLOR_CACHE = Path(__file__).parent.parent / "data" / "stamp_quad_colors.json"
PIXEL_CACHE = Path(__file__).parent.parent / "data" / "stamp_pixels.npz"
KMEANS_CACHE = Path(__file__).parent.parent / "data" / "stamp_kmeans_colors.json"


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


def _is_animated(img_path: Path) -> bool:
    """画像がアニメーション（複数フレーム）かどうかを判定。"""
    try:
        img = Image.open(img_path)
        img.seek(1)
        return True
    except EOFError:
        return False


def _kmeans_dominant(pixels: np.ndarray, k: int = 3, max_iter: int = 10) -> list[float]:
    """pixels: [N, 3] float32。K-meansで最大クラスタの重心色を返す。固定シードで再現性を保証。"""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pixels), size=min(k, len(pixels)), replace=False)
    centers = pixels[idx].astype(np.float32).copy()
    labels = np.zeros(len(pixels), dtype=np.int32)

    for _ in range(max_iter):
        dists = ((pixels[:, None] - centers[None]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = pixels[mask].mean(axis=0)

    counts = np.bincount(labels, minlength=k)
    return centers[counts.argmax()].tolist()


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


def calc_quad_colors_kmeans(arr: np.ndarray, k: int = 3) -> list[list[float]]:
    """画像配列を2×2に分割し、各象限のK-means支配色 (計4×3=12値) を返す。"""
    h, w = arr.shape[:2]
    mh, mw = h // 2, w // 2
    quads = [
        arr[:mh, :mw],
        arr[:mh, mw:],
        arr[mh:, :mw],
        arr[mh:, mw:],
    ]
    return [_kmeans_dominant(q.reshape(-1, 3).astype(np.float32), k=k) for q in quads]


def _load_rgb_image(img_path: Path) -> Image.Image:
    """画像をRGB（白背景合成済み）として読み込む。リサイズなし。"""
    img = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    return bg.convert("RGB")


def process_all_stamps(stamps_dir: Path, size: int = 64, pixel_size: int = 32, kmeans_k: int = 3) -> dict[str, dict]:
    """全スタンプの4象限平均色・K-means色・ピクセル配列・不透明率を計算してキャッシュに保存。
    Returns: {スタンプ名: {"colors": [[R,G,B]x4], "opacity": float}, ...}
    """
    stamp_map: dict[str, dict] = {}
    kmeans_map: dict[str, dict] = {}
    pixel_names: list[str] = []
    pixel_arrays: list[np.ndarray] = []

    stamp_files = sorted(stamps_dir.glob("*"))
    stamp_files = [f for f in stamp_files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")]
    total = len(stamp_files)

    for i, stamp_path in enumerate(stamp_files, 1):
        try:
            rgb = _load_rgb_image(stamp_path)
            quad_arr = np.array(rgb.resize((size, size), Image.LANCZOS))
            pixel_arr = np.array(rgb.resize((pixel_size, pixel_size), Image.LANCZOS))
            opacity = calc_opacity(stamp_path, size)
            animated = _is_animated(stamp_path)
            name = stamp_path.stem
            meta = {"opacity": round(opacity, 3), "animated": animated}
            stamp_map[name] = {**meta, "colors": calc_quad_colors(quad_arr)}
            kmeans_map[name] = {**meta, "colors": calc_quad_colors_kmeans(quad_arr, k=kmeans_k)}
            pixel_names.append(name)
            pixel_arrays.append(pixel_arr.flatten())
        except Exception as e:
            print(f"  [{i}/{total}] スキップ: {stamp_path.name} ({e})")
            continue

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] 処理中...")

    _save_cache(stamp_map)
    _save_kmeans_cache(kmeans_map)
    _save_pixel_cache(pixel_names, pixel_arrays, pixel_size)
    print(f"  {len(stamp_map)} スタンプの色情報を計算完了")
    return stamp_map


def split_input_image_pixels(image_path: str, grid_size: int, pixel_size: int = 32) -> tuple[np.ndarray, int, int]:
    """入力画像をグリッドに分割し、各セルを pixel_size×pixel_size にリサイズして返す。
    Returns: (pixels [rows*cols, pixel_size*pixel_size*3] uint8, rows, cols)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    long = max(w, h)
    cell_size = long // grid_size
    cols = w // cell_size
    rows = h // cell_size

    crop_w = cols * cell_size
    crop_h = rows * cell_size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))
    arr = np.array(img)

    pixels: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            cell = arr[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size]
            cell_img = Image.fromarray(cell).resize((pixel_size, pixel_size), Image.LANCZOS)
            pixels.append(np.array(cell_img).flatten())

    return np.array(pixels, dtype=np.uint8), rows, cols


def split_input_image_kmeans(image_path: str, grid_size: int, k: int = 3) -> tuple[np.ndarray, int, int]:
    """入力画像をグリッドに分割し、各セルの4象限K-means支配色を計算。
    Returns: (色配列 [rows, cols, 4, 3], rows, cols)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    long = max(w, h)
    cell_size = long // grid_size
    cols = w // cell_size
    rows = h // cell_size

    crop_w = cols * cell_size
    crop_h = rows * cell_size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))
    arr = np.array(img)

    colors = np.zeros((rows, cols, 4, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            cell = arr[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size]
            colors[r, c] = calc_quad_colors_kmeans(cell, k=k)

    return colors, rows, cols


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


def _save_kmeans_cache(kmeans_map: dict[str, dict]) -> None:
    KMEANS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    KMEANS_CACHE.write_text(json.dumps(kmeans_map, ensure_ascii=False, indent=2))


def _save_pixel_cache(names: list[str], arrays: list[np.ndarray], pixel_size: int) -> None:
    PIXEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(PIXEL_CACHE),
        names=np.array(names),
        arrays=np.array(arrays, dtype=np.uint8),
        pixel_size=np.array([pixel_size]),
    )


def load_kmeans_cache(
    min_opacity: float = 0.0,
    exclude_animated: bool = False,
) -> Optional[dict[str, list[list[float]]]]:
    """K-meansキャッシュを読み込み、条件に合わないスタンプを除外して色情報のみ返す。"""
    if not KMEANS_CACHE.exists():
        return None
    raw = json.loads(KMEANS_CACHE.read_text())
    filtered = {}
    for name, data in raw.items():
        if isinstance(data, dict):
            if data.get("opacity", 1.0) < min_opacity:
                continue
            if exclude_animated and data.get("animated", False):
                continue
            filtered[name] = data["colors"]
        else:
            filtered[name] = data
    return filtered


def load_pixel_cache(
    min_opacity: float = 0.0,
    exclude_animated: bool = False,
) -> Optional[tuple[list[str], np.ndarray, int]]:
    """ピクセルキャッシュを読み込み、フィルタ済みの (names, matrix, pixel_size) を返す。
    matrix の shape は [N, pixel_size*pixel_size*3] uint8。
    """
    if not PIXEL_CACHE.exists():
        return None

    data = np.load(str(PIXEL_CACHE), allow_pickle=False)
    all_names: list[str] = data["names"].tolist()
    all_arrays: np.ndarray = data["arrays"]
    pixel_size = int(data["pixel_size"][0])

    meta: dict = {}
    if COLOR_CACHE.exists():
        meta = json.loads(COLOR_CACHE.read_text())

    valid_names: list[str] = []
    valid_indices: list[int] = []
    for i, name in enumerate(all_names):
        m = meta.get(name, {})
        if isinstance(m, dict):
            if m.get("opacity", 1.0) < min_opacity:
                continue
            if exclude_animated and m.get("animated", False):
                continue
        valid_names.append(name)
        valid_indices.append(i)

    return valid_names, all_arrays[valid_indices], pixel_size


def load_color_cache(
    min_opacity: float = 0.0,
    exclude_animated: bool = False,
) -> Optional[dict[str, list[list[float]]]]:
    """キャッシュを読み込み、条件に合わないスタンプを除外して色情報のみ返す。"""
    if not COLOR_CACHE.exists():
        return None
    raw = json.loads(COLOR_CACHE.read_text())
    filtered = {}
    for name, data in raw.items():
        if isinstance(data, dict):
            if data.get("opacity", 1.0) < min_opacity:
                continue
            if exclude_animated and data.get("animated", False):
                continue
            filtered[name] = data["colors"]
        else:
            filtered[name] = data
    return filtered
