import cv2
import numpy as np
from collections import Counter
import urllib.request
import sys
import os

# =========================
# ASSETS (cố định trong repo)
# =========================
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ZENO_LOGO  = os.path.join(ASSETS_DIR, "logo_zenohomes.png")
HOTLINE_PNG = os.path.join(ASSETS_DIR, "hotline.png")

# =========================
# TỌA ĐỘ LOGO PACIFIC (đo tay, cố định)
# =========================
LOGO_BOX_LEFT  = (124, 612, 354, 675)
LOGO_BOX_RIGHT = (540, 420, 775, 482)

# =========================
# HÀM TIỆN ÍCH
# =========================

def paste_rgba(base_img, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    roi = base_img[y:y+h, x:x+w]
    if overlay_rgba.shape[2] == 4:
        alpha = overlay_rgba[:, :, 3:4] / 255.0
        roi[:] = (alpha * overlay_rgba[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    else:
        roi[:] = overlay_rgba
    base_img[y:y+h, x:x+w] = roi


def fit_resize(overlay, max_w, max_h):
    oh, ow = overlay.shape[:2]
    scale = min(max_w / ow, max_h / oh)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    return cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_to_content(overlay_rgba, alpha_thresh=10, margin=6):
    alpha = overlay_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_thresh)
    if len(xs) == 0:
        return overlay_rgba
    x1 = max(xs.min() - margin, 0)
    x2 = min(xs.max() + margin, overlay_rgba.shape[1])
    y1 = max(ys.min() - margin, 0)
    y2 = min(ys.max() + margin, overlay_rgba.shape[0])
    return overlay_rgba[y1:y2, x1:x2]


def gold_ratio(img, box):
    x1, y1, x2, y2 = box
    region = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 70, 110])
    upper = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask.sum() / 255 / mask.size


def detect_banner_box(img, sample_x=40):
    h, w = img.shape[:2]
    col = img[:, sample_x, :].astype(int)

    search_zone = col[int(h * 0.85):int(h * 0.99)]
    counter = Counter(tuple(c) for c in search_zone)
    banner_left_color, _ = counter.most_common(1)[0]
    banner_left_color = np.array(banner_left_color)

    matches_y = np.where(np.abs(col - banner_left_color).sum(axis=1) < 10)[0]
    y1, y2 = int(matches_y.min()), int(matches_y.max())

    mid_y = (y1 + y2) // 2
    row = img[mid_y, :, :].astype(int)

    above_row = img[max(y1 - 20, 0), :, :].astype(int)
    counter2 = Counter(tuple(c) for c in above_row)
    bg_color, _ = counter2.most_common(1)[0]
    bg_color = np.array(bg_color)

    diffs = np.abs(row - bg_color).sum(axis=1)
    inside = np.where(diffs > 25)[0]
    x1b, x2b = int(inside.min()), int(inside.max())

    return x1b, y1, x2b, y2


def detect_logo_box(img):
    score_left  = gold_ratio(img, LOGO_BOX_LEFT)
    score_right = gold_ratio(img, LOGO_BOX_RIGHT)
    print(f"Gold ratio - LEFT: {score_left:.4f} | RIGHT: {score_right:.4f}")
    if score_left >= score_right:
        print("=> Layout: LOGO BÊN TRÁI")
        return LOGO_BOX_LEFT
    else:
        print("=> Layout: LOGO BÊN PHẢI")
        return LOGO_BOX_RIGHT


# =========================
# HÀM CHÍNH
# =========================

def process_image(image_url: str, output_path: str):
    # --- Tải ảnh gốc từ URL ---
    print(f"Đang tải ảnh: {image_url}")
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh từ URL")

    # --- 1) Thay logo ---
    logo_box = detect_logo_box(img)
    lx1, ly1, lx2, ly2 = logo_box
    logo_w, logo_h = lx2 - lx1, ly2 - ly1

    corner_samples = [
        img[ly1+1, lx1+1], img[ly1+1, lx2-2],
        img[ly2-2, lx1+1], img[ly2-2, lx2-2],
    ]
    bg_sample = np.median(corner_samples, axis=0).astype(np.uint8).tolist()
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), bg_sample, -1)

    zeno = cv2.imread(ZENO_LOGO, cv2.IMREAD_UNCHANGED)
    if zeno is None:
        raise FileNotFoundError(f"Không đọc được {ZENO_LOGO}")
    zeno = crop_to_content(zeno)
    zeno_fit = fit_resize(zeno, int(logo_w * 0.95), int(logo_h * 0.95))
    zh, zw = zeno_fit.shape[:2]
    paste_rgba(img, zeno_fit, lx1 + (logo_w - zw) // 2, ly1 + (logo_h - zh) // 2)

    # --- 2) Thay hotline ---
    bx1, by1, bx2, by2 = detect_banner_box(img)
    print(f"Banner: ({bx1},{by1}) -> ({bx2},{by2})")
    banner_w, banner_h = bx2 - bx1, by2 - by1

    mid_y = (by1 + by2) // 2
    color_left_real  = img[mid_y, bx1+5].astype(np.float64)
    color_right_real = img[mid_y, bx2-5].astype(np.float64)

    gradient = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
    t = np.linspace(0, 1, banner_w).reshape(1, -1, 1)
    gradient[:] = (color_left_real*(1-t) + color_right_real*t).astype(np.uint8)
    img[by1:by2, bx1:bx2] = gradient

    hotline = cv2.imread(HOTLINE_PNG, cv2.IMREAD_UNCHANGED)
    if hotline is None:
        raise FileNotFoundError(f"Không đọc được {HOTLINE_PNG}")
    hotline = crop_to_content(hotline)
    hotline_fit = fit_resize(hotline, int(banner_w * 0.8), int(banner_h * 0.95))
    hh, hw = hotline_fit.shape[:2]
    paste_rgba(img, hotline_fit,
               bx1 + (banner_w - hw) // 2,
               by1 + (banner_h - hh) // 2)

    # --- Lưu kết quả ---
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Đã lưu: {output_path}")


# =========================
# ENTRY POINT (gọi từ CLI hoặc Actions)
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process.py <image_url> <output_path>")
        sys.exit(1)
    process_image(sys.argv[1], sys.argv[2])
