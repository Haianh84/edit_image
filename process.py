import cv2
import numpy as np
import urllib.request
import sys
import os

# =========================
# ASSETS (cố định trong repo)
# =========================
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ZENO_LOGO   = os.path.join(ASSETS_DIR, "logo_zenohomes.png")
HOTLINE_PNG = os.path.join(ASSETS_DIR, "hotline.png")

# =========================
# TỶ LỆ TỌA ĐỘ CHUẨN (Tính toán dựa trên ảnh gốc mẫu 905 x 1280)
# Công thức: tỷ lệ = tọa độ / kích thước gốc
# =========================
# LOGO_BOX = (x1, y1, x2, y2)
# Mẫu trái: (24, 587, 440, 680)
LOGO_PCT_LEFT  = (24/905, 587/1280, 440/905, 680/1280)
# Mẫu phải: (540, 420, 775, 482)
LOGO_PCT_RIGHT = (540/905, 420/1280, 775/905, 482/1280)

# BANNER_BOX = (x1, y1, x2, y2) -> Mẫu: (24, 1184, 878, 1251)
BANNER_PCT = (24/905, 1184/1280, 878/905, 1251/1280)

# =========================
# HÀM TIỆN ÍCH
# =========================

def paste_rgba(base_img, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    bh, bw = base_img.shape[:2]
    h = min(h, bh - y)
    w = min(w, bw - x)
    if h <= 0 or w <= 0:
        return
    roi = base_img[y:y+h, x:x+w]
    ov  = overlay_rgba[:h, :w]
    if ov.shape[2] == 4:
        alpha = ov[:, :, 3:4] / 255.0
        roi[:] = (alpha * ov[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    else:
        roi[:] = ov
    base_img[y:y+h, x:x+w] = roi

def fit_resize(overlay, max_w, max_h):
    oh, ow = overlay.shape[:2]
    scale = min(max_w / ow, max_h / oh)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    return cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

def crop_to_content(overlay_rgba, alpha_thresh=10, margin=4):
    if overlay_rgba.shape[2] < 4:
        return overlay_rgba
    alpha = overlay_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_thresh)
    if len(xs) == 0:
        return overlay_rgba
    x1 = max(xs.min() - margin, 0)
    x2 = min(xs.max() + margin, overlay_rgba.shape[1])
    y1 = max(ys.min() - margin, 0)
    y2 = min(ys.max() + margin, overlay_rgba.shape[0])
    return overlay_rgba[y1:y2, x1:x2]

def get_absolute_box(pct_box, img_w, img_h):
    """Chuyển đổi từ tỷ lệ % sang tọa độ pixel thực tế dựa trên kích thước ảnh"""
    x1 = int(pct_box[0] * img_w)
    y1 = int(pct_box[1] * img_h)
    x2 = int(pct_box[2] * img_w)
    y2 = int(pct_box[3] * img_h)
    # Giới hạn không vượt quá biên ảnh
    return (max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2))

def detect_logo_box(img, img_w, img_h):
    """Phát hiện layout dựa vào màu nền xanh đậm đặc trưng tại 2 vị trí."""
    box_left = get_absolute_box(LOGO_PCT_LEFT, img_w, img_h)
    box_right = get_absolute_box(LOGO_PCT_RIGHT, img_w, img_h)

    lx1, ly1, lx2, ly2 = box_left
    rx1, ry1, rx2, ry2 = box_right

    mid_lx = (lx1 + lx2) // 2
    mid_ly = (ly1 + ly2) // 2
    mid_rx = (rx1 + rx2) // 2
    mid_ry = (ry1 + ry2) // 2

    bg_green = np.array([32, 64, 7])

    color_left  = img[mid_ly, mid_lx].astype(int)
    color_right = img[min(mid_ry, img_h - 1), min(mid_rx, img_w - 1)].astype(int)

    diff_left  = np.abs(color_left  - bg_green).sum()
    diff_right = np.abs(color_right - bg_green).sum()

    print(f"Diff to green - LEFT: {diff_left} | RIGHT: {diff_right}")

    if diff_left <= diff_right:
        print("=> Layout: LOGO BÊN TRÁI")
        return box_left
    else:
        print("=> Layout: LOGO BÊN PHẢI")
        return box_right

# =========================
# HÀM CHÍNH
# =========================

def process_image(image_url: str, output_path: str):
    print(f"Đang tải ảnh: {image_url}")
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh từ URL")

    img_h, img_w = img.shape[:2]
    print(f"Kích thước ảnh thực tế: {img_w}x{img_h}")

    # --- 1) Thay logo ---
    logo_box = detect_logo_box(img, img_w, img_h)
    lx1, ly1, lx2, ly2 = logo_box
    logo_w, logo_h = lx2 - lx1, ly2 - ly1

    # Lấy màu nền thực tế từ pixel ngay bên cạnh vùng logo
    sample_points = []
    if lx1 > 5:
        for dy in range(0, logo_h, max(1, logo_h // 8)):
            sample_points.append(img[ly1 + dy, lx1 - 5])
    if lx2 < img_w - 5:
        for dy in range(0, logo_h, max(1, logo_h // 8)):
            sample_points.append(img[ly1 + dy, lx2 + 5])
    if ly1 > 5:
        for dx in range(0, logo_w, max(1, logo_w // 8)):
            sample_points.append(img[ly1 - 5, lx1 + dx])

    if len(sample_points) == 0:
        sample_points.append([32, 64, 7]) # Fallback màu mặc định nếu không lấy được mẫu

    bg_color = np.median(sample_points, axis=0).astype(int).tolist()
    print(f"Màu nền lấy từ ảnh: BGR={bg_color}")
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), bg_color, -1)

    zeno = cv2.imread(ZENO_LOGO, cv2.IMREAD_UNCHANGED)
    if zeno is None:
        raise FileNotFoundError(f"Không đọc được {ZENO_LOGO}")
    zeno = crop_to_content(zeno)
    zeno_fit = fit_resize(zeno, int(logo_w * 0.85), int(logo_h * 0.80))
    zh, zw = zeno_fit.shape[:2]
    zx = lx1 + (logo_w - zw) // 2
    zy = ly1 + (logo_h - zh) // 2
    paste_rgba(img, zeno_fit, zx, zy)
    print(f"Đã dán logo Zeno: {zw}x{zh} tại ({zx},{zy})")

    # --- 2) Thay hotline ---
    bx1, by1, bx2, by2 = get_absolute_box(BANNER_PCT, img_w, img_h)
    print(f"Banner (tính theo tỷ lệ): ({bx1},{by1}) -> ({bx2},{by2})")
    banner_w, banner_h = bx2 - bx1, by2 - by1

    mid_y = (by1 + by2) // 2
    # Đảm bảo điểm lấy mẫu không bị vọt ra ngoài biên ảnh
    sample_x_left = min(bx1 + 5, img_w - 1)
    sample_x_right = max(bx2 - 5, 0)
    
    color_left_real  = img[mid_y, sample_x_left].astype(np.float64)
    color_right_real = img[mid_y, sample_x_right].astype(np.float64)

    gradient = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
    t = np.linspace(0, 1, banner_w).reshape(1, -1, 1)
    gradient[:] = (color_left_real * (1 - t) + color_right_real * t).astype(np.uint8)
    img[by1:by2, bx1:bx2] = gradient

    hotline = cv2.imread(HOTLINE_PNG, cv2.IMREAD_UNCHANGED)
    if hotline is None:
        raise FileNotFoundError(f"Không đọc được {HOTLINE_PNG}")
    hotline = crop_to_content(hotline)
    hotline_fit = fit_resize(hotline, int(banner_w * 0.78), int(banner_h * 0.82))
    hh, hw = hotline_fit.shape[:2]
    hx = bx1 + (banner_w - hw) // 2
    hy = by1 + (banner_h - hh) // 2
    paste_rgba(img, hotline_fit, hx, hy)
    print(f"Đã dán hotline: {hw}x{hh} tại ({hx},{hy})")

    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Đã lưu thành công: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process.py <image_url> <output_path>")
        sys.exit(1)
    process_image(sys.argv[1], sys.argv[2])
