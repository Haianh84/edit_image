import cv2
import numpy as np
import urllib.request
import sys
import os

# =========================
# ASSETS (cố định trong repo)
# =========================
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ZENO_LOGO    = os.path.join(ASSETS_DIR, "logo_zenohomes.png")
PACIFIC_LOGO = os.path.join(ASSETS_DIR, "logo_pacific.png")
# HOTLINE_PNG không còn dùng vì hotline chỉ bị xoá, không thay mới nữa.

# =========================
# TỶ LỆ TỌA ĐỘ DỰ PHÒNG (chỉ dùng khi auto-detect thất bại hoàn toàn)
# Tính toán dựa trên ảnh gốc mẫu 905 x 1280
# =========================
LOGO_PCT_LEFT  = (24/905, 587/1280, 440/905, 680/1280)
LOGO_PCT_RIGHT = (535/905, 368/1280, 800/905, 445/1280)
BANNER_PCT     = (24/905, 1184/1280, 878/905, 1251/1280)

# Ngưỡng điểm match tối thiểu để tin vào kết quả auto-detect logo.
# Nếu điểm thấp hơn ngưỡng này, hệ thống sẽ dùng lại box phần trăm cố định làm dự phòng.
LOGO_MATCH_MIN_SCORE = 0.40

# Giới hạn kích thước/dung lượng ảnh input để tránh lỗi hoặc tốn tài nguyên runner.
MIN_IMG_DIM = 150          # px - ảnh nhỏ hơn cạnh này thì auto-detect logo/banner không đáng tin
MAX_IMG_DIM = 8000         # px - ảnh lớn hơn cạnh này sẽ được resize xuống để tránh OOM/timeout
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # 25MB - chặn tải ảnh quá nặng

# =========================
# HÀM TIỆN ÍCH DÙNG CHUNG
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
    """Chuyển đổi từ tỷ lệ % sang tọa độ pixel thực tế dựa trên kích thước ảnh."""
    x1 = int(pct_box[0] * img_w)
    y1 = int(pct_box[1] * img_h)
    x2 = int(pct_box[2] * img_w)
    y2 = int(pct_box[3] * img_h)
    return (max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2))

def is_gold_pixel(b, g, r):
    """Đặc trưng màu vàng/gold của logo Pacific Homes (BGR)."""
    return (r > 110) and (g > 80) and (b < 170) and (r >= b + 15)

# =========================
# AUTO-DETECT LOGO (không phụ thuộc tọa độ cố định)
# =========================

def _load_logo_template_mask():
    """Đọc logo_pacific.png và trả về binary mask (shape) đã crop sát viền."""
    logo = cv2.imread(PACIFIC_LOGO, cv2.IMREAD_UNCHANGED)
    if logo is None or logo.shape[2] < 4:
        return None
    alpha_full = logo[:, :, 3]
    ys, xs = np.where(alpha_full > 10)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    alpha_cropped = alpha_full[y1:y2+1, x1:x2+1]
    _, binmask = cv2.threshold(alpha_cropped, 30, 255, cv2.THRESH_BINARY)
    return binmask.astype(np.float32)

def _img_to_gold_mask_f32(img):
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    mask = (r > 110) & (g > 80) & (b < 170) & (r >= b + 15)
    return (mask.astype(np.uint8) * 255).astype(np.float32)

def detect_logo_box_auto(img, img_w, img_h):
    """
    Dò vị trí logo Pacific Homes thật trong ảnh bằng multi-scale template matching
    trên gold-mask (không phụ thuộc tọa độ % cố định, hoạt động dù logo bị lệch
    trái/phải/trên/dưới hoặc to/nhỏ khác nhau giữa các ảnh).
    Trả về (box, score). box = None nếu không tìm được mask hợp lệ.
    """
    template_mask = _load_logo_template_mask()
    if template_mask is None:
        return None, 0.0

    gold_mask = _img_to_gold_mask_f32(img)
    th0, tw0 = template_mask.shape

    # Logo Pacific Homes trong các ảnh mẫu thường chiếm khoảng 15%-55% chiều rộng ảnh.
    target_widths = np.linspace(0.12 * img_w, 0.55 * img_w, 50)
    scale_factors = target_widths / tw0

    best = None
    for scale in scale_factors:
        tw = max(8, int(tw0 * scale))
        th = max(4, int(th0 * scale))
        if tw >= img_w or th >= img_h or tw < 8 or th < 4:
            continue
        tmpl_resized = cv2.resize(template_mask, (tw, th))
        res = cv2.matchTemplate(gold_mask, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, tw, th)

    if best is None:
        return None, 0.0

    score, (x, y), tw, th = best

    # Nới rộng box thêm margin an toàn (theo % kích thước logo) để đảm bảo che kín
    # hoàn toàn logo cũ, kể cả khi template match không khớp chính xác 100% ở mép.
    margin_x = max(4, int(tw * 0.06))
    margin_y = max(4, int(th * 0.18))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w - 1, x + tw + margin_x)
    y2 = min(img_h - 1, y + th + margin_y)

    box = (x1, y1, x2, y2)
    return box, score

def detect_logo_box_fallback(img, img_w, img_h):
    """Phương án dự phòng: tọa độ % cố định + so màu nền để chọn layout trái/phải."""
    box_left = get_absolute_box(LOGO_PCT_LEFT, img_w, img_h)
    box_right = get_absolute_box(LOGO_PCT_RIGHT, img_w, img_h)

    lx1, ly1, lx2, ly2 = box_left
    rx1, ry1, rx2, ry2 = box_right

    mid_lx, mid_ly = (lx1 + lx2) // 2, (ly1 + ly2) // 2
    mid_rx, mid_ry = (rx1 + rx2) // 2, (ry1 + ry2) // 2

    bg_green = np.array([32, 64, 7])
    color_left  = img[mid_ly, mid_lx].astype(int)
    color_right = img[min(mid_ry, img_h - 1), min(mid_rx, img_w - 1)].astype(int)

    diff_left  = np.abs(color_left  - bg_green).sum()
    diff_right = np.abs(color_right - bg_green).sum()

    print(f"[Fallback] Diff to green - LEFT: {diff_left} | RIGHT: {diff_right}")
    return box_left if diff_left <= diff_right else box_right

def detect_logo_box(img, img_w, img_h):
    """
    Chiến lược: thử auto-detect bằng template matching trước (chịu được logo bị
    lệch vị trí/tỉ lệ). Nếu điểm match quá thấp (không đáng tin), rơi về phương
    án tọa độ % cố định như cũ.
    """
    auto_box, score = detect_logo_box_auto(img, img_w, img_h)
    print(f"[Auto-detect] Logo box: {auto_box} | score={score:.3f}")

    if auto_box is not None and score >= LOGO_MATCH_MIN_SCORE:
        print("=> Dùng vị trí logo AUTO-DETECT (độ tin cậy đủ cao).")
        return auto_box

    print("=> Điểm auto-detect thấp, dùng phương án dự phòng theo tỷ lệ % cố định.")
    return detect_logo_box_fallback(img, img_w, img_h)

# =========================
# AUTO-DETECT BANNER HOTLINE (không phụ thuộc tọa độ cố định)
# =========================

def detect_banner_box_auto(img, img_w, img_h):
    """
    Dò vùng banner hotline (dải gradient xanh->vàng/gold ở đáy ảnh) bằng cách quét
    từ đáy lên tại 1 cột sát mép phải, tìm dải pixel vàng liên tục.
    Trả về (x1, y1, x2, y2) hoặc None nếu không tìm thấy banner nào.
    """
    margin = max(2, int(img_w * 0.04))
    x_right_sample = img_w - margin - 5
    if x_right_sample <= 0:
        x_right_sample = img_w - 1

    search_top = int(img_h * 0.5)  # chỉ tìm trong nửa dưới ảnh
    rows = []
    for y in range(img_h - 1, search_top, -1):
        b, g, r = img[y, x_right_sample].astype(int)
        rows.append((y, is_gold_pixel(b, g, r)))

    # Bỏ qua viền nền mỏng ở đáy cùng (không vàng) để tìm điểm bắt đầu banner
    i = 0
    while i < len(rows) and not rows[i][1]:
        i += 1
    if i >= len(rows):
        return None  # Không có banner màu vàng nào ở đáy ảnh

    # Đi lên cho đến khi hết dải vàng liên tục (cho phép vài dòng nhiễu nhỏ)
    j = i
    miss = 0
    last_gold_y = rows[i][0]
    while j < len(rows):
        y, gold = rows[j]
        if gold:
            miss = 0
            last_gold_y = y
        else:
            miss += 1
            if miss > 4:
                break
        j += 1

    by1 = max(0, last_gold_y - 14)  # nới rộng lên trên 1 chút để che kín viền banner
    by2 = img_h
    bx1 = margin
    bx2 = img_w - margin
    return (bx1, by1, bx2, by2)

def detect_banner_box(img, img_w, img_h):
    """Auto-detect banner hotline; nếu thất bại, dùng tọa độ % cố định làm dự phòng."""
    auto_box = detect_banner_box_auto(img, img_w, img_h)
    if auto_box is not None:
        print(f"[Auto-detect] Banner box: {auto_box}")
        return auto_box
    print("=> Không auto-detect được banner, dùng phương án dự phòng theo tỷ lệ % cố định.")
    return get_absolute_box(BANNER_PCT, img_w, img_h)

# =========================
# HÀM CHÍNH
# =========================

def process_image(image_url: str, output_path: str):
    print(f"Đang tải ảnh: {image_url}")
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        # Đọc có giới hạn dung lượng để tránh ảnh quá nặng làm tốn RAM/timeout runner.
        raw = resp.read(MAX_DOWNLOAD_BYTES + 1)
        if len(raw) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Ảnh vượt quá giới hạn dung lượng cho phép ({MAX_DOWNLOAD_BYTES // (1024*1024)}MB)."
            )
        data = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh từ URL")

    img_h, img_w = img.shape[:2]
    print(f"Kích thước ảnh thực tế: {img_w}x{img_h}")

    # Validate kích thước tối thiểu: ảnh quá nhỏ thì auto-detect logo/banner không đáng tin cậy.
    if img_w < MIN_IMG_DIM or img_h < MIN_IMG_DIM:
        raise ValueError(
            f"Ảnh quá nhỏ ({img_w}x{img_h}). Kích thước tối thiểu yêu cầu là {MIN_IMG_DIM}x{MIN_IMG_DIM}px."
        )

    # Nếu ảnh quá lớn, resize xuống để tránh tốn RAM/CPU quá mức trên runner free tier.
    # Tỷ lệ khung hình được giữ nguyên.
    if max(img_w, img_h) > MAX_IMG_DIM:
        scale = MAX_IMG_DIM / max(img_w, img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        print(f"Ảnh quá lớn, resize {img_w}x{img_h} -> {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_h, img_w = img.shape[:2]

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
        sample_points.append([32, 64, 7])  # Fallback màu mặc định nếu không lấy được mẫu

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

    # --- 2) Xoá hotline cũ bằng gradient (không thay hotline mới) ---
    bx1, by1, bx2, by2 = detect_banner_box(img, img_w, img_h)
    print(f"Banner: ({bx1},{by1}) -> ({bx2},{by2})")
    banner_w, banner_h = bx2 - bx1, by2 - by1

    if banner_w > 0 and banner_h > 0:
        mid_y = (by1 + by2) // 2
        sample_x_left = min(bx1 + 5, img_w - 1)
        sample_x_right = max(bx2 - 5, 0)

        color_left_real  = img[mid_y, sample_x_left].astype(np.float64)
        color_right_real = img[mid_y, sample_x_right].astype(np.float64)

        # Vẽ lại gradient nền (giống màu banner gốc) để xoá sạch chữ hotline cũ,
        # KHÔNG dán chữ hotline mới lên trên nữa.
        gradient = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
        t = np.linspace(0, 1, banner_w).reshape(1, -1, 1)
        gradient[:] = (color_left_real * (1 - t) + color_right_real * t).astype(np.uint8)
        img[by1:by2, bx1:bx2] = gradient
        print("Đã xoá hotline cũ (không dán hotline mới).")
    else:
        print("Bỏ qua bước xoá hotline (không xác định được vùng banner hợp lệ).")

    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Đã lưu thành công: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process.py <image_url> <output_path>")
        sys.exit(1)
    process_image(sys.argv[1], sys.argv[2])
