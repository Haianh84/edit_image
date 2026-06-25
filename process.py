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
PACIFIC_LOGO = os.path.join(ASSETS_DIR, "logo_pacific.png")

# =========================
# TỶ LỆ TỌA ĐỘ CHUẨN
# =========================
LOGO_PCT_LEFT  = (24/905, 587/1280, 440/905, 680/1280)
LOGO_PCT_RIGHT = (540/905, 420/1280, 775/905, 482/1280)

# Banner kéo full chiều ngang
BANNER_PCT = (
    0,
    1184/1280,
    1,
    1251/1280
)

LOGO_MATCH_MIN_SCORE = 0.40


# =========================
# HÀM TIỆN ÍCH
# =========================

def paste_rgba(base_img, overlay_rgba, x, y):
    """
    Dán PNG RGBA lên ảnh nền.
    Tự xử lý khi ảnh bị tràn mép.
    """

    bh, bw = base_img.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    if x >= bw or y >= bh:
        return

    if x < 0:
        overlay_rgba = overlay_rgba[:, -x:]
        ow = overlay_rgba.shape[1]
        x = 0

    if y < 0:
        overlay_rgba = overlay_rgba[-y:, :]
        oh = overlay_rgba.shape[0]
        y = 0

    w = min(ow, bw - x)
    h = min(oh, bh - y)

    if w <= 0 or h <= 0:
        return

    overlay_rgba = overlay_rgba[:h, :w]

    roi = base_img[y:y+h, x:x+w]

    if overlay_rgba.shape[2] == 4:

        alpha = overlay_rgba[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]

        roi[:] = (
            overlay_rgba[:, :, :3].astype(np.float32) * alpha +
            roi.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

    else:
        roi[:] = overlay_rgba[:, :, :3]


def fit_resize(img, max_w, max_h):

    h, w = img.shape[:2]

    scale = min(max_w / w, max_h / h)

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    return cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


def crop_to_content(overlay_rgba, alpha_thresh=1):

    if overlay_rgba.shape[2] < 4:
        return overlay_rgba

    alpha = overlay_rgba[:, :, 3]

    ys, xs = np.where(alpha > alpha_thresh)

    if len(xs) == 0:
        return overlay_rgba

    x1 = xs.min()
    x2 = xs.max() + 1

    y1 = ys.min()
    y2 = ys.max() + 1

    return overlay_rgba[y1:y2, x1:x2]


def get_absolute_box(pct_box, img_w, img_h):

    x1 = int(round(pct_box[0] * img_w))
    y1 = int(round(pct_box[1] * img_h))
    x2 = int(round(pct_box[2] * img_w))
    y2 = int(round(pct_box[3] * img_h))

    return (
        max(0, x1),
        max(0, y1),
        min(img_w, x2),
        min(img_h, y2)
    )
# =========================
# LOGIC AUTO-DETECT LOGO
# =========================

def _load_logo_template_mask():
    logo = cv2.imread(PACIFIC_LOGO, cv2.IMREAD_UNCHANGED)

    if logo is None:
        return None

    if logo.shape[2] < 4:
        return None

    alpha = logo[:, :, 3]

    ys, xs = np.where(alpha > 10)

    if len(xs) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    alpha = alpha[y1:y2+1, x1:x2+1]

    _, alpha = cv2.threshold(
        alpha,
        30,
        255,
        cv2.THRESH_BINARY
    )

    return alpha.astype(np.float32)


def _img_to_gold_mask_f32(img):

    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)

    mask = (
        (r > 110) &
        (g > 80) &
        (b < 170) &
        (r >= b + 15)
    )

    return (mask.astype(np.uint8) * 255).astype(np.float32)


def detect_logo_box_auto(img, img_w, img_h):

    template = _load_logo_template_mask()

    if template is None:
        return None, 0

    gold = _img_to_gold_mask_f32(img)

    th0, tw0 = template.shape

    target_widths = np.linspace(
        0.12 * img_w,
        0.55 * img_w,
        50
    )

    best_score = -1
    best_box = None

    for target_w in target_widths:

        scale = target_w / tw0

        tw = max(8, int(tw0 * scale))
        th = max(4, int(th0 * scale))

        if tw >= img_w or th >= img_h:
            continue

        tmpl = cv2.resize(
            template,
            (tw, th),
            interpolation=cv2.INTER_AREA
        )

        res = cv2.matchTemplate(
            gold,
            tmpl,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, loc = cv2.minMaxLoc(res)

        if score > best_score:

            best_score = score

            x, y = loc

            margin_x = max(4, int(tw * 0.06))
            margin_y = max(4, int(th * 0.18))

            best_box = (
                max(0, x - margin_x),
                max(0, y - margin_y),
                min(img_w, x + tw + margin_x),
                min(img_h, y + th + margin_y)
            )

    return best_box, best_score


def detect_logo_box_fallback(img, img_w, img_h):

    left = get_absolute_box(
        LOGO_PCT_LEFT,
        img_w,
        img_h
    )

    right = get_absolute_box(
        LOGO_PCT_RIGHT,
        img_w,
        img_h
    )

    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right

    mid_lx = (lx1 + lx2) // 2
    mid_ly = (ly1 + ly2) // 2

    mid_rx = (rx1 + rx2) // 2
    mid_ry = (ry1 + ry2) // 2

    bg_green = np.array([32, 64, 7])

    left_color = img[mid_ly, mid_lx].astype(int)
    right_color = img[mid_ry, mid_rx].astype(int)

    diff_left = np.abs(left_color - bg_green).sum()
    diff_right = np.abs(right_color - bg_green).sum()

    print(
        f"[Fallback] LEFT={diff_left} RIGHT={diff_right}"
    )

    if diff_left <= diff_right:
        return left

    return right


def detect_logo_box(img, img_w, img_h):

    auto_box, score = detect_logo_box_auto(
        img,
        img_w,
        img_h
    )

    print(
        f"[Auto-detect] Logo box: {auto_box} | score={score:.3f}"
    )

    if auto_box is not None and score >= LOGO_MATCH_MIN_SCORE:

        print("=> Dùng AUTO DETECT.")

        return auto_box

    print("=> Dùng FALLBACK.")

    return detect_logo_box_fallback(
        img,
        img_w,
        img_h
    )

# =========================
# HÀM CHÍNH
# =========================

def process_image(image_url: str, output_path: str):

    print(f"Đang tải ảnh: {image_url}")

    req = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Không decode được ảnh")

    img_h, img_w = img.shape[:2]

    print(f"Kích thước ảnh thực tế: {img_w}x{img_h}")

    # =====================================================
    # THAY LOGO
    # =====================================================

    logo_box = detect_logo_box(
        img,
        img_w,
        img_h
    )

    lx1, ly1, lx2, ly2 = logo_box

    logo_w = lx2 - lx1
    logo_h = ly2 - ly1

    sample_points = []

    if lx1 > 5:
        for dy in range(
            0,
            logo_h,
            max(1, logo_h // 8)
        ):
            sample_points.append(
                img[ly1 + dy, lx1 - 5]
            )

    if lx2 < img_w - 5:
        for dy in range(
            0,
            logo_h,
            max(1, logo_h // 8)
        ):
            sample_points.append(
                img[ly1 + dy, lx2 + 5]
            )

    if ly1 > 5:
        for dx in range(
            0,
            logo_w,
            max(1, logo_w // 8)
        ):
            sample_points.append(
                img[ly1 - 5, lx1 + dx]
            )

    if len(sample_points) == 0:
        sample_points.append([32, 64, 7])

    bg_color = np.median(
        sample_points,
        axis=0
    ).astype(int).tolist()

    print(f"Màu nền: {bg_color}")

    cv2.rectangle(
        img,
        (lx1, ly1),
        (lx2, ly2),
        bg_color,
        -1
    )

    zeno = cv2.imread(
        ZENO_LOGO,
        cv2.IMREAD_UNCHANGED
    )

    if zeno is None:
        raise FileNotFoundError(ZENO_LOGO)

    zeno = crop_to_content(zeno)

    zeno = fit_resize(
        zeno,
        int(logo_w * 0.85),
        int(logo_h * 0.80)
    )

    zh, zw = zeno.shape[:2]

    zx = lx1 + (logo_w - zw) // 2
    zy = ly1 + (logo_h - zh) // 2

    paste_rgba(
        img,
        zeno,
        zx,
        zy
    )

    print(
        f"Đã dán logo {zw}x{zh}"
    )

    # =====================================================
    # HOTLINE FULL CHIỀU NGANG
    # =====================================================

    _, by1, _, by2 = get_absolute_box(
        BANNER_PCT,
        img_w,
        img_h
    )

    banner_h = by2 - by1

    hotline = cv2.imread(
        HOTLINE_PNG,
        cv2.IMREAD_UNCHANGED
    )

    if hotline is None:
        raise FileNotFoundError(HOTLINE_PNG)

    hotline = crop_to_content(
        hotline,
        alpha_thresh=1
    )

    # resize full chiều ngang ảnh
    hotline = cv2.resize(
        hotline,
        (img_w, banner_h),
        interpolation=cv2.INTER_AREA
    )

    paste_rgba(
        img,
        hotline,
        0,
        by1
    )

    print(
        f"Đã dán hotline full width {img_w}x{banner_h}"
    )

    cv2.imwrite(
        output_path,
        img,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            95
        ]
    )

    print(
        f"Đã lưu: {output_path}"
    )
# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage:")
        print("python process.py <image_url> <output_path>")
        sys.exit(1)

    image_url = sys.argv[1]
    output_path = sys.argv[2]

    try:

        process_image(
            image_url,
            output_path
        )

        print("=" * 50)
        print("HOÀN THÀNH")
        print("=" * 50)

    except Exception as e:

        print("=" * 50)
        print("LỖI:")
        print(e)
        print("=" * 50)

        raise