import cv2
import numpy as np
import urllib.request
import sys
import os
import glob

# =========================
# ASSETS (cố định trong repo)
# =========================
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

HOTLINE_PNG = os.path.join(ASSETS_DIR, "hotline.png")

# Thư viện logo: mỗi mẫu logo "nguồn" (của các sàn/CTV khác) đặt trong logo_loc/
# để hệ thống tự so khớp xem ảnh input đang dùng logo nào -> che đi và dán logo
# thay thế lấy từ logo_thay_the/ (thường chỉ có 1 file: logo Zeno Homes).
LOGO_LOC_DIR       = os.path.join(ASSETS_DIR, "logo", "logo_loc")
LOGO_THAY_THE_DIR  = os.path.join(ASSETS_DIR, "logo", "logo_thay_the")

# Thư viện QR: tương tự logo — qr_loc/ chứa các mẫu khung QR đã gặp trước đây
# (dùng làm phương án dự phòng), qr_thay_the/ chứa QR/Zalo thật của Zeno Homes.
QR_LOC_DIR         = os.path.join(ASSETS_DIR, "qr", "qr_loc")
QR_THAY_THE_DIR    = os.path.join(ASSETS_DIR, "qr", "qr_thay_the")

# =========================
# TỶ LỆ TỌA ĐỘ DỰ PHÒNG (chỉ dùng khi auto-detect thất bại hoàn toàn,
# tham chiếu theo layout ảnh Pacific Homes gốc)
# =========================
LOGO_PCT_LEFT  = (24/905, 587/1280, 440/905, 680/1280)
LOGO_PCT_RIGHT = (540/905, 420/1280, 775/905, 482/1280)
BANNER_PCT = (24/905, 1184/1280, 887/905, 1251/1280)

# Chỉ những layout dưới đây (theo tên file trong logo_loc/) mới có banner hotline
# đúng vị trí BANNER_PCT. Layout khác (như Hừng Đông) có bố cục khác hẳn -> KHÔNG
# được đè banner hotline lên, kẻo ghi đè nhầm vào nội dung khác (vd banner "ZENO
# HOMES" ở cuối ảnh). Khi thêm 1 layout mới có banner hotline riêng, thêm tên file
# logo_loc tương ứng vào đây và (nếu vị trí khác) tạo thêm 1 bộ toạ độ % riêng.
HOTLINE_BANNER_TEMPLATES = {"logo_pacific.png"}

# Ngưỡng điểm match tối thiểu để tin vào kết quả auto-detect logo / QR.
LOGO_MATCH_MIN_SCORE = 0.40
QR_MATCH_MIN_SCORE = 0.35

# QRCodeDetector chỉ định vị đúng vùng mã QR (các ô vuông đen/trắng), KHÔNG
# tính viền trắng (quiet zone) quanh QR. Cần "nới" thêm ra để che trọn khung QR
# gốc (viền trắng + viền màu nếu có). Đo trên mẫu Hừng Đông thực tế: vùng lõi
# QR ~79x74px, khung QR đầy đủ ~125x127px -> hệ số nới ~0.30 mỗi bên là khớp.
# Nếu khung QR của nguồn khác có thêm chữ/logo RIÊNG nằm dưới QR (khác với icon
# nằm LỒNG bên trong QR như mẫu Zalo), tăng QR_PAD_BOTTOM_PCT lên cho nguồn đó.
QR_PAD_SIDE_PCT   = 0.30   # nới thêm 2 bên trái/phải (% chiều rộng vùng QR lõi)
QR_PAD_TOP_PCT    = 0.30   # nới thêm phía trên
QR_PAD_BOTTOM_PCT = 0.30   # nới thêm phía dưới

# =========================
# HÀM TIỆN ÍCH
# =========================

def load_image_normalized(path):
    """Đọc ảnh và luôn trả về ít nhất 3 kênh màu (BGR hoặc BGRA), kể cả khi
    file gốc là ảnh xám (ví dụ QR tạo bằng thư viện qrcode)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

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
    return (max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2))

def list_asset_files(folder):
    """Trả về danh sách file ảnh trong 1 thư mục thư viện (logo_loc, qr_loc, ...),
    sắp xếp theo tên để kết quả ổn định."""
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

def pick_replacement_asset(folder):
    """Chọn ảnh thay thế trong logo_thay_the/ hoặc qr_thay_the/.
    Ưu tiên file tên 'default.*', nếu không có thì lấy file đầu tiên (theo tên)."""
    files = list_asset_files(folder)
    if not files:
        return None
    for f in files:
        if os.path.splitext(os.path.basename(f))[0].lower() == "default":
            return f
    return files[0]

def sample_bg_color(img, x1, y1, x2, y2, img_w, img_h):
    """Lấy màu nền thực tế từ các pixel sát viền vùng cần che."""
    w, h = x2 - x1, y2 - y1
    sample_points = []
    if x1 > 5:
        for dy in range(0, h, max(1, h // 8)):
            sample_points.append(img[y1 + dy, x1 - 5])
    if x2 < img_w - 5:
        for dy in range(0, h, max(1, h // 8)):
            sample_points.append(img[y1 + dy, x2 + 5])
    if y1 > 5:
        for dx in range(0, w, max(1, w // 8)):
            sample_points.append(img[y1 - 5, x1 + dx])
    if y2 < img_h - 5:
        for dx in range(0, w, max(1, w // 8)):
            sample_points.append(img[min(y2 + 5, img_h - 1), x1 + dx])
    if len(sample_points) == 0:
        sample_points.append([32, 64, 7])  # fallback màu mặc định
    return np.median(sample_points, axis=0).astype(int).tolist()

def fill_area_gradient(img, x1, y1, x2, y2, feather=25):
    """Che 1 vùng bằng CHÍNH màu nền xung quanh, tự nhiên có dải màu chuyển
    (gradient) thay vì 1 màu phẳng — dùng content-aware inpaint (OpenCV Telea)
    dựng lại vùng đó dựa trên viền màu/độ sáng xung quanh nó. Hợp với các nền
    dạng gradient (xanh đậm -> nhạt) phổ biến ở các mẫu bất động sản.
    LƯU Ý: lấp NGUYÊN CẢ vùng hình chữ nhật — dùng khi biết chắc cả vùng đó
    cần thay (vd toàn bộ hộp che trống). Với logo cũ, nên dùng
    fill_logo_area_gradient() bên dưới để tránh streak (xem giải thích ở đó)."""
    h, w = img.shape[:2]
    fx1 = max(0, x1 - feather)
    fy1 = max(0, y1 - feather)
    fx2 = min(w, x2 + feather)
    fy2 = min(h, y2 + feather)

    mask = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.uint8)
    mx1, my1 = x1 - fx1, y1 - fy1
    mx2, my2 = x2 - fx1, y2 - fy1
    cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

    region = img[fy1:fy2, fx1:fx2]
    filled = cv2.inpaint(region, mask, 7, cv2.INPAINT_TELEA)
    img[y1:y2, x1:x2] = filled[my1:my2, mx1:mx2]

def fill_logo_area_gradient(img, x1, y1, x2, y2, context=20, dilate=7):
    """Che LOGO CŨ bằng chính nền xung quanh — CHỈ tái tạo (inpaint) đúng
    những pixel thực sự thuộc logo cũ (theo mặt nạ màu vàng gold), KHÔNG lấp
    cả khối hình chữ nhật bao quanh nó.

    Lý do: inpaint trên 1 vùng chữ nhật lớn (vd 160x200px) thường bị lỗi
    "đường nối dọc" (streak) ở giữa — nơi các hướng lấp từ nhiều phía của
    thuật toán Telea gặp nhau — vì thiếu ngữ cảnh màu ở giữa vùng quá lớn.
    Logo cũ chỉ chiếm 1 phần nhỏ/mảnh trong hộp đó (phần còn lại vốn đã là
    nền đúng rồi, không cần đụng vào) nên inpaint đúng phần đó cho mượt và an
    toàn hơn nhiều so với lấp cả hộp.

    QUAN TRỌNG: mặt nạ màu vàng chỉ được tính trong PHẠM VI HỘP LOGO GỐC
    (x1,y1,x2,y2), KHÔNG tính trên cả vùng context mở rộng — vì nếu ngay
    sát hộp logo có viền vàng của 1 khối khác (vd hộp thông tin DT đất...),
    tính gold-mask trên cả vùng context sẽ nhận nhầm luôn viền đó là "logo
    cũ" và xoá/tái tạo mất viền của khối bên cạnh. Vùng context mở rộng chỉ
    dùng làm NGỮ CẢNH màu để inpaint tham chiếu, không bị coi là cần lấp."""
    h, w = img.shape[:2]
    fx1 = max(0, x1 - context)
    fy1 = max(0, y1 - context)
    fx2 = min(w, x2 + context)
    fy2 = min(h, y2 + context)

    region = img[fy1:fy2, fx1:fx2]

    # Tính gold-mask CHỈ trong đúng hộp logo gốc, rồi đặt vào đúng vị trí
    # trong mask kích thước bằng region (context) -> phần context xung quanh
    # luôn là 0 (không bị lấp), chỉ dùng làm nguồn tham chiếu màu.
    box_only = img[y1:y2, x1:x2]
    box_gold = _img_to_gold_mask_f32(box_only).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        box_gold = cv2.dilate(box_gold, kernel)

    gold = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.uint8)
    by1, bx1 = y1 - fy1, x1 - fx1
    by2, bx2 = by1 + (y2 - y1), bx1 + (x2 - x1)
    gold[by1:by2, bx1:bx2] = box_gold

    filled = cv2.inpaint(region, gold, 7, cv2.INPAINT_TELEA)
    img[fy1:fy2, fx1:fx2] = filled

# =========================
# LOGIC AUTO-DETECT LOGO (SO KHỚP VỚI THƯ VIỆN logo_loc/)
# =========================

def _img_to_gold_mask_f32(img):
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    mask = (r > 110) & (g > 80) & (b < 170) & (r >= b + 15)
    return (mask.astype(np.uint8) * 255).astype(np.float32)

def _template_to_gold_mask(path):
    """Đọc 1 file mẫu logo trong logo_loc/ và trả về mask nhị phân đã crop sát nội dung.
    Hỗ trợ cả ảnh có nền trong suốt (dùng alpha) lẫn ảnh cắt trực tiếp từ ảnh gốc
    (không có alpha -> tự nhận diện theo màu vàng gold)."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    has_real_alpha = raw.ndim == 3 and raw.shape[2] == 4 and raw[:, :, 3].min() < 250

    if has_real_alpha:
        alpha_full = raw[:, :, 3]
        ys, xs = np.where(alpha_full > 10)
        if len(xs) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        alpha_cropped = alpha_full[y1:y2+1, x1:x2+1]
        _, binmask = cv2.threshold(alpha_cropped, 30, 255, cv2.THRESH_BINARY)
        return binmask.astype(np.float32)

    bgr = raw[:, :, :3] if raw.ndim == 3 and raw.shape[2] >= 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    gold = _img_to_gold_mask_f32(bgr)
    ys, xs = np.where(gold > 0)
    if len(xs) == 0:
        return None
    margin = 3
    x1 = max(0, xs.min() - margin)
    x2 = min(gold.shape[1], xs.max() + margin)
    y1 = max(0, ys.min() - margin)
    y2 = min(gold.shape[0], ys.max() + margin)
    return gold[y1:y2, x1:x2]

def _match_template_multiscale(target_mask, template_mask, img_w, img_h,
                                min_width_pct=0.10, max_width_pct=0.55, steps=40):
    th0, tw0 = template_mask.shape
    target_widths = np.linspace(min_width_pct * img_w, max_width_pct * img_w, steps)
    scale_factors = target_widths / tw0

    best = None
    for scale in scale_factors:
        tw = max(8, int(tw0 * scale))
        th = max(4, int(th0 * scale))
        if tw >= img_w or th >= img_h or tw < 8 or th < 4:
            continue
        tmpl_resized = cv2.resize(template_mask, (tw, th))
        res = cv2.matchTemplate(target_mask, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, tw, th)
    return best

def detect_logo_box_auto(img, img_w, img_h):
    """So khớp ảnh input với TỪNG mẫu logo trong logo_loc/, chọn ra kết quả
    (mẫu + vị trí) có điểm số cao nhất."""
    templates = list_asset_files(LOGO_LOC_DIR)
    if not templates:
        return None, 0.0, None

    gold_mask = _img_to_gold_mask_f32(img)

    best_overall = None  # (score, box, template_name)
    for tpl_path in templates:
        template_mask = _template_to_gold_mask(tpl_path)
        if template_mask is None:
            continue
        result = _match_template_multiscale(gold_mask, template_mask, img_w, img_h)
        if result is None:
            continue
        score, (x, y), tw, th = result
        margin_x = max(4, int(tw * 0.06))
        margin_y = max(4, int(th * 0.18))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w - 1, x + tw + margin_x)
        y2 = min(img_h - 1, y + th + margin_y)
        print(f"  [logo_loc] {os.path.basename(tpl_path)} -> score={score:.3f} box=({x1},{y1},{x2},{y2})")
        if best_overall is None or score > best_overall[0]:
            best_overall = (score, (x1, y1, x2, y2), os.path.basename(tpl_path))

    if best_overall is None:
        return None, 0.0, None
    score, box, name = best_overall
    return box, score, name

def detect_logo_box_fallback(img, img_w, img_h):
    """Phương án dự phòng cuối cùng nếu KHÔNG mẫu nào trong logo_loc/ khớp
    (theo layout ảnh Pacific Homes gốc)."""
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
    auto_box, score, tpl_name = detect_logo_box_auto(img, img_w, img_h)
    print(f"[Auto-detect] Logo khớp nhất: {tpl_name} | box={auto_box} | score={score:.3f}")

    if auto_box is not None and score >= LOGO_MATCH_MIN_SCORE:
        print(f"=> Dùng vị trí logo AUTO-DETECT (khớp với '{tpl_name}' trong logo_loc/).")
        return auto_box, tpl_name

    print("=> Không mẫu nào trong logo_loc/ đạt ngưỡng, dùng phương án dự phòng theo layout gốc.")
    return detect_logo_box_fallback(img, img_w, img_h), None

# =========================
# LOGIC AUTO-DETECT QR (SO KHỚP VỚI THƯ VIỆN qr_loc/ + cv2 QRCodeDetector)
# =========================

def _pad_box(x1, y1, x2, y2, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * QR_PAD_SIDE_PCT))
    x2 = min(img_w - 1, int(x2 + w * QR_PAD_SIDE_PCT))
    y1 = max(0, int(y1 - h * QR_PAD_TOP_PCT))
    y2 = min(img_h - 1, int(y2 + h * QR_PAD_BOTTOM_PCT))
    return (x1, y1, x2, y2)

def _find_enclosing_border_panel(img, cx, cy):
    """QR thường được đặt lồng bên trong 1 khung/panel có viền màu nổi bật (vàng/
    gold ở mẫu Hừng Đông). Nếu nới viền QR quá tay, ảnh dán ra có thể tràn ra
    NGOÀI panel đó (đè lên viền vàng hoặc ra hẳn nền ngoài panel). Hàm này tìm
    panel viền màu bao quanh điểm (cx,cy) để dùng làm giới hạn không được vượt
    qua khi dán QR thay thế."""
    h, w = img.shape[:2]
    b, g, r = cv2.split(img.astype(int))
    # Viền vàng/gold: R,G cao, B thấp hẳn (áp dụng cho mẫu Hừng Đông; có thể cần
    # nới thêm điều kiện màu nếu gặp panel viền màu khác trong tương lai).
    border_mask = (((r > 180) & (g > 150) & (b < 200) &
                     ((r - b) > 40) & ((g - b) > 20))).astype(np.uint8) * 255
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 5000:
            continue
        # panel phải chứa điểm trung tâm QR bên trong nó
        if x <= cx <= x + cw and y <= cy <= y + ch:
            if best is None or area < best[2] * best[3]:  # panel NHỎ NHẤT chứa QR (bám sát nhất)
                best = (x, y, cw, ch)

    if best is None:
        return None
    x, y, cw, ch = best
    return (x, y, x + cw, y + ch)

def _clamp_box_to_panel(box, img, margin=3):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    panel = _find_enclosing_border_panel(img, cx, cy)
    if panel is None:
        return box
    px1, py1, px2, py2 = panel
    nx1 = max(x1, px1 + margin)
    ny1 = max(y1, py1 + margin)
    nx2 = min(x2, px2 - margin)
    ny2 = min(y2, py2 - margin)
    if nx2 - nx1 < 10 or ny2 - ny1 < 10:
        # panel phát hiện có vẻ không hợp lệ (quá nhỏ so với QR) -> bỏ qua, giữ box gốc
        return box
    if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
        print(f"  [QR] Panel viền màu bao quanh: {panel} -> giới hạn lại box QR "
              f"về ({nx1},{ny1},{nx2},{ny2}) để không tràn ra ngoài viền.")
    return (nx1, ny1, nx2, ny2)

def detect_qr_box_native(img):
    """Dùng bộ dò QR chuẩn của OpenCV — định vị được BẤT KỲ QR nào bất kể
    nội dung/mã hoá bên trong (không cần giải mã được), không cần thư viện mẫu.
    detect() định vị đáng tin cậy hơn detectMulti() với QR bị nén/mờ nhẹ."""
    img_h, img_w = img.shape[:2]
    try:
        detector = cv2.QRCodeDetector()
        boxes = []

        ok, points = detector.detect(img)
        if ok and points is not None and len(points) > 0:
            quad = points[0]
            xs, ys = quad[:, 0], quad[:, 1]
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

        ok_multi, points_multi = detector.detectMulti(img)
        if ok_multi and points_multi is not None:
            for quad in points_multi:
                xs, ys = quad[:, 0], quad[:, 1]
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    except Exception as e:
        print(f"[QR-native] Lỗi: {e}")
        return None

    if not boxes:
        return None

    # Lấy QR có diện tích lõi lớn nhất (thường là QR chính, không phải logo nhỏ khác)
    best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x1, y1, x2, y2 = best_box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)
    padded = _pad_box(x1, y1, x2, y2, img_w, img_h)
    return _clamp_box_to_panel(padded, img)

def detect_qr_box_template(img, img_w, img_h):
    """Dự phòng: so khớp khung/thiết kế QR với thư viện qr_loc/ (dùng khi
    bộ dò QR chuẩn không nhận ra, ví dụ QR bị đè hiệu ứng đồ hoạ)."""
    templates = list_asset_files(QR_LOC_DIR)
    if not templates:
        return None, 0.0, None

    gray_target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    best_overall = None
    for tpl_path in templates:
        tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        tpl = tpl.astype(np.float32)
        result = _match_template_multiscale(gray_target, tpl, img_w, img_h,
                                             min_width_pct=0.06, max_width_pct=0.30, steps=30)
        if result is None:
            continue
        score, (x, y), tw, th = result
        box = (x, y, min(img_w - 1, x + tw), min(img_h - 1, y + th))
        print(f"  [qr_loc] {os.path.basename(tpl_path)} -> score={score:.3f} box={box}")
        if best_overall is None or score > best_overall[0]:
            best_overall = (score, box, os.path.basename(tpl_path))

    if best_overall is None:
        return None, 0.0, None
    score, box, name = best_overall
    return box, score, name

def detect_qr_box(img, img_w, img_h):
    native_box = detect_qr_box_native(img)
    if native_box is not None:
        print(f"[QR] Phát hiện bằng bộ dò QR chuẩn (đã nới viền): {native_box}")
        return native_box

    print("[QR] Bộ dò chuẩn không thấy QR, thử so khớp với thư viện qr_loc/...")
    box, score, name = detect_qr_box_template(img, img_w, img_h)
    if box is not None and score >= QR_MATCH_MIN_SCORE:
        x1, y1, x2, y2 = _pad_box(*box, img_w, img_h)
        x1, y1, x2, y2 = _clamp_box_to_panel((x1, y1, x2, y2), img)
        print(f"=> Dùng vị trí QR khớp với '{name}' trong qr_loc/ (score={score:.3f}), box=({x1},{y1},{x2},{y2}).")
        return (x1, y1, x2, y2)

    print("[QR] Không tìm thấy QR nào trong ảnh -> bỏ qua bước thay QR.")
    return None

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

    # --- 1) Thay logo (so khớp với thư viện logo_loc/, dán ảnh trong logo_thay_the/) ---
    logo_box, matched_tpl_name = detect_logo_box(img, img_w, img_h)
    lx1, ly1, lx2, ly2 = logo_box
    logo_w, logo_h = lx2 - lx1, ly2 - ly1

    print(f"Che nền vùng logo bằng dải màu chuyển dựa theo viền xung quanh...")
    fill_logo_area_gradient(img, lx1, ly1, lx2, ly2)

    zeno_logo_path = pick_replacement_asset(LOGO_THAY_THE_DIR)
    if zeno_logo_path is None:
        raise FileNotFoundError(f"Không có file nào trong {LOGO_THAY_THE_DIR}")
    zeno = load_image_normalized(zeno_logo_path)
    if zeno is None:
        raise FileNotFoundError(f"Không đọc được {zeno_logo_path}")
    zeno = crop_to_content(zeno)
    zeno_fit = fit_resize(zeno, int(logo_w * 0.85), int(logo_h * 0.80))
    zh, zw = zeno_fit.shape[:2]
    zx = lx1 + (logo_w - zw) // 2
    zy = ly1 + (logo_h - zh) // 2
    paste_rgba(img, zeno_fit, zx, zy)
    print(f"Đã dán logo thay thế ({os.path.basename(zeno_logo_path)}): {zw}x{zh} tại ({zx},{zy})")

    # --- 2) Thay QR (so khớp/định vị QR, dán ảnh trong qr_thay_the/) ---
    qr_box = detect_qr_box(img, img_w, img_h)
    if qr_box is not None:
        qx1, qy1, qx2, qy2 = qr_box
        qr_w, qr_h = qx2 - qx1, qy2 - qy1

        qr_replacement_path = pick_replacement_asset(QR_THAY_THE_DIR)
        if qr_replacement_path is None:
            print(f"[QR] Không có file thay thế trong {QR_THAY_THE_DIR} -> bỏ qua bước thay QR.")
        else:
            qr_bg = sample_bg_color(img, qx1, qy1, qx2, qy2, img_w, img_h)
            cv2.rectangle(img, (qx1, qy1), (qx2, qy2), qr_bg, -1)

            new_qr = load_image_normalized(qr_replacement_path)
            if new_qr is None:
                raise FileNotFoundError(f"Không đọc được {qr_replacement_path}")
            new_qr = crop_to_content(new_qr)
            new_qr_fit = fit_resize(new_qr, qr_w, qr_h)
            nqh, nqw = new_qr_fit.shape[:2]
            nqx = qx1 + (qr_w - nqw) // 2
            nqy = qy1 + (qr_h - nqh) // 2
            paste_rgba(img, new_qr_fit, nqx, nqy)
            print(f"Đã dán QR thay thế ({os.path.basename(qr_replacement_path)}): {nqw}x{nqh} tại ({nqx},{nqy})")

    # --- 3) Thay hotline (chỉ áp dụng cho layout đã biết có banner hotline
    #         ở đúng vị trí BANNER_PCT — tránh đè nhầm lên layout khác) ---
    apply_hotline = (matched_tpl_name is None) or (matched_tpl_name in HOTLINE_BANNER_TEMPLATES)
    if not apply_hotline:
        print(f"[Hotline] Layout '{matched_tpl_name}' chưa khai báo vị trí banner hotline "
              f"-> bỏ qua bước này (không đụng vào ảnh gốc ở khu vực đó).")
    else:
        bx1, by1, bx2, by2 = get_absolute_box(BANNER_PCT, img_w, img_h)
        print(f"Banner (tính theo tỷ lệ): ({bx1},{by1}) -> ({bx2},{by2})")

        banner_w = bx2 - bx1
        banner_h = by2 - by1

        hotline = load_image_normalized(HOTLINE_PNG)
        if hotline is None:
            raise FileNotFoundError(f"Không đọc được {HOTLINE_PNG}")

        # crop sát nội dung
        hotline = crop_to_content(
            hotline,
            alpha_thresh=1,
            margin=0
        )

        # kéo full chiều ngang banner
        hotline_fit = cv2.resize(
            hotline,
            (banner_w, banner_h),
            interpolation=cv2.INTER_AREA
        )

        # giữ nguyên nền banner gốc, chỉ đè hotline lên
        paste_rgba(
            img,
            hotline_fit,
            bx1,
            by1
        )
        print(f"Đã dán hotline full banner: {banner_w}x{banner_h}")

    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Đã lưu thành công: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process.py <image_url> <output_path>")
        sys.exit(1)
    process_image(sys.argv[1], sys.argv[2])
