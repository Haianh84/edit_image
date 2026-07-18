# Zeno Image Processor

Tự động dò và thay **logo** + **QR** của sàn/CTV khác thành thương hiệu Zeno Homes
trên ảnh bất động sản, dựa trên **thư viện mẫu** (so khớp để biết ảnh đang dùng
logo/QR nào rồi mới che + dán đè). Chạy 100% trên **GitHub Actions**, miễn phí,
không cần server.

---

## Cấu trúc repo

```
├── process.py                        # Script xử lý ảnh
├── debug_test_local.py               # Test nhanh 1 ảnh trên máy local (không cần GitHub Actions)
├── assets/
│   ├── hotline.png                   # Banner hotline mới (chỉ dùng cho layout Pacific, xem bên dưới)
│   ├── logo/
│   │   ├── logo_loc/                 # THƯ VIỆN logo "nguồn" cần dò tìm & che đi
│   │   │   ├── logo_pacific.png      #   ví dụ: logo Pacific Homes
│   │   │   └── logo_hungdong.png     #   ví dụ: logo Hừng Đông
│   │   └── logo_thay_the/            # Logo THAY THẾ sẽ dán đè lên (thường chỉ 1 file)
│   │       └── logo_zenohomes.png
│   └── qr/
│       ├── qr_loc/                   # THƯ VIỆN khung QR "nguồn" từng gặp (dùng làm dự phòng)
│       │   └── qr_hungdong_zalo.png
│       └── qr_thay_the/              # QR/Zalo THẬT của Zeno Homes sẽ dán đè lên
│           └── default.png           # ⚠️ hiện là QR PLACEHOLDER — xem mục "Việc cần làm"
└── .github/workflows/
    ├── process-image.yml             # Workflow xử lý 1 ảnh
    └── cleanup-releases.yml          # Dọn release cũ
```

---

## Cơ chế hoạt động (logo & QR)

### Logo
1. Script lần lượt so khớp (template matching theo màu vàng gold) ảnh input với
   **từng file** trong `logo_loc/`.
2. Mẫu nào cho điểm khớp cao nhất **và** vượt ngưỡng (`LOGO_MATCH_MIN_SCORE`,
   mặc định 0.40) sẽ được coi là logo đang xuất hiện trong ảnh -> tô màu nền che
   vùng đó -> dán logo trong `logo_thay_the/` vào đúng vị trí, đúng tỉ lệ.
3. Nếu không mẫu nào đạt ngưỡng, dùng phương án dự phòng theo toạ độ % cố định
   (chỉ đúng với layout Pacific Homes gốc).

**Muốn thêm 1 logo "nguồn" mới (sàn/CTV khác)?**
Chỉ cần thêm 1 file `.png` vào `assets/logo/logo_loc/` — không cần sửa code.
Ảnh mẫu nên là 1 khung logo cắt sát (có nền trong suốt hoặc cắt kèm 1 chút nền
xung quanh cũng được, script tự nhận diện phần màu vàng).

### QR
1. Ưu tiên dùng bộ dò QR chuẩn của OpenCV (`cv2.QRCodeDetector`) — tự định vị
   được **bất kỳ QR nào** dù nội dung mã hoá bên trong khác nhau (số điện thoại/
   Zalo khác nhau giữa các ảnh), không cần thư viện mẫu.
2. Vùng dò được sẽ được "nới" thêm ra (`QR_PAD_*` trong `process.py`) để che
   trọn cả viền trắng và logo/chữ (vd chữ "Zalo") thường nằm cạnh QR.
3. Nếu bộ dò chuẩn không thấy QR (bị mờ/đè hiệu ứng...), mới rơi xuống so khớp
   với thư viện `qr_loc/` (so khớp theo khung/thiết kế, không so khớp từng ô đen
   trắng vì nội dung QR luôn khác nhau giữa các ảnh).
4. Khi xác định được vùng QR, che nền rồi dán QR/Zalo thật của Zeno Homes trong
   `qr_thay_the/` vào.

### Banner hotline
Layout Pacific Homes gốc có 1 banner hotline ở cuối ảnh (`BANNER_PCT`). Các
layout khác (vd Hừng Đông) **không có banner này ở cùng vị trí** — nếu áp dụng
nhầm sẽ đè lên nội dung khác (vd banner "ZENO HOMES" có sẵn). Vì vậy bước này
chỉ chạy khi logo khớp nằm trong `HOTLINE_BANNER_TEMPLATES` (mặc định chỉ có
`logo_pacific.png`). Layout nào có banner hotline riêng, thêm tên file logo
tương ứng vào set này trong `process.py` (và khai báo toạ độ % nếu vị trí khác
`BANNER_PCT`).

---

## ⚠️ Việc cần làm trước khi dùng thật

- **`assets/qr/qr_thay_the/default.png` hiện là QR giả (placeholder)** để test
  cơ chế dán — hãy thay bằng file QR/Zalo THẬT của Zeno Homes trước khi chạy
  hàng loạt cho khách.
- Thư viện `logo_loc/` mới có 2 mẫu (Pacific, Hừng Đông). Gặp sàn/CTV nào khác
  nữa thì cứ thêm file logo của họ vào thư mục này — không cần đụng code.

---

## Test nhanh 1 ảnh trên máy (không cần chờ GitHub Actions)

```bash
pip install opencv-python-headless numpy
python debug_test_local.py duong/dan/anh_goc.jpg anh_ket_qua.jpg
```

Log in ra sẽ cho biết: khớp với mẫu logo nào trong `logo_loc/`, điểm số bao
nhiêu, có tìm thấy QR không, và có áp dụng banner hotline hay không — rất hữu
ích để kiểm tra trước khi thêm 1 layout mới vào thư viện.

---

## Cách dùng (qua GitHub Actions)

### Cách 1: Qua giao diện GitHub (đơn giản nhất)

1. Vào tab **Actions** của repo
2. Chọn workflow **"Process Image"**
3. Nhấn **"Run workflow"**
4. Điền vào:
   - **image_url**: link ảnh gốc (ví dụ: `https://photo-stal-17.zdn.vn/.../abc.jpg`)
   - **job_id**: tên tùy ý để nhận dạng (ví dụ: `haiau_001`)
5. Nhấn **"Run workflow"** màu xanh
6. Chờ ~1 phút → vào log xem link ảnh kết quả ở cuối

### Cách 2: Gọi qua GitHub API (tự động hóa)

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/workflows/process-image.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "image_url": "https://photo-stal-17.zdn.vn/.../abc.jpg",
      "job_id": "haiau_001"
    }
  }'
```

Sau đó poll kết quả:
```bash
curl -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/runs?per_page=1
```

---

## Lấy link ảnh kết quả

```
https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/result-YYYYMMDD-HHMMSS-JOB_ID/result_JOB_ID.jpg
```

Link này **tồn tại vĩnh viễn** (trừ khi bạn xóa Release thủ công).

---

## Giới hạn free tier GitHub

| Thứ | Giới hạn | Thực tế |
|-----|----------|---------|
| Actions minutes | 2.000 phút/tháng | ~1 phút/ảnh → ~2.000 ảnh/tháng |
| Storage Release | 1 GB | ~500KB/ảnh → ~2.000 ảnh |
| Repo size | 1 GB | không ảnh hưởng (ảnh lưu ở Release) |

---

## Setup lần đầu

1. Fork hoặc tạo repo mới, upload toàn bộ file này lên
2. Vào **Settings → Actions → General** → chọn **"Read and write permissions"** cho Workflow permissions
3. Xong! Không cần cấu hình gì thêm (`GITHUB_TOKEN` được tạo tự động)
