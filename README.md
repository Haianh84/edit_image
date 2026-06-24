# Zeno Image Processor

Tự động thay logo Pacific → Zeno Homes và cập nhật hotline trên ảnh bất động sản.  
Chạy 100% trên **GitHub Actions**, miễn phí, không cần server.

---

## Cấu trúc repo

```
├── process.py                        # Script xử lý ảnh
├── assets/
│   ├── logo_zenohomes.png            # Logo mới (cố định)
│   ├── hotline.png                   # Hotline mới (cố định)
│   └── logo_pacific.png              # Logo cũ (tham khảo)
└── .github/workflows/
    └── process-image.yml             # GitHub Actions workflow
```

---

## Cách dùng

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
# Xem run mới nhất
curl -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/runs?per_page=1
```

---

## Lấy link ảnh kết quả

Sau khi workflow chạy xong, link ảnh có dạng:
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
