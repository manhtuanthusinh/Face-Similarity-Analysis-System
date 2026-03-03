import os
import numpy as np
from PIL import Image
from core.data_load import load_and_aligned_face


def safe_align_face(image_path):
    """Tải hoặc align khuôn mặt an toàn, bỏ qua nếu đã aligned hoặc lỗi detect"""
    # Nếu ảnh đã aligned --> đọc trực tiếp
    if "aligned" in os.path.basename(image_path).lower():
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Cannot open aligned image {image_path}: {e}")
            return None

    # Nếu chưa aligned --> dùng MTCNN
    try:
        face_img = load_and_aligned_face(image_path)
        if face_img is None:
            print(f"[WARN] No face detected: {os.path.basename(image_path)}")
        return face_img
    except Exception as e:
        print(f"[ERROR] Alignment failed for {os.path.basename(image_path)}: {e}")
        return None

