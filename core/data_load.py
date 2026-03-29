import glob
import os
from AdaFace.face_alignment import align
from config import DATASET_PATH

def load_images_path():
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in exts:
        # duyệt ảnh trong mỗi thư mục con (1 thư mục = 1 person)
        image_files.extend(glob.glob(os.path.join(DATASET_PATH, "*", ext)))
    image_files = sorted(image_files)
    # debug
    print(f"[INFO] Found {len(image_files)} images in {DATASET_PATH}")
    return image_files

def get_person_name(path):
    return os.path.basename(os.path.dirname(path))

def load_and_aligned_face(path):
    aligned_img = align.get_aligned_face(path)
    if aligned_img is None:
        return None
    return aligned_img  # return PIL Image (RGB)
