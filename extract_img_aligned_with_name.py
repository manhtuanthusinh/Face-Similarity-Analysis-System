import os
import shutil
import pandas as pd

# ==== CONFIG ====
excel_path = "data/20251022/event.xlsx"
src_root = "data/"       
dst_root = "data/20251022/aligned_img_with_name/"  

df = pd.read_excel(excel_path, header=8)

# Filter out unwanted rows
df = df[df['Tiêu đề'] != 'unknown']

# Create destination folder if not exist
os.makedirs(dst_root, exist_ok=True)

# Iterate through each row
for _, row in df.iterrows():
    person_name = str(row['Tiêu đề']).strip()
    img_name = str(row['Ảnh cắt']).strip()

    # Create person subfolder
    person_folder = os.path.join(dst_root, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Build source and destination paths
    src_path = os.path.join(src_root, img_name)
    dst_path = os.path.join(person_folder, os.path.basename(img_name))

    # Copy file if exists
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"File not found: {src_path}")

print("Done")
