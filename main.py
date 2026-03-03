import csv
import os
import numpy as np
from core.data_load import load_images_path, get_person_name, load_and_aligned_face
from core.model_load import load_pretrained_model, to_input
from config import DATASET_PATH, OUTPUT_CSV, MODEL_ARCHITECTURE

# save csv output
if __name__ == '__main__':
    model = load_pretrained_model(MODEL_ARCHITECTURE)
    image_files = load_images_path()

    # to csv
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok = True)
    with open(OUTPUT_CSV, "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["person", "image", "embedding"])
        for path in image_files:
            fname = os.path.basename(path)
            person = get_person_name(path)
            aligned_img = load_and_aligned_face(path)
            print(f"[INFO] Processing {fname} of {person}")
            if aligned_img is None:
                print(f"[WARN] Face not detected: {fname}")
                continue
            try:
                input_tensor = to_input(aligned_img)
                feature, _ = model(input_tensor)
                feature = feature.detach().cpu().numpy().flatten()
                print(f"[INFO] Embedding shape: {feature.shape}")
                writer.writerow([person, fname, feature.tolist()])
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

    print(f"[INFO] Done. Embeddings saved to {OUTPUT_CSV}")
