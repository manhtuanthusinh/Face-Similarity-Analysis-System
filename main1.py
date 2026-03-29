import os
import numpy as np
from core.data_load import load_images_path, get_person_name, load_and_aligned_face
from core.model_load import load_pretrained_model, to_input
from core.safe_align_face import safe_align_face
from config import DATASET_PATH, MODEL_ARCHITECTURE, OUTPUT_FEATURES_NPY, OUTPUT_LABELS_NPY

# save npy output
if __name__ == '__main__':
    model = load_pretrained_model(MODEL_ARCHITECTURE)
    image_files = load_images_path()

    embeddings = []
    labels = []
   
    for path in image_files:
        fname = os.path.basename(path)
        person = get_person_name(path)
        print(f"[INFO] Processing {fname} of {person}")

        aligned_img = safe_align_face(path)
        if aligned_img is None:
            print(f"[WARN] Skipping {fname} due to alignment failure")
            continue

        try:
            input_tensor = to_input(aligned_img)
            feature, _ = model(input_tensor)
            feature = feature.detach().cpu().numpy().flatten()
            embeddings.append(feature)
            labels.append(person)
            print(f"[INFO] Embedding shape: {feature.shape}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            continue
    
    embeddings = np.stack(embeddings)
    labels = np.array(labels, dtype=object)

    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_NPY), exist_ok=True)
    np.save(OUTPUT_FEATURES_NPY, embeddings)
    np.save(OUTPUT_LABELS_NPY, labels)

    print(f"[INFO] Done. Saved {len(embeddings)} embeddings.")
    print(f"[INFO] Features saved to: {OUTPUT_FEATURES_NPY}")
    print(f"[INFO] Labels saved to:   {OUTPUT_LABELS_NPY}")
