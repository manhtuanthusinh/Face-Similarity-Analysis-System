import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from core.data_load import load_and_aligned_face
from core.model_load import load_pretrained_model, to_input
from config import MODEL_ARCHITECTURE, OUTPUT_FEATURES_NPY, OUTPUT_LABELS_NPY, OUTPUT_XLSX

# CONFIG
QUERY_FOLDER = "data/20251022/aligned_img_with_name"   # input folder
THRESHOLD = 0.3


def get_image_files(folder):
    exts = [".jpg", ".jpeg", ".png"]
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return files


def safe_align_face(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open aligned image {image_path}: {e}")
        return None


def extract_embeddings(model, folder):
    """Extract embeddings for all images in a folder"""
    embeddings, img_names, person_names = [], [], []

    image_files = get_image_files(folder)
    for img_path in image_files:
        fname = os.path.basename(img_path)
        person = os.path.basename(os.path.dirname(img_path))
        face_img = safe_align_face(img_path)

        if face_img is None:
            continue

        try:
            input_tensor = to_input(face_img)
            with torch.no_grad():
                feature, _ = model(input_tensor)
            emb = feature.detach().cpu().numpy().flatten()
            
            ## normalize
            # emb = emb / np.linalg.norm(emb)

            embeddings.append(emb)
            img_names.append(fname)
            person_names.append(person)
            print(f"[INFO] Embedded {person}/{fname}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    return np.array(embeddings), img_names, person_names


# MAIN 
def main():
    # Load model
    print("[INFO] Loading pretrained model...")
    model = load_pretrained_model(MODEL_ARCHITECTURE)
    model.eval()

    # Load database
    print("[INFO] Loading database embeddings...")
    db_features = np.load(OUTPUT_FEATURES_NPY)
    db_labels = np.load(OUTPUT_LABELS_NPY, allow_pickle=True)

    ## Normalize DB features
    # db_features = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)

    # Extract embeddings for query images
    print(f"[INFO] Extracting embeddings from folder: {QUERY_FOLDER}")
    query_embs, img_names, person_names = extract_embeddings(model, QUERY_FOLDER)

    if len(query_embs) == 0:
        print("[ERROR] No valid query embeddings found")
        return

    print("[INFO] Computing cosine similarity...")
    sims = cosine_similarity(query_embs, db_features)

    # results
    results = []
    for i, q_name in enumerate(img_names):
        person_folder = person_names[i]
        top_idx = np.argsort(sims[i])[::-1][:1]
        best_label = db_labels[top_idx[0]]
        best_sim = float(sims[i][top_idx[0]])
        is_match = (best_sim >= THRESHOLD) & (best_label == person_folder)

        if is_match:
            reason = ""
        elif person_folder not in db_labels:
            reason = "person not in db"
        else:
            reason = "wrong detection"

        results.append({
            "person_folder": person_folder,
            "image_name": q_name,
            "top1_match_name": best_label,
            "cosine_similarity": round(best_sim, 4),
            "threshold": THRESHOLD,
            "match": bool(is_match),
            "reason": reason 
        })
    
        print(f"\n[RESULT] {person_folder}/{q_name} → {best_label} ({best_sim:.3f})   Match={is_match}")

    # Save results to Excel
    df = pd.DataFrame(results)
    # df['match'] = df["match"].apply(lambda x: "True" if x else "False") 
    df['match'] = df["match"].map({True: "True", False: "False"})
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\n[INFO] Results saved to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
