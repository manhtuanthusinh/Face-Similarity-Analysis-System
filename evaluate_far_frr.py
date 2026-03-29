import csv
import numpy as np
import matplotlib.pyplot as plt
from core.metrics import compute_far_frr
import ast

def load_embeddings(csv_file):
    data = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # emb = np.array(eval(row["embedding"]), dtype=np.float32)
            emb = np.array(ast.literal_eval(row["embedding"]), dtype=np.float32)
            data.append((row["person"], row["image"], emb))
    return data

if __name__ == "__main__":
    embeddings = load_embeddings("output/embedding_vectors.csv")
    # thresholds = np.linspace(0.1, 0.8, 8)
    thresholds = np.linspace(0.2, 0.3, 11)
    fars, frrs = [], []

    # for thr in tqdm(thresholds, desc="Evaluating thresholds"):
    for thr in thresholds:
        far, frr = compute_far_frr(embeddings, threshold=thr)
        fars.append(far)
        frrs.append(frr)
        print(f"Threshold={thr:.2f} | FAR={far:.3f} | FRR={frr:.3f}")

    plt.figure(figsize=(10,6))
    plt.plot(thresholds, fars, marker='o', label="FAR (False Acceptance Rate)")
    plt.plot(thresholds, frrs, marker='s', label="FRR (False Rejection Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.title("FAR & FRR vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/far_frr_plot.png")
    plt.show()