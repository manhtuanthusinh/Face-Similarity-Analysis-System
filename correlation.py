import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

class CorrelationMatrix(object):
    def __init__(self, features_path, label_path):
        self.features = np.load(features_path)
        self.features = self.features.reshape(self.features.shape[0], -1)
        self.labels = np.sort(np.load(label_path, allow_pickle=True))
    
    def calculate_correlation_matrix(self):
        # calculate correlation matrix by cosine similarity
        # using scikit-learn
        correlation_matrix = cosine_similarity(self.features, self.features)
        # clip value from -1 ,1 to 0,1
        # correlation_matrix = np.clip(correlation_matrix, 0, 1)
        return correlation_matrix
    
    def far_frr(self, correlation_matrix):
        """
        Calculate far and frr
        Args:
            correlation_matrix: correlation matrix
            labels: labels
        Returns:
            far: false accept rate
            frr: false reject rate
        Approach:
            - For each label, calculate the inter-class distance and intra-class distance
            -> for the threshold step 
            -> calculate the inter-rejecting rate and intra-accepting rate
            -> far and frr for each label
            -> far and frr for all labels
            -> plot the far and frr for all labels
            - x label: threshold
            - y left label: far
            - y right label: frr
        """
        # Convert to numpy array if not already
        labels = np.array(self.labels)
        
        # Create label comparison matrix (vectorized way)
        # labels_matrix[i,j] = True if labels[i] == labels[j]
        labels_matrix = labels[:, None] == labels[None, :]
        
        # Create masks for intra-class and inter-class
        # Remove diagonal (self-comparison)
        mask_diagonal = np.eye(len(labels), dtype=bool)
        
        # Intra-class mask: same labels but not diagonal
        intra_mask = labels_matrix & ~mask_diagonal
        
        # Inter-class mask: different labels
        inter_mask = ~labels_matrix
        
        # Extract intra-class and inter-class distances
        intra_distances = correlation_matrix[intra_mask]
        inter_distances = correlation_matrix[inter_mask]
        
        # Define threshold range
        min_sim = min(correlation_matrix.min(), 0)
        max_sim = correlation_matrix.max()
        thresholds = np.linspace(min_sim, max_sim, 100)
        
        # Calculate FAR and FRR for each threshold (vectorized)
        far_list = []
        frr_list = []
        
        for threshold in thresholds:
            # FAR: False Accept Rate
            # Percentage of inter-class pairs accepted (similarity >= threshold)
            false_accepts = np.sum(inter_distances >= threshold)
            total_inter = len(inter_distances)
            far = false_accepts / total_inter if total_inter > 0 else 0
            
            # FRR: False Reject Rate  
            # Percentage of intra-class pairs rejected (similarity < threshold)
            false_rejects = np.sum(intra_distances < threshold)
            total_intra = len(intra_distances)
            frr = false_rejects / total_intra if total_intra > 0 else 0
            
            far_list.append(far)
            frr_list.append(frr)
        
        far_array = np.array(far_list)
        frr_array = np.array(frr_list)
        
        # Plot FAR and FRR
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left y-axis for FAR
        color = 'tab:red'
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('FAR (False Accept Rate)', color=color)
        ax1.plot(thresholds, far_array, color=color, label='FAR', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Right y-axis for FRR
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('FRR (False Reject Rate)', color=color)
        ax2.plot(thresholds, frr_array, color=color, label='FRR', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Find EER (Equal Error Rate) point
        eer_idx = np.argmin(np.abs(far_array - frr_array))
        eer_threshold = thresholds[eer_idx]
        eer_value = (far_array[eer_idx] + frr_array[eer_idx]) / 2
        
        # Mark EER point
        ax1.axvline(x=eer_threshold, color='green', linestyle='--', alpha=0.7)
        ax1.text(eer_threshold, eer_value, f'EER: {eer_value:.4f}\nThreshold: {eer_threshold:.4f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green"))
        
        plt.title('FAR vs FRR Curve')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Total intra-class pairs: {len(intra_distances)}")
        print(f"Total inter-class pairs: {len(inter_distances)}")
        print(f"EER: {eer_value:.4f} at threshold: {eer_threshold:.4f}")
        print(f"Intra-class similarity - Mean: {intra_distances.mean():.4f}, Std: {intra_distances.std():.4f}")
        print(f"Inter-class similarity - Mean: {inter_distances.mean():.4f}, Std: {inter_distances.std():.4f}")
        
        return {
            'thresholds': thresholds,
            'far': far_array,
            'frr': frr_array,
            'eer': eer_value,
            'eer_threshold': eer_threshold,
            'intra_distances': intra_distances,
            'inter_distances': inter_distances
        }
    
    def adaptive_threshold(self, correlation_matrix):
        """
        Calculate adaptive threshold
        Args:
            correlation_matrix: correlation matrix
        Returns:
            adaptive_threshold: adaptive threshold
        Approach:
            - For each row, calculate the maximal value to classify to intra class and minimal value to classify to inter class
            - Adaptive threshold = (max_intra + min_inter) / 2
            - All of the inter-class distance should be less than the threshold
            - All of the intra-class distance should be greater than the threshold
            - Using numpy tensor to calculate the adaptive threshold
            - Plot the adaptive threshold for each sample (row)
        """
        labels = np.array(self.labels)
        n_samples = len(labels)
        
        # Initialize arrays
        adaptive_thresholds = np.zeros(n_samples)
        min_intra_per_sample = np.zeros(n_samples)
        max_inter_per_sample = np.zeros(n_samples)
        
        # Debug: print first few labels to understand the data structure
        print(f"Total samples: {n_samples}")
        print(f"First 10 labels: {labels[:10]}")
        print(f"Unique labels: {np.unique(labels)}")
        print(f"Label counts: {np.unique(labels, return_counts=True)[1]}")
        
        # Single loop for each sample row
        for i in range(n_samples):
            # Get row i similarities
            row_similarities = correlation_matrix[i, :]
            # Get label of current sample
            current_label = labels[i]
            
            # Create intra-class mask: samples with the same label (excluding self)
            intra_mask = (labels == current_label)
            intra_mask[i] = False  # Exclude self
            
            # Create inter-class mask: samples with different labels
            inter_mask = (labels != current_label)
            
            # Debug for first sample
            if i == 0:
                print(f"\nSample {i}: label = {current_label}")
                print(f"Intra-class indices: {np.where(intra_mask)[0]}")
                print(f"Inter-class indices: {np.where(inter_mask)[0][:10]}...")  # Show first 10
                print(f"Intra count: {np.sum(intra_mask)}, Inter count: {np.sum(inter_mask)}")
            
            # Calculate min(intra) and max(inter) for this sample
            intra_sims = row_similarities[intra_mask]
            inter_sims = row_similarities[inter_mask]

            min_intra = np.min(intra_sims) if len(intra_sims) > 0 else 0
            max_inter = np.max(inter_sims) if len(inter_sims) > 0 else 1

            min_intra_per_sample[i] = min_intra
            max_inter_per_sample[i] = max_inter
            
            # Adaptive threshold = midpoint between min_intra and max_inter
            # For separation: min_intra should > threshold > max_inter
            adaptive_thresholds[i] = (min_intra + max_inter) / 2
        
        # Calculate gap for separability analysis (positive gap = separable)
        gap = min_intra_per_sample - max_inter_per_sample
        
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Adaptive thresholds per sample
        ax1.plot(adaptive_thresholds, 'b-', linewidth=2, label='Adaptive Threshold')
        ax1.plot(min_intra_per_sample, 'g--', alpha=0.7, label='Min Intra-class')
        ax1.plot(max_inter_per_sample, 'r--', alpha=0.7, label='Max Inter-class')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Similarity')
        ax1.set_title('Adaptive Thresholds per Sample')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        # Plot 2: Gap between min_intra and max_inter
        ax2.plot(gap, 'purple', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Gap (Overlap)')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Gap (Min Intra - Max Inter)')
        ax2.set_title('Separability Gap per Sample')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of adaptive thresholds
        ax3.hist(adaptive_thresholds, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Adaptive Threshold Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Adaptive Thresholds')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Histogram of gaps
        ax4.hist(gap, bins=50, alpha=0.7, edgecolor='black', color='purple')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Gap')
        ax4.set_xlabel('Gap Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Separability Gaps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n=== Adaptive Threshold Analysis ===")
        print(f"Average adaptive threshold: {np.mean(adaptive_thresholds):.4f}")
        print(f"Std of adaptive thresholds: {np.std(adaptive_thresholds):.4f}")
        print(f"Min adaptive threshold: {np.min(adaptive_thresholds):.4f}")
        print(f"Max adaptive threshold: {np.max(adaptive_thresholds):.4f}")
        
        # Separability analysis
        positive_gap_samples = np.sum(gap > 0)
        negative_gap_samples = np.sum(gap <= 0)
        print(f"\nSeparability Analysis:")
        print(f"Samples with positive gap (separable): {positive_gap_samples}/{n_samples} ({positive_gap_samples/n_samples*100:.1f}%)")
        print(f"Samples with negative gap (overlapping): {negative_gap_samples}/{n_samples} ({negative_gap_samples/n_samples*100:.1f}%)")
        print(f"Average gap: {np.mean(gap):.4f}")
        print(f"Gap std: {np.std(gap):.4f}")
        
        return {
            'adaptive_thresholds': adaptive_thresholds,
            'min_intra_per_sample': min_intra_per_sample,
            'max_inter_per_sample': max_inter_per_sample,
            'gap': gap
        }

if __name__ == "__main__":
    import os
    root = r""
    target = 'output'
    features_path = os.path.join(root, target, "features.npy")
    label_path = os.path.join(root, target, "labels.npy")
    correlation_matrix = CorrelationMatrix(features_path, label_path)
    corr_matrix = correlation_matrix.calculate_correlation_matrix()
    plt.imshow(corr_matrix)
    plt.show()
    # Calculate FAR/FRR curves
    far_frr_results = correlation_matrix.far_frr(corr_matrix)
    # Calculate adaptive thresholds
    adaptive_results = correlation_matrix.adaptive_threshold(corr_matrix)