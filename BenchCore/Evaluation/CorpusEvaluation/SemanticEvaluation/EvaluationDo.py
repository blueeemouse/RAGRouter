import numpy as np
from typing import Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import skew


class SemanticEvaluator:
    """Evaluate semantic quality of embeddings using space metrics"""

    def __init__(self, hubness_k: int = 10):
        """
        Args:
            hubness_k: k value for hubness calculation (default: 10)
        """
        self.hubness_k = hubness_k

    def evaluate(self, embeddings_by_doc: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate semantic quality of embeddings

        Args:
            embeddings_by_doc: Output from EmbeddingSaver.load()
                              Format: {doc_id: {"chunks": [...], "embeddings": array}}

        Returns:
            Dict containing semantic evaluation metrics
        """
        # Step 1: Merge all embeddings into one matrix
        all_embeddings = []
        for doc_id in sorted(embeddings_by_doc.keys()):
            doc_embeddings = embeddings_by_doc[doc_id]["embeddings"]
            all_embeddings.append(doc_embeddings)

        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')

        n_samples = embeddings_matrix.shape[0]
        embedding_dim = embeddings_matrix.shape[1]

        if n_samples == 0:
            return self._empty_metrics()

        # Step 2: Compute space metrics
        try:
            semantic_complexity = self._compute_twonn(embeddings_matrix)
            semantic_distinguishability = self._compute_centroid_distance(embeddings_matrix)
            semantic_interference = self._compute_hubness(embeddings_matrix)

            return {
                "status": "success",
                "total_embeddings": n_samples,
                "embedding_dimension": embedding_dim,
                "semantic_complexity": semantic_complexity,
                "semantic_distinguishability": semantic_distinguishability,
                "semantic_interference": semantic_interference
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "total_embeddings": n_samples,
                "embedding_dimension": embedding_dim
            }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for empty embeddings"""
        return {
            "status": "failed",
            "error": "No embeddings provided",
            "total_embeddings": 0,
            "embedding_dimension": 0
        }

    def _compute_twonn(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute TwoNN intrinsic dimension estimation (semantic complexity)

        Args:
            embeddings: numpy array, shape [n_samples, embedding_dim]

        Returns:
            dict: {"intrinsic_dimension": float, "valid_samples": int}
        """
        n_samples = embeddings.shape[0]

        # 1. Need at least 3 samples
        if n_samples < 3:
            return {"intrinsic_dimension": None, "valid_samples": 0}

        # 2. L2 normalization
        embeddings = normalize(embeddings, norm='l2')

        # 3. Find nearest neighbors (k=3: self, nearest, second nearest)
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', n_jobs=-1).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # distances[:, 0] is self (0)
        r1 = distances[:, 1]
        r2 = distances[:, 2]

        # 4. Filter duplicate points (r1 == 0 cases)
        mask = r1 > 1e-7
        r1 = r1[mask]
        r2 = r2[mask]

        if len(r1) < 2:
            return {"intrinsic_dimension": None, "valid_samples": 0}

        # 5. Compute � = r2 / r1
        mu = r2 / r1

        # Filter mu >= 1.0
        mu = mu[mu >= 1.0]

        if len(mu) == 0:
            return {"intrinsic_dimension": None, "valid_samples": 0}

        # 6. Compute intrinsic dimension (MLE formula)
        # ID = 1 / mean(ln(mu))
        intrinsic_dim = 1.0 / np.mean(np.log(mu))

        return {
            "intrinsic_dimension": round(float(intrinsic_dim), 4),
            "valid_samples": len(mu)
        }

    def _compute_centroid_distance(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute average centroid distance (semantic distinguishability)

        Args:
            embeddings: numpy array, shape [n_samples, embedding_dim]

        Returns:
            dict: centroid distance statistics
        """
        # 1. Compute centroid
        centroid = np.mean(embeddings, axis=0)

        # 2. Compute cosine distance from each point to centroid
        distances = []
        for emb in embeddings:
            # Cosine distance = 1 - cosine similarity
            sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            dist = 1 - sim
            distances.append(dist)

        distances = np.array(distances)

        return {
            "avg_centroid_distance": round(float(np.mean(distances)), 4),
            "std_centroid_distance": round(float(np.std(distances)), 4),
            "min_centroid_distance": round(float(np.min(distances)), 4),
            "max_centroid_distance": round(float(np.max(distances)), 4)
        }

    def _compute_hubness(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute hubness score (semantic interference)
        Based on skewness of k-occurrence distribution

        Args:
            embeddings: numpy array, shape [n_samples, embedding_dim]

        Returns:
            dict: hubness metrics
        """
        k = self.hubness_k
        n_samples = embeddings.shape[0]

        if n_samples < k + 1:
            return {
                "hubness_score": None,
                "k": k,
                "top_hubs": [],
                "debug_max_occurrence": 0,
                "debug_mean_occurrence": 0.0
            }

        # 1. L2 normalization
        embeddings = normalize(embeddings, norm='l2')

        # 2. Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean', n_jobs=-1).fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)

        # 3. Count k-occurrence (how many times each point appears as neighbor)
        neighbor_indices = indices[:, 1:].flatten()
        k_occurrence = np.bincount(neighbor_indices, minlength=n_samples)

        # 4. Compute hubness score (skewness)
        try:
            hubness_score = skew(k_occurrence)
            if np.isnan(hubness_score):
                hubness_score = 0.0
        except Exception:
            hubness_score = 0.0

        # 5. Find top hubs
        top_indices = np.argsort(k_occurrence)[::-1][:5]
        top_hubs = [
            {
                "chunk_index": int(idx),
                "occurrence": int(k_occurrence[idx])
            }
            for idx in top_indices
            if k_occurrence[idx] > 0
        ]

        return {
            "hubness_score": round(float(hubness_score), 4),
            "k": k,
            "top_hubs": top_hubs,
            "debug_max_occurrence": int(np.max(k_occurrence)),
            "debug_mean_occurrence": round(float(np.mean(k_occurrence)), 4)
        }


# Usage example
if __name__ == "__main__":
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver

    dataset_name = "hotpotqa"

    # Step 1: Load embeddings
    print("Loading embeddings...")
    embeddings_by_doc = EmbeddingSaver.load(dataset_name)

    # Step 2: Evaluate semantic quality
    print("\nEvaluating semantic quality...")
    evaluator = SemanticEvaluator(hubness_k=10)
    metrics = evaluator.evaluate(embeddings_by_doc)

    # Print results
    print("\n=== Semantic Evaluation Results ===")
    print(f"Status: {metrics.get('status')}")
    print(f"Total Embeddings: {metrics.get('total_embeddings')}")
    print(f"Embedding Dimension: {metrics.get('embedding_dimension')}")

    if metrics.get('status') == 'success':
        print("\nSemantic Complexity:")
        complexity = metrics['semantic_complexity']
        print(f"  Intrinsic Dimension: {complexity.get('intrinsic_dimension')}")
        print(f"  Valid Samples: {complexity.get('valid_samples')}")

        print("\nSemantic Distinguishability:")
        distinguishability = metrics['semantic_distinguishability']
        print(f"  Avg Centroid Distance: {distinguishability.get('avg_centroid_distance')}")
        print(f"  Std Centroid Distance: {distinguishability.get('std_centroid_distance')}")
        print(f"  Min Centroid Distance: {distinguishability.get('min_centroid_distance')}")
        print(f"  Max Centroid Distance: {distinguishability.get('max_centroid_distance')}")

        print("\nSemantic Interference:")
        interference = metrics['semantic_interference']
        print(f"  Hubness Score: {interference.get('hubness_score')}")
        print(f"  k: {interference.get('k')}")
        print(f"  Debug Max Occurrence: {interference.get('debug_max_occurrence')}")
        print(f"  Debug Mean Occurrence: {interference.get('debug_mean_occurrence')}")
        print(f"  Top 5 Hubs: {interference.get('top_hubs')}")
