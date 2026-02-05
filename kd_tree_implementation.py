from sklearn.neighbors import KDTree
import time
import numpy as np
from data_preparation import load_and_prepare_data


def benchmark_kd_tree(data_norm, top_k=5, range_radius=0.05, query_indices=None):
    """Benchmark KD-tree build and query times over multiple queries."""
    start = time.time()
    kd_tree = KDTree(data_norm, leaf_size=30)
    build_time = time.time() - start

    if query_indices is None:
        query_indices = [0]

    knn_times = []
    range_times = []
    for qi in query_indices:
        query = data_norm[int(qi)]
        start = time.time()
        kd_tree.query(query.reshape(1, -1), k=top_k)
        knn_times.append(time.time() - start)

        start = time.time()
        kd_tree.query_radius(query.reshape(1, -1), r=range_radius)
        range_times.append(time.time() - start)

    return {
        "build_time": build_time,
        "knn_time": float(np.mean(knn_times)),
        "range_time": float(np.mean(range_times)),
    }


def test_kd_tree():
    """Δοκιμή k-d tree"""
    
    # Φόρτωση
    df, data_norm, _ = load_and_prepare_data()
    
    # Build
    print("\n=== k-d TREE ===")
    start = time.time()
    kd_tree = KDTree(data_norm, leaf_size=30)
    build_time = time.time() - start
    print(f"Build: {build_time:.4f}s")
    
    # Test query
    query = data_norm[0]
    
    # k-NN
    start = time.time()
    distances, indices = kd_tree.query(query.reshape(1, -1), k=5)
    knn_time = time.time() - start
    print(f"k-NN: {knn_time*1000:.4f}ms")
    
    # Range
    start = time.time()
    indices_range = kd_tree.query_radius(query.reshape(1, -1), r=0.05)
    range_time = time.time() - start
    print(f"Range: {range_time*1000:.4f}ms ({len(indices_range[0])} results)")
    
    # Επιστροφή αποτελεσμάτων
    return {
        'name': 'k-d Tree',
        'build_time': build_time,
        'knn_time': knn_time,
        'range_time': range_time,
        'range_count': len(indices_range[0])
    }


if __name__ == "__main__":
    result = test_kd_tree()