from rtree import index
import time
import numpy as np
from data_preparation import load_and_prepare_data


def benchmark_rtree(data_norm, top_k=5, range_radius=0.05, query_indices=None):
    """Benchmark R-tree build and query times over multiple queries."""
    start = time.time()
    p = index.Property()
    p.dimension = data_norm.shape[1]
    rtree_idx = index.Index(properties=p)

    for i, point in enumerate(data_norm):
        bbox = tuple(point) + tuple(point)
        rtree_idx.insert(i, bbox)

    build_time = time.time() - start

    if query_indices is None:
        query_indices = [0]

    knn_times = []
    range_times = []
    for qi in query_indices:
        query = data_norm[int(qi)]

        start = time.time()
        query_bbox = tuple(query - range_radius) + tuple(query + range_radius)
        list(rtree_idx.intersection(query_bbox))
        range_times.append(time.time() - start)

        start = time.time()
        temp_bbox = tuple(query - 0.2) + tuple(query + 0.2)
        candidates = list(rtree_idx.intersection(temp_bbox))
        if candidates:
            dists = [np.linalg.norm(data_norm[idx] - query) for idx in candidates[:1000]]
            np.argsort(dists)[:top_k]
        knn_times.append(time.time() - start)

    return {
        "build_time": build_time,
        "knn_time": float(np.mean(knn_times)),
        "range_time": float(np.mean(range_times)),
    }


def test_rtree():
    """Δοκιμή R-tree"""
    
    # Φόρτωση
    df, data_norm, _ = load_and_prepare_data()
    
    print("\n=== R-TREE ===")
    
    # Build
    start = time.time()
    p = index.Property()
    p.dimension = data_norm.shape[1]
    rtree_idx = index.Index(properties=p)
    
    for i, point in enumerate(data_norm):
        bbox = tuple(point) + tuple(point)
        rtree_idx.insert(i, bbox)
        if i % 100000 == 0:
            print(f"  Inserted {i} points...")
    
    build_time = time.time() - start
    print(f"Build: {build_time:.4f}s")
    
    # Test query
    query = data_norm[0]
    
    # Range query
    start = time.time()
    query_bbox = tuple(query - 0.05) + tuple(query + 0.05)
    results = list(rtree_idx.intersection(query_bbox))
    range_time = time.time() - start
    print(f"Range: {range_time*1000:.4f}ms ({len(results)} results)")
    
    # k-NN (approximation με range)
    start = time.time()
    temp_bbox = tuple(query - 0.2) + tuple(query + 0.2)
    candidates = list(rtree_idx.intersection(temp_bbox))
    dists = [np.linalg.norm(data_norm[idx] - query) for idx in candidates[:1000]]
    knn_time = time.time() - start
    print(f"k-NN: {knn_time*1000:.4f}ms (approximate)")
    
    return {
        'name': 'R-tree',
        'build_time': build_time,
        'knn_time': knn_time,
        'range_time': range_time,
        'range_count': len(results)
    }


if __name__ == "__main__":
    result = test_rtree()