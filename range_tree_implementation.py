import numpy as np
import time
from data_preparation import load_and_prepare_data


class RangeTree1D:
    """1D Range Tree (Binary Search Tree)"""
    
    def __init__(self, points, indices, dim):
        self.dim = dim
        
        if len(points) == 0:
            self.value = None
            self.left = None
            self.right = None
            self.indices = []
            return
        
        sorted_order = np.argsort(points[:, dim])
        sorted_points = points[sorted_order]
        sorted_indices = indices[sorted_order]
        
        mid = len(sorted_points) // 2
        self.value = sorted_points[mid, dim]
        self.point = sorted_points[mid]
        self.index = sorted_indices[mid]
        
        if mid > 0:
            self.left = RangeTree1D(sorted_points[:mid], sorted_indices[:mid], dim)
        else:
            self.left = None
            
        if mid + 1 < len(sorted_points):
            self.right = RangeTree1D(sorted_points[mid+1:], sorted_indices[mid+1:], dim)
        else:
            self.right = None
        
        self.indices = sorted_indices
    
    def range_query_1d(self, min_val, max_val):
        """1D range query"""
        if self.value is None:
            return []
        
        results = []
        
        if min_val <= self.value <= max_val:
            results.append(self.index)
        
        if self.left is not None and min_val <= self.value:
            results.extend(self.left.range_query_1d(min_val, max_val))
        
        if self.right is not None and max_val >= self.value:
            results.extend(self.right.range_query_1d(min_val, max_val))
        
        return results


class RangeTree5D:
    """k-D Range Tree - Cascading Binary Search Trees"""
    
    def __init__(self, points, indices=None, dims=None):
        if indices is None:
            indices = np.arange(len(points))

        self.dims = dims if dims is not None else points.shape[1]
        self.trees = []
        for dim in range(self.dims):
            tree = RangeTree1D(points, indices, dim)
            self.trees.append(tree)
    
    def range_query(self, query_bounds):
        """5D range query"""
        candidate_sets = []
        
        for dim, (min_val, max_val) in enumerate(query_bounds):
            results = self.trees[dim].range_query_1d(min_val, max_val)
            candidate_sets.append(set(results))
        
        if len(candidate_sets) == 0:
            return []
        
        final_results = candidate_sets[0]
        for s in candidate_sets[1:]:
            final_results = final_results.intersection(s)
        
        return list(final_results)


def test_range_tree():
    """Δοκιμή Range Tree"""
    
    df, data_norm, _ = load_and_prepare_data()
    
    print("\n=== RANGE TREE ===")
    
    # Build
    start = time.time()
    dims = data_norm.shape[1]
    range_tree = RangeTree5D(data_norm, dims=dims)
    build_time = time.time() - start
    print(f"Build: {build_time:.4f}s")
    
    # Test query
    query_point = data_norm[0]
    
    # Range query
    radius = 0.05
    query_bounds = [(q - radius, q + radius) for q in query_point]
    
    start = time.time()
    results = range_tree.range_query(query_bounds)
    range_time = time.time() - start
    
    print(f"Range: {range_time*1000:.4f}ms ({len(results)} results)")
    
    # k-NN approximation
    start = time.time()
    small_radius = 0.1
    small_bounds = [(q - small_radius, q + small_radius) for q in query_point]
    candidates = range_tree.range_query(small_bounds)
    
    if len(candidates) > 0:
        sample_candidates = candidates[:min(1000, len(candidates))]
        distances = [np.linalg.norm(data_norm[idx] - query_point) for idx in sample_candidates]
        k_nearest_indices = np.argsort(distances)[:5]
    
    knn_time = time.time() - start
    
    print(f"k-NN: {knn_time*1000:.4f}ms (approximate)")
    
    return {
        'name': 'Range Tree',
        'build_time': build_time,
        'knn_time': knn_time,
        'range_time': range_time,
        'range_count': len(results)
    }


if __name__ == "__main__":
    result = test_range_tree()