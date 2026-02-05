import numpy as np
import time
from data_preparation import load_and_prepare_data


class QuadTreeNode:
    """Κόμβος Quad Tree για k-D"""
    
    def __init__(self, bounds, depth=0, max_depth=10, max_points=100, dim=None):
        self.bounds = bounds
        self.depth = depth
        self.max_depth = max_depth
        self.max_points = max_points
        self.points = []
        self.children = None
        self.dim = dim if dim is not None else len(bounds)
    
    def insert(self, idx, point):
        """Εισαγωγή σημείου"""
        if self.children is not None:
            child_idx = self._get_child_index(point)
            self.children[child_idx].insert(idx, point)
            return
        
        self.points.append((idx, point))
        
        if len(self.points) > self.max_points and self.depth < self.max_depth:
            self._split()
    
    def _split(self):
        """Διαίρεσε τον κόμβο σε 2^k παιδιά"""
        self.children = []
        
        for i in range(2 ** self.dim):
            child_bounds = []
            binary = format(i, f'0{self.dim}b')
            
            for dim, bit in enumerate(binary):
                min_val, max_val = self.bounds[dim]
                mid = (min_val + max_val) / 2
                
                if bit == '0':
                    child_bounds.append((min_val, mid))
                else:
                    child_bounds.append((mid, max_val))
            
            child = QuadTreeNode(child_bounds, self.depth + 1, 
                                self.max_depth, self.max_points, dim=self.dim)
            self.children.append(child)
        
        for idx, point in self.points:
            child_idx = self._get_child_index(point)
            self.children[child_idx].insert(idx, point)
        
        self.points = []
    
    def _get_child_index(self, point):
        """Βρες ποιο παιδί ανήκει το σημείο"""
        index = 0
        for dim in range(self.dim):
            min_val, max_val = self.bounds[dim]
            mid = (min_val + max_val) / 2
            if point[dim] >= mid:
                index |= (1 << (self.dim - 1 - dim))
        return index
    
    def range_query(self, query_bounds):
        """Range query"""
        results = []
        
        if not self._intersects(query_bounds):
            return results
        
        if self.children is not None:
            for child in self.children:
                results.extend(child.range_query(query_bounds))
        else:
            for idx, point in self.points:
                if self._point_in_bounds(point, query_bounds):
                    results.append(idx)
        
        return results
    
    def _intersects(self, query_bounds):
        """Έλεγχος αν το query τέμνει τα bounds"""
        for dim in range(self.dim):
            if query_bounds[dim][1] < self.bounds[dim][0] or \
               query_bounds[dim][0] > self.bounds[dim][1]:
                return False
        return True
    
    def _point_in_bounds(self, point, bounds):
        """Έλεγχος αν το σημείο είναι μέσα στα bounds"""
        for dim in range(self.dim):
            if point[dim] < bounds[dim][0] or point[dim] > bounds[dim][1]:
                return False
        return True


def test_quad_tree():
    """Δοκιμή Quad Tree"""
    
    df, data_norm, _ = load_and_prepare_data()
    
    print("\n=== QUAD TREE ===")
    
    # Build
    start = time.time()
    dims = data_norm.shape[1]
    bounds = [(0, 1)] * dims
    quad_tree = QuadTreeNode(bounds, max_depth=8, max_points=100, dim=dims)
    
    for i, point in enumerate(data_norm):
        quad_tree.insert(i, point)
        if i % 100000 == 0:
            print(f"  Inserted {i} points...")
    
    build_time = time.time() - start
    print(f"Build: {build_time:.4f}s")
    
    # Test query
    query_point = data_norm[0]
    
    # Range query
    radius = 0.05
    query_bounds = [(q - radius, q + radius) for q in query_point]
    
    start = time.time()
    results = quad_tree.range_query(query_bounds)
    range_time = time.time() - start
    
    print(f"Range: {range_time*1000:.4f}ms ({len(results)} results)")
    
    # k-NN approximation
    start = time.time()
    small_radius = 0.1
    small_bounds = [(q - small_radius, q + small_radius) for q in query_point]
    candidates = quad_tree.range_query(small_bounds)
    
    if len(candidates) > 0:
        # Περιόρισε σε 1000 για ταχύτητα
        sample_candidates = candidates[:min(1000, len(candidates))]
        distances = [np.linalg.norm(data_norm[idx] - query_point) for idx in sample_candidates]
        k_nearest_indices = np.argsort(distances)[:5]
    
    knn_time = time.time() - start
    
    print(f"k-NN: {knn_time*1000:.4f}ms (approximate)")
    
    return {
        'name': 'Quad Tree',
        'build_time': build_time,
        'knn_time': knn_time,
        'range_time': range_time,
        'range_count': len(results)
    }


if __name__ == "__main__":
    result = test_quad_tree()