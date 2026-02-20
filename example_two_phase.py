"""
Example: Two-Phase Indexing and LSH

This script demonstrates the clear separation of indexing (Phase 1) 
and querying (Phase 2) phases.
"""

import numpy as np
from datasketch import MinHash, MinHashLSH


def demonstrate_two_phase_lsh():
    """
    Simple demonstration of the two-phase LSH approach.
    """
    print("=" * 70)
    print("TWO-PHASE LSH DEMONSTRATION")
    print("=" * 70)
    
    # Sample data: movie genres
    movies = [
        {"title": "The Matrix", "genres": ["action", "sci-fi"]},
        {"title": "Inception", "genres": ["action", "sci-fi", "thriller"]},
        {"title": "The Notebook", "genres": ["romance", "drama"]},
        {"title": "Interstellar", "genres": ["sci-fi", "drama", "adventure"]},
        {"title": "John Wick", "genres": ["action", "thriller"]},
    ]
    
    # ========== PHASE 1: INDEXING ==========
    print("\n" + "=" * 70)
    print("PHASE 1: BUILDING LSH INDEX")
    print("=" * 70)
    
    # Create LSH index
    threshold = 0.3  # Jaccard similarity threshold
    num_perm = 128   # Number of permutations for MinHash
    
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    
    print(f"\nIndexing {len(movies)} movies...")
    for i, movie in enumerate(movies):
        # Create MinHash for each movie's genres
        m = MinHash(num_perm=num_perm)
        for genre in movie["genres"]:
            m.update(genre.encode('utf8'))
        
        # Insert into LSH index
        lsh.insert(f"movie_{i}", m)
        minhashes.append(m)
        
        print(f"  ✓ Indexed: {movie['title']} - {movie['genres']}")
    
    print(f"\n✓ LSH index built successfully!")
    print(f"  - Threshold: {threshold}")
    print(f"  - Permutations: {num_perm}")
    
    # ========== PHASE 2: QUERYING ==========
    print("\n" + "=" * 70)
    print("PHASE 2: QUERYING LSH INDEX")
    print("=" * 70)
    
    # Query for similar movies to "The Matrix"
    query_idx = 0
    query_movie = movies[query_idx]
    
    print(f"\nQuery: {query_movie['title']} - {query_movie['genres']}")
    print(f"\nFinding similar movies...")
    
    # Query the LSH index
    result = lsh.query(minhashes[query_idx])
    
    print(f"\nSimilar movies (Jaccard similarity >= {threshold}):")
    for movie_id in result:
        idx = int(movie_id.split("_")[1])
        if idx != query_idx:  # Skip the query movie itself
            similar_movie = movies[idx]
            
            # Calculate exact Jaccard similarity
            query_set = set(query_movie["genres"])
            similar_set = set(similar_movie["genres"])
            jaccard = len(query_set & similar_set) / len(query_set | similar_set)
            
            print(f"  ✓ {similar_movie['title']} - {similar_movie['genres']}")
            print(f"    Jaccard similarity: {jaccard:.3f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Points:")
    print("1. Phase 1 (Indexing): Built LSH index once from all movies")
    print("2. Phase 2 (Querying): Can now query multiple times efficiently")
    print("3. The index can be reused for many queries without rebuilding")
    

if __name__ == "__main__":
    demonstrate_two_phase_lsh()
