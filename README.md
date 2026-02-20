# Polydiastates - Multi-dimensional Data Indexing

This project implements and compares various spatial indexing structures combined with Locality-Sensitive Hashing (LSH) for efficient multi-dimensional movie data search.

## Two-Phase Approach

The implementation follows a clear **two-phase methodology**:

### Phase 1: Indexing (Build)
- **Spatial Indexes**: Build data structures for 5D numerical movie attributes (budget, revenue, runtime, popularity, vote_average)
  - k-d Tree
  - Quad Tree
  - Range Tree
  - R-tree
- **LSH Index**: Build MinHash-based LSH index for textual attributes (genres, production companies)

### Phase 2: Querying
- **Spatial Queries**: 
  - k-NN (k-nearest neighbors) search
  - Range queries
- **LSH Queries**: Find similar movies based on textual attributes using Jaccard similarity

## Project Structure

```
.
├── example_two_phase.py         # Simple demonstration of two-phase LSH
├── data_preparation.py          # Data loading and normalization
├── lsh_implementation.py        # LSH index building and querying
├── kd_tree_implementation.py    # k-d tree implementation
├── quad_tree_implementation.py  # Quad tree implementation
├── range_tree_implementation.py # Range tree implementation
├── rtree_implementation.py      # R-tree implementation
├── final_comparison.py          # Combined benchmark of all structures
└── .vscode/                     # VS Code configuration
    ├── settings.json            # Editor settings
    ├── launch.json              # Debug configurations
    └── extensions.json          # Recommended extensions
```

## Running in VS Code

### Prerequisites
1. Install Python 3.8+
2. Install recommended VS Code extensions:
   - Python (ms-python.python)
   - Pylance (ms-python.vscode-pylance)
   - Jupyter (ms-toolsai.jupyter)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn datasketch rtree openpyxl
```

### Running the Code

#### Option 1: Using VS Code Debugger
1. Open the project folder in VS Code
2. Press `F5` or go to Run and Debug (Ctrl+Shift+D)
3. Select the configuration you want to run:
   - "Python: Two-Phase Example" - Simple LSH demonstration (recommended first run!)
   - "Python: LSH Implementation" - Run LSH tests
   - "Python: Final Comparison" - Run complete benchmark
   - "Python: k-d Tree" - Test k-d tree only
   - etc.

#### Option 2: Using Terminal
```bash
# Quick demonstration of two-phase approach (no data file needed)
python example_two_phase.py

# Test individual implementations (requires data file)
python lsh_implementation.py
python kd_tree_implementation.py
python rtree_implementation.py

# Run complete comparison (requires data file)
python final_comparison.py

# Test data preparation (requires data file)
python data_preparation.py
```

## Key Features

### LSH Implementation (`lsh_implementation.py`)
- **Phase 1**: `build_lsh_index()` - Builds MinHash signatures and LSH index
- **Phase 2**: `query_lsh()` - Queries the index for similar items
- `filtered_lsh_top_n()` - Complete pipeline with filtering

### Final Comparison (`final_comparison.py`)
- Benchmarks all spatial structures + LSH
- Clear separation of indexing and querying phases
- Performance metrics for build time and query time
- Supports multiple queries for statistical analysis

## Performance Metrics

The implementation tracks:
- **Build Time**: Time to construct the index (Phase 1)
- **Query Time**: Time to execute queries (Phase 2)
  - k-NN query time
  - Range query time
  - LSH query time

## Data Format

The code expects a CSV or XLSX file named `data_movies_clean.csv` or `data_movies_clean.xlsx` with the following columns:
- Numerical: `budget`, `revenue`, `runtime`, `popularity`, `vote_average`
- Textual: `genres`, `production_companies` (or similar)
- Metadata: `title`, `release_year`, `original_language`, `origin_country`

## Example Output

```
======================================================================
PHASE 1: BUILDING INDEXES
======================================================================
Building LSH index for 'genre_names'...
✓ LSH index built in 0.1234s

======================================================================
PHASE 2: QUERYING INDEXES
======================================================================

1. Testing k-d Tree...
  Building k-d tree index...
  ✓ k-d tree: build=0.5678s, knn=0.000123s, range=0.000456s

2. Testing Quad Tree...
...
```

## License

This project is for educational purposes.
