import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

from data_preparation import _load_dataset
from quad_tree_implementation import QuadTreeNode
from range_tree_implementation import RangeTree5D
from kd_tree_implementation import benchmark_kd_tree
from rtree_implementation import benchmark_rtree
from lsh_implementation import _resolve_column, _extract_year, _contains_any_country, build_lsh_index, query_lsh


def combined_test_with_lsh(
    text_attribute="genre_names",
    year_range=(2000, 2020),
    popularity_range=(3, 6),
    vote_average_range=(3, 5),
    runtime_range=(30, 60),
    origin_countries=("US", "GB"),
    original_language="en",
    top_k=5,
    range_radius=0.05,
    lsh_threshold=0.5,
    lsh_num_perm=128,
    num_queries=50,
    random_seed=42,
):
    """Συνδυαστικό benchmark: Spatial index (k=5) + LSH για textual attribute."""

    print("=" * 70)
    print("ΣΥΓΚΡΙΣΗ: k-d / Quad / Range / R-tree + LSH")
    print("=" * 70)

    # Υποψήφιες στήλες που χρειάζονται
    title_candidates = ["title", "original_title", "movie_title"]
    year_candidates = ["release_year", "year", "release_date"]
    popularity_candidates = ["popularity"]
    vote_candidates = ["vote_average", "vote_avg"]
    runtime_candidates = ["runtime", "duration"]
    language_candidates = ["original_language", "language"]
    country_candidates = ["origin_country", "origin_countries", "production_countries", "country"]
    text_candidates = [text_attribute, "production_companies", "production_company_names", "genres", "genre_names"]

    numeric_candidates = ["budget", "revenue", "runtime", "popularity", "vote_average"]
    candidate_set = set(
        title_candidates
        + year_candidates
        + popularity_candidates
        + vote_candidates
        + runtime_candidates
        + language_candidates
        + country_candidates
        + text_candidates
        + numeric_candidates
    )

    df = _load_dataset(columns=lambda c: c in candidate_set)

    title_col = _resolve_column(df, title_candidates, "τίτλο")
    year_col = None
    for candidate in year_candidates:
        if candidate in df.columns:
            year_col = candidate
            break
    if year_col is None:
        raise ValueError("Δεν βρέθηκε στήλη για έτος/ημερομηνία κυκλοφορίας.")

    popularity_col = _resolve_column(df, popularity_candidates, "popularity")
    vote_col = _resolve_column(df, vote_candidates, "vote_average")
    runtime_col = _resolve_column(df, runtime_candidates, "runtime")
    language_col = _resolve_column(df, language_candidates, "original_language")
    country_col = _resolve_column(df, country_candidates, "origin_country")
    text_col = _resolve_column(df, text_candidates, "textual attribute")

    year_series = _extract_year(df[year_col])
    df = df.copy()
    df["__year__"] = year_series

    df = df[
        (df["__year__"].between(year_range[0], year_range[1], inclusive="both"))
        & (pd.to_numeric(df[popularity_col], errors="coerce").between(*popularity_range, inclusive="both"))
        & (pd.to_numeric(df[vote_col], errors="coerce").between(*vote_average_range, inclusive="both"))
        & (pd.to_numeric(df[runtime_col], errors="coerce").between(*runtime_range, inclusive="both"))
        & (df[language_col].astype(str).str.lower() == str(original_language).lower())
    ]

    df = df[df[country_col].apply(lambda v: _contains_any_country(v, origin_countries))]

    if df.empty:
        raise ValueError("Δεν βρέθηκαν ταινίες με τα συγκεκριμένα φίλτρα.")

    df = df.reset_index(drop=True)

    # Prepare 5D normalized data
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    if len(numeric_cols) == 0:
        raise ValueError("Δεν βρέθηκαν αριθμητικές στήλες για indexing.")
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]

    data_5d = df[numeric_cols].values
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_5d)
    dims = data_norm.shape[1]

    # LSH build (once)
    lsh, minhashes, lsh_build_time = build_lsh_index(df, text_col, threshold=lsh_threshold, num_perm=lsh_num_perm)

    # Επιλογή query indices για εξαντλητική αξιολόγηση (δειγματοληψία)
    rng = np.random.default_rng(random_seed)
    num_queries = min(num_queries, len(df))
    query_indices = rng.choice(len(df), size=num_queries, replace=False)

    results = []

    # k-d tree
    lsh_times = []
    for qi in query_indices:
        _, lsh_q = query_lsh(lsh, minhashes, query_idx=int(qi))
        lsh_times.append(lsh_q)
    lsh_query_time = float(np.mean(lsh_times))

    kd_metrics = benchmark_kd_tree(
        data_norm,
        top_k=top_k,
        range_radius=range_radius,
        query_indices=query_indices,
    )
    build_time = kd_metrics["build_time"]
    knn_time = kd_metrics["knn_time"]
    range_time = kd_metrics["range_time"]
    results.append({
        "name": "k-d Tree + LSH",
        "build_time": build_time,
        "knn_time": knn_time,
        "range_time": range_time,
        "lsh_build_time": lsh_build_time,
        "lsh_query_time": lsh_query_time,
    })

    # Quad tree
    start = time.time()
    bounds = [(0, 1)] * dims
    quad_tree = QuadTreeNode(bounds, max_depth=8, max_points=100, dim=dims)
    for i, point in enumerate(data_norm):
        quad_tree.insert(i, point)
    build_time = time.time() - start
    knn_times = []
    range_times = []
    for qi in query_indices:
        query = data_norm[qi]
        query_bounds = [(q - range_radius, q + range_radius) for q in query]
        start = time.time()
        quad_tree.range_query(query_bounds)
        range_times.append(time.time() - start)

        start = time.time()
        small_bounds = [(q - 0.1, q + 0.1) for q in query]
        candidates = quad_tree.range_query(small_bounds)
        if candidates:
            sample_candidates = candidates[:min(1000, len(candidates))]
            distances = [np.linalg.norm(data_norm[idx] - query) for idx in sample_candidates]
            np.argsort(distances)[:top_k]
        knn_times.append(time.time() - start)

    knn_time = float(np.mean(knn_times))
    range_time = float(np.mean(range_times))
    results.append({
        "name": "Quad Tree + LSH",
        "build_time": build_time,
        "knn_time": knn_time,
        "range_time": range_time,
        "lsh_build_time": lsh_build_time,
        "lsh_query_time": lsh_query_time,
    })

    # Range tree
    start = time.time()
    range_tree = RangeTree5D(data_norm, dims=dims)
    build_time = time.time() - start
    knn_times = []
    range_times = []
    for qi in query_indices:
        query = data_norm[qi]
        query_bounds = [(q - range_radius, q + range_radius) for q in query]
        start = time.time()
        range_tree.range_query(query_bounds)
        range_times.append(time.time() - start)

        start = time.time()
        small_bounds = [(q - 0.1, q + 0.1) for q in query]
        candidates = range_tree.range_query(small_bounds)
        if candidates:
            sample_candidates = candidates[:min(1000, len(candidates))]
            distances = [np.linalg.norm(data_norm[idx] - query) for idx in sample_candidates]
            np.argsort(distances)[:top_k]
        knn_times.append(time.time() - start)

    knn_time = float(np.mean(knn_times))
    range_time = float(np.mean(range_times))
    results.append({
        "name": "Range Tree + LSH",
        "build_time": build_time,
        "knn_time": knn_time,
        "range_time": range_time,
        "lsh_build_time": lsh_build_time,
        "lsh_query_time": lsh_query_time,
    })

    # R-tree
    rtree_metrics = benchmark_rtree(
        data_norm,
        top_k=top_k,
        range_radius=range_radius,
        query_indices=query_indices,
    )
    build_time = rtree_metrics["build_time"]
    knn_time = rtree_metrics["knn_time"]
    range_time = rtree_metrics["range_time"]
    results.append({
        "name": "R-tree + LSH",
        "build_time": build_time,
        "knn_time": knn_time,
        "range_time": range_time,
        "lsh_build_time": lsh_build_time,
        "lsh_query_time": lsh_query_time,
    })

    df_results = pd.DataFrame(results)
    pd.options.display.float_format = "{:.4f}".format
    print("\n" + "=" * 70)
    print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ - SPATIAL + LSH")
    print("=" * 70)
    print(df_results.to_string(index=False))

    return df_results


if __name__ == "__main__":
    combined_test_with_lsh(text_attribute="production_company_names")