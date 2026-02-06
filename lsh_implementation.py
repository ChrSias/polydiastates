"""
LSH Implementation for Movie Similarity Search

This module implements a TWO-PHASE APPROACH for finding similar movies:

PHASE 1 (INDEXING):
    - Build MinHash signatures for each movie based on textual attributes
    - Insert signatures into LSH (Locality-Sensitive Hashing) data structure
    - This preprocessing step enables efficient similarity search

PHASE 2 (QUERYING):
    - Query the pre-built LSH index to find candidate similar items
    - Compute Jaccard similarity for ranking
    - Return top-N most similar movies

The separation of indexing and querying phases allows for:
    - One-time index construction with multiple queries
    - Efficient similarity search over large datasets
    - Clear performance benchmarking of each phase
"""
from datasketch import MinHash, MinHashLSH
import time
import json
import pandas as pd
from data_preparation import _load_dataset


def _resolve_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Λείπει στήλη για {label}. Διαθέσιμες στήλες: {list(df.columns)}"
    )


def _parse_tokens(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Προσπάθησε JSON list/dict
    if text.startswith("[") or text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                tokens = []
                for item in obj:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("iso_3166_1") or item.get("iso_639_1")
                        if name:
                            tokens.append(str(name).strip().lower())
                    else:
                        tokens.append(str(item).strip().lower())
                return [t for t in tokens if t]
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("iso_3166_1") or obj.get("iso_639_1")
                return [str(name).strip().lower()] if name else []
        except Exception:
            # προσπάθησε μετατροπή από Python list string με μονά εισαγωγικά
            if text.startswith("[") and "'" in text:
                try:
                    fixed = text.replace("'", '"')
                    obj = json.loads(fixed)
                    if isinstance(obj, list):
                        return [str(v).strip().lower() for v in obj if str(v).strip()]
                except Exception:
                    pass

    # Χώρισε με κοινά separators
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [t.strip().lower() for t in text.split(sep) if t.strip()]

    return [text.lower()]


def _extract_year(series):
    if series.dtype.kind in {"i", "u"}:
        return series
    return pd.to_datetime(series, errors="coerce").dt.year


def _contains_any_country(value, allowed_countries):
    tokens = _parse_tokens(value)
    tokens_upper = {t.upper() for t in tokens}
    return any(c.upper() in tokens_upper for c in allowed_countries)


def build_lsh_index(df, text_col, threshold=0.5, num_perm=128):
    """
    PHASE 1: Build LSH index for a textual attribute column and return timing.
    
    This is the indexing phase where we create MinHash signatures and insert them
    into the LSH data structure for efficient similarity search.
    
    Args:
        df: DataFrame with movie data
        text_col: Column name containing textual tokens
        threshold: Jaccard similarity threshold for LSH
        num_perm: Number of permutations for MinHash
        
    Returns:
        lsh: MinHashLSH index structure
        minhashes: List of MinHash objects for each movie
        build_time: Time taken to build the index
    """
    start = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    tokens_list = df[text_col].apply(_parse_tokens)
    for i, tokens in enumerate(tokens_list):
        m = MinHash(num_perm=num_perm)
        for t in tokens:
            m.update(t.encode("utf8"))
        lsh.insert(f"movie_{i}", m)
        minhashes.append(m)

    build_time = time.time() - start
    return lsh, minhashes, build_time


def query_lsh(lsh, minhashes, query_idx=0):
    """
    PHASE 2: Query LSH index and return timing and results.
    
    This is the querying phase where we use the pre-built LSH index to find
    similar items efficiently.
    
    Args:
        lsh: Pre-built MinHashLSH index
        minhashes: List of MinHash objects
        query_idx: Index of the query item
        
    Returns:
        result: List of similar item identifiers
        query_time: Time taken to query
    """
    start = time.time()
    result = lsh.query(minhashes[query_idx])
    query_time = time.time() - start
    return result, query_time


def filtered_lsh_top_n(
    text_attribute="production_company_names",
    year_range=(2000, 2020),
    popularity_range=(3, 6),
    vote_average_range=(3, 5),
    runtime_range=(30, 60),
    origin_countries=("US", "GB"),
    original_language="en",
    top_n=3,
    query_title=None,
    threshold=0.5,
    num_perm=128,
):
    """
    Complete pipeline: Filter movies by attributes, then find similar ones using two-phase LSH.
    
    Preprocessing: Filter movies by numerical/categorical attributes
    Phase 1 (Indexing): Build LSH index from filtered movie tokens
    Phase 2 (Querying): Query LSH to find similar movies and rank by Jaccard similarity
    
    Args:
        text_attribute: Textual column to use for similarity (e.g., "genre_names", "production_company_names")
        year_range: Tuple of (min_year, max_year) for filtering
        popularity_range: Tuple of (min_popularity, max_popularity) for filtering
        vote_average_range: Tuple of (min_vote, max_vote) for filtering
        runtime_range: Tuple of (min_runtime, max_runtime) for filtering in minutes
        origin_countries: Tuple of country codes to filter by (e.g., ("US", "GB"))
        original_language: Language code to filter by (e.g., "en")
        top_n: Number of most similar movies to return
        query_title: Optional title of query movie. If None, uses first movie in filtered set
        threshold: Jaccard similarity threshold for LSH (0.0 to 1.0)
        num_perm: Number of permutations for MinHash (higher = more accurate but slower)
        
    Returns:
        result: DataFrame with top N similar movies and their attributes
        query: Title of the query movie
    """
    # Υποψήφιες στήλες που χρειάζονται
    title_candidates = ["title", "original_title", "movie_title"]
    year_candidates = ["release_year", "year", "release_date"]
    popularity_candidates = ["popularity"]
    vote_candidates = ["vote_average", "vote_avg"]
    runtime_candidates = ["runtime", "duration"]
    language_candidates = ["original_language", "language"]
    country_candidates = ["origin_country", "origin_countries", "production_countries", "country"]
    text_candidates = [text_attribute, "production_companies", "production_company_names", "genres", "genre_names"]

    candidate_set = set(
        title_candidates
        + year_candidates
        + popularity_candidates
        + vote_candidates
        + runtime_candidates
        + language_candidates
        + country_candidates
        + text_candidates
    )

    # Φόρτωση μόνο των υποψήφιων στηλών για ταχύτητα
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

    # Φιλτράρισμα
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

    # Tokens για textual attribute
    df = df.reset_index(drop=True)
    tokens_list = df[text_col].apply(_parse_tokens)
    valid_mask = tokens_list.apply(len) > 0
    df = df[valid_mask].reset_index(drop=True)
    tokens_list = tokens_list[valid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("Οι ταινίες δεν έχουν tokens στο επιλεγμένο attribute.")

    # Επιλογή query ταινίας
    if query_title:
        matches = df[df[title_col].astype(str).str.lower() == str(query_title).lower()]
        if matches.empty:
            raise ValueError("Δεν βρέθηκε ταινία με τον συγκεκριμένο τίτλο στα φίλτρα.")
        query_idx = matches.index[0]
    else:
        query_idx = 0

    # ========== PHASE 1: INDEXING ==========
    # Build LSH index from movie tokens
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    for i, tokens in enumerate(tokens_list):
        m = MinHash(num_perm=num_perm)
        for t in tokens:
            m.update(t.encode("utf8"))
        lsh.insert(f"movie_{i}", m)
        minhashes.append(m)

    # ========== PHASE 2: LSH QUERYING ==========
    # Query LSH index to find similar movies
    candidates = lsh.query(minhashes[query_idx])
    candidate_indices = [int(c.split("_")[1]) for c in candidates if c != f"movie_{query_idx}"]
    if not candidate_indices:
        candidate_indices = [i for i in range(len(df)) if i != query_idx]

    # Jaccard για ranking
    query_tokens = set(tokens_list[query_idx])
    scored = []
    for idx in candidate_indices:
        cand_tokens = set(tokens_list[idx])
        if not cand_tokens:
            continue
        jaccard = len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)
        scored.append((idx, jaccard))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in scored[:top_n]]

    result = df.loc[top_indices, [title_col, text_col, popularity_col, vote_col, runtime_col, language_col, country_col, "__year__"]]
    result = result.copy()
    result["similarity"] = [score for _, score in scored[:top_n]]

    return result, df.loc[query_idx, title_col]


if __name__ == "__main__":
    result, query = filtered_lsh_top_n(text_attribute="genre_names", top_n=3)
    print(f"Query: {query}")
    print(result.to_string(index=False))