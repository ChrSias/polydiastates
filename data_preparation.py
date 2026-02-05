import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def _load_dataset(columns=None):
    """
    Φόρτωση dataset από csv ή xlsx με προαιρετική επιλογή στηλών.

    Args:
        columns: Λίστα στηλών που θα φορτωθούν (None = όλες)

    Returns:
        df: DataFrame
    """
    print("Φόρτωση δεδομένων...")
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data_movies_clean.csv"
    xlsx_path = base_dir / "data_movies_clean.csv.xlsx"

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace", usecols=columns)
        except TypeError:
            return pd.read_csv(csv_path, encoding="utf-8", usecols=columns)

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, usecols=columns, engine="openpyxl")
        # Cache σε CSV για πιο γρήγορα επόμενα runs
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
        except Exception:
            pass
        return df

    raise FileNotFoundError(
        "Δεν βρέθηκε αρχείο δεδομένων. Αναμενόταν 'data_movies_clean.csv' ή "
        "'data_movies_clean.csv.xlsx' στο φάκελο του project."
    )


def load_and_prepare_data():
    """
    Φορτώνει το movies dataset και το ετοιμάζει για indexing.
    
    Returns:
        df: Το πλήρες DataFrame με όλες τις στήλες
        data_normalized: Τα 5D normalized δεδομένα (946460 x 5)
        scaler: Το MinMaxScaler object
    """
    
    # Οι 5 στήλες για indexing
    numerical_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']
    required_cols = ['title'] + numerical_cols

    # Φόρτωση μόνο των απαιτούμενων στηλών
    df = _load_dataset(columns=required_cols)
    
    # Πάρε μόνο αυτές τις στήλες
    data_5d = df[numerical_cols].values
    
    # Κανονικοποίηση στο [0, 1]
    print("Κανονικοποίηση δεδομένων...")
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_5d)
    
    print(f"✓ Φορτώθηκαν {len(df)} ταινίες")
    print(f"✓ Normalized data: {data_normalized.shape}")
    
    return df, data_normalized, scaler


# Αν τρέχεις απευθείας αυτό το αρχείο, κάνε ένα test
if __name__ == "__main__":
    df, data_norm, scaler = load_and_prepare_data()
    
    print("\n=== TEST ===")
    print(f"Πρώτη ταινία: {df.iloc[0]['title']}")
    print(f"Original values: {df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average']].iloc[0].values}")
    print(f"Normalized values: {data_norm[0]}")