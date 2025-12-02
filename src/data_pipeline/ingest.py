# src/data_pipeline/ingest.py
from pathlib import Path
import pandas as pd
import pickle
import joblib
import json
import sqlite3
from typing import Optional, Dict, Any

SUPPORTED_FORMATS = [
    ".csv", ".xlsx", ".xls", ".parquet", ".feather",
    ".pkl", ".pickle", ".joblib", ".json", ".html",
    ".sql", ".sqlite"
]

def _convert_to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise ValueError("Cannot convert loaded object to DataFrame")

def ingest_any(path: str, sql_table: Optional[str] = None, sql_query: Optional[str] = None) -> pd.DataFrame:
    """
    Load a dataset from many file types and return a pandas DataFrame.
    Supported: csv, xlsx, parquet, feather, pkl/pickle, joblib, json, html, sqlite/sql.
    For SQL use either sql_table (reads whole table) or sql_query (custom).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = p.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {ext}. Supported: {SUPPORTED_FORMATS}")

    if ext == ".csv":
        return pd.read_csv(p)
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(p)
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".feather":
        return pd.read_feather(p)
    if ext in [".pkl", ".pickle"]:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        return _convert_to_df(obj)
    if ext == ".joblib":
        obj = joblib.load(p)
        return _convert_to_df(obj)
    if ext == ".json":
        # try to read as records first
        try:
            return pd.read_json(p, orient="records", lines=True)
        except Exception:
            return pd.read_json(p)
    if ext == ".html":
        # returns list of tables; pick the first
        tables = pd.read_html(p)
        if len(tables) == 0:
            raise ValueError("No tables found in HTML file")
        return tables[0]
    if ext in [".sql", ".sqlite"]:
        # For sqlite file: provide table name or query
        conn = sqlite3.connect(str(p))
        if sql_query:
            df = pd.read_sql_query(sql_query, conn)
        elif sql_table:
            df = pd.read_sql_query(f"SELECT * FROM {sql_table}", conn)
        else:
            # list tables and pick first if only one
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            if not tables:
                raise ValueError("No tables found in sqlite file and no table/query provided")
            df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)
        conn.close()
        return df

    raise ValueError("Unhandled file extension")

def basic_cleaning(df: pd.DataFrame, convert_numeric: bool = True, drop_duplicates: bool = True,
                   fill_na_with: Optional[Dict[str, Any]] = None, sample_info: bool = False) -> pd.DataFrame:
    """
    Basic cleaning steps:
      - convert columns that look numeric to numeric (pd.to_numeric with coercion)
      - optionally drop duplicates
      - optionally fill NA for specified columns (dict column->value)
    Returns cleaned DataFrame.
    """
    df = df.copy()

    if convert_numeric:
        for col in df.columns:
            # attempt conversion if dtype is object
            if df[col].dtype == "object":
                # strip spaces and common noise
                df[col] = df[col].astype(str).str.strip()
                # try converting
                coerced = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
                # replace only if many values converted
                non_na = coerced.notna().sum()
                if non_na > 0:
                    df[col] = coerced

    if drop_duplicates:
        df = df.drop_duplicates()

    if fill_na_with:
        df = df.fillna(value=fill_na_with)

    if sample_info:
        print("[INGEST CLEAN] shape:", df.shape)
        print("[INGEST CLEAN] dtypes:\n", df.dtypes)
        print("[INGEST CLEAN] head:\n", df.head(3))

    return df

# Convenience function: ingest then clean then save raw copy
def ingest_and_prepare(path: str, out_raw_dir: str = "data/raw", **clean_kwargs):
    df = ingest_any(path)
    df = basic_cleaning(df, **clean_kwargs)
    out_dir = Path(out_raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingested_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[INGEST] Loaded data from {path}")
    print(f"[INGEST] Saved cleaned raw copy to {out_path}")
    return df
