import polars as pl
from sqlalchemy import Engine
import util.s3 as s3
import util.sql_queries as sql

# Retrieve data from s3 and keep required data
def get_text(engine: Engine, table_name: str, batch: int | None = None, token: str | None = None) -> pl.LazyFrame:
    # Get all text columns
    text_cols = sql.get_text_cols(engine, table_name)
    
    # Download raw data from s3 -- Lambda always uploads parquet to datasets folder, regardless of batch number
    df_raw = s3.download("datasets", table_name, batch, token)

    # Reformat data to prep for preprocessing
    df_list: list[pl.LazyFrame] = []
    for col in text_cols:
        df_list.append((
            df_raw
                .select([col, "record_id"])
                .rename({ f"{col}": "text" })
                .with_columns(col=pl.lit(col))
                .cast({ "text": pl.Utf8 })
        ))
    df = pl.concat(df_list)
    
    return df

# Get the values of a subset of columns for each record
def get_columns(table_name: str, columns: list[str], batch: int | None = None, token: str | None = None) -> pl.LazyFrame:
    df = s3.download("datasets", table_name, batch, token)
    
    return df.select(["record_id"] + columns)

# Get tokens for clustering computation
def get_tokens_for_clustering(table_name: str, embed_col: str = None, embed_value: str = None, token: str | None = None):
    """Load tokens from S3 for clustering computation."""
    import pandas as pd
    
    # Download tokens from S3
    df_tokens = s3.download("tokens", table_name, None, token).collect().to_pandas()
    
    # Filter by embed_col if specified
    if embed_col and embed_value is not None:
        # Need to join with original dataset to get the embed_col values
        df_raw = s3.download("datasets", table_name, None, token).collect().to_pandas()
        
        # Filter raw data
        df_raw_filtered = df_raw[df_raw[embed_col] == embed_value][['record_id']]
        
        # Join to get only tokens for filtered records
        df_tokens = df_tokens.merge(df_raw_filtered, on='record_id')
    
    return df_tokens

# Get unique values for an embed column
def get_unique_embed_values(table_name: str, embed_col: str, token: str | None = None) -> list:
    """Get unique values for a grouping column."""
    df_raw = s3.download("datasets", table_name, None, token)
    unique_vals = df_raw.select(embed_col).unique().collect().to_series().to_list()
    return unique_vals
