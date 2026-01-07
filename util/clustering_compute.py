"""
UMAP and HDBSCAN clustering computation for Democracy Viewer datasets.
"""

import datetime as dt
import hashlib
import json
import numpy as np
import pandas as pd
import polars as pl
from time import time
from tqdm import tqdm
import humanize
import os

# Clustering libraries
import umap
import hdbscan

# Database interaction
from util.s3 import upload_file, BASE_PATH
from util.sql_connect import sql_connect
import util.sql_queries as sql
from util.embeddings_save import load_data_from_pkl


def hash_params(params: dict) -> str:
    """Create MD5 hash of parameters for caching."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def compute_umap(embedding_matrix: np.ndarray, n_neighbors: int = 15, 
                min_dist: float = 0.1, metric: str = "cosine",
                n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Compute UMAP dimensionality reduction."""
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    umap_coords = reducer.fit_transform(embedding_matrix)
    print(f"✓ UMAP complete: {umap_coords.shape}")
    
    return umap_coords


def compute_hdbscan(embedding_matrix: np.ndarray, min_cluster_size: int = 5,
                   min_samples: int = 3, metric: str = "euclidean",
                   cluster_selection_method: str = "eom") -> tuple:
    """Compute HDBSCAN clustering."""
    print(f"Computing HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(embedding_matrix)
    probabilities = clusterer.probabilities_
    
    # Print cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"✓ Found {n_clusters} clusters and {n_noise} noise points")
    
    return cluster_labels, probabilities


def save_clustering_results(table_name: str, record_ids: list, umap_coords: np.ndarray,
                           cluster_labels: np.ndarray, probabilities: np.ndarray,
                           umap_params: dict, hdbscan_params: dict,
                           embed_col: str = None, embed_value: str = None):
    """Save combined UMAP and HDBSCAN results."""
    
    umap_hash = hash_params(umap_params)
    hdbscan_hash = hash_params(hdbscan_params)
    
    # Create combined DataFrame
    df = pl.DataFrame({
        "record_id": record_ids,
        "x": umap_coords[:, 0],
        "y": umap_coords[:, 1],
        "cluster_id": cluster_labels,
        "probability": probabilities
    })
    
    # Ensure local directory exists
    local_folder = f"clustering_{table_name}"
    os.makedirs(f"{BASE_PATH}/{local_folder}", exist_ok=True)
    
    # Save combined results
    combined_filename = f"clustering_{umap_hash}_{hdbscan_hash}.parquet"
    local_path = f"{BASE_PATH}/{local_folder}/{combined_filename}"
    df.write_parquet(local_path)
    
    # Upload to S3
    s3_folder = f"tables/clustering_{table_name}"
    upload_file(local_folder, s3_folder, combined_filename, None)
    
    # Register in database
    engine, meta = sql_connect()
    s3_path = f"{s3_folder}/{combined_filename}"
    
    # Register both UMAP and HDBSCAN separately for querying
    sql.register_clustering_result(
        engine, table_name, "umap", umap_hash, s3_path,
        embed_col, embed_value, umap_params
    )
    
    sql.register_clustering_result(
        engine, table_name, "hdbscan", hdbscan_hash, s3_path,
        embed_col, embed_value, hdbscan_params
    )
    
    print(f"✓ Clustering results saved to {s3_path}")


def get_document_embeddings(table_name: str, embed_col: str = None, 
                           embed_value: str = None) -> tuple:
    """
    Get document-level embeddings by averaging word vectors per document.
    
    Returns:
        Tuple of (embedding_matrix, record_ids)
    """
    print("Loading Word2Vec model and computing document embeddings...")
    
    # Load the appropriate Word2Vec model
    if embed_col and embed_value:
        model_name = f"model_{embed_col}_{embed_value}"
    else:
        model_name = "model"
    
    try:
        model = load_data_from_pkl(table_name, model_name, None)
    except Exception as e:
        raise Exception(f"Failed to load Word2Vec model '{model_name}': {str(e)}")
    
    # Load token data from S3 to get document-word mappings
    from util.data_queries import get_tokens_for_clustering
    
    try:
        df_tokens = get_tokens_for_clustering(table_name, embed_col, embed_value)
    except Exception as e:
        raise Exception(f"Failed to load token data: {str(e)}")
    
    # Compute document vectors by averaging word vectors
    print("Computing document vectors...")
    doc_vectors = {}
    
    for record_id in tqdm(df_tokens['record_id'].unique()):
        words = df_tokens[df_tokens['record_id'] == record_id]['word'].tolist()
        
        # Get vectors for words that exist in model
        word_vecs = []
        for word in words:
            if word in model.wv:
                word_vecs.append(model.wv[word])
        
        if len(word_vecs) > 0:
            # Average word vectors to get document vector
            doc_vectors[record_id] = np.mean(word_vecs, axis=0)
    
    # Convert to matrix
    record_ids = list(doc_vectors.keys())
    embedding_matrix = np.array([doc_vectors[rid] for rid in record_ids])
    
    print(f"✓ Computed embeddings for {len(record_ids)} documents")
    
    return embedding_matrix, record_ids


def compute_clustering_for_group(table_name: str, embed_col: str, embed_value: str,
                                 umap_params: dict, hdbscan_params: dict):
    """Compute clustering for a specific group."""
    print(f"\nProcessing group: {embed_col}={embed_value}")
    
    try:
        # Get embeddings
        embedding_matrix, record_ids = get_document_embeddings(
            table_name, embed_col, embed_value
        )
        
        # Compute UMAP
        umap_coords = compute_umap(embedding_matrix, **umap_params)
        
        # Compute HDBSCAN on UMAP coordinates
        cluster_labels, probabilities = compute_hdbscan(umap_coords, **hdbscan_params)
        
        # Save results
        save_clustering_results(
            table_name, record_ids, umap_coords, cluster_labels, probabilities,
            umap_params, hdbscan_params, embed_col, str(embed_value)
        )
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing group {embed_col}={embed_value}: {str(e)}")
        return False


def start_clustering(table_name: str, embed_cols: list, num_threads: int, 
                    clustering_params: dict):
    """
    Main entry point for clustering computation.
    
    Args:
        table_name: Dataset table name
        embed_cols: List of grouping columns (if empty, compute globally)
        num_threads: Number of threads
        clustering_params: Dict with UMAP and HDBSCAN parameters
    """
    print("\n" + "="*80)
    print("STARTING UMAP/HDBSCAN CLUSTERING")
    print("="*80)
    
    start_time = time()
    
    # Extract parameters
    umap_params = {
        "n_neighbors": clustering_params.get("umap_n_neighbors", 15),
        "min_dist": clustering_params.get("umap_min_dist", 0.1),
        "metric": "cosine",
        "n_components": 2,
        "random_state": 42
    }
    
    hdbscan_params = {
        "min_cluster_size": clustering_params.get("hdbscan_min_cluster_size", 5),
        "min_samples": clustering_params.get("hdbscan_min_samples", 3),
        "metric": "euclidean",
        "cluster_selection_method": "eom"
    }
    
    print(f"UMAP params: {umap_params}")
    print(f"HDBSCAN params: {hdbscan_params}")
    
    try:
        if embed_cols and len(embed_cols) > 0:
            # Compute per group
            print(f"\nComputing clustering per group: {embed_cols}")
            
            # Get unique values for the first embed column
            embed_col = embed_cols[0]
            from util.data_queries import get_unique_embed_values
            
            try:
                unique_values = get_unique_embed_values(table_name, embed_col)
                print(f"Found {len(unique_values)} unique values in {embed_col}")
                
                success_count = 0
                for embed_value in unique_values:
                    if compute_clustering_for_group(
                        table_name, embed_col, embed_value,
                        umap_params, hdbscan_params
                    ):
                        success_count += 1
                
                print(f"\n✓ Completed clustering for {success_count}/{len(unique_values)} groups")
                
            except Exception as e:
                print(f"Error getting unique values: {str(e)}")
                raise
        else:
            # Compute globally
            print("\nComputing global clustering...")
            
            # Get embeddings
            embedding_matrix, record_ids = get_document_embeddings(table_name)
            
            # Compute UMAP
            umap_coords = compute_umap(embedding_matrix, **umap_params)
            
            # Compute HDBSCAN
            cluster_labels, probabilities = compute_hdbscan(umap_coords, **hdbscan_params)
            
            # Save results
            save_clustering_results(
                table_name, record_ids, umap_coords, cluster_labels, probabilities,
                umap_params, hdbscan_params
            )
        
        elapsed = time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ CLUSTERING COMPLETE: {humanize.precisedelta(dt.timedelta(seconds=elapsed))}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        elapsed = time() - start_time
        print(f"\n{'='*80}")
        print(f"✗ CLUSTERING FAILED after {humanize.precisedelta(dt.timedelta(seconds=elapsed))}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")
        # Don't raise - allow pipeline to continue
