from sqlalchemy import Column, Integer, String, Boolean, BigInteger, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

SQL_BASE = declarative_base()

class Users(SQL_BASE):
    __tablename__ = "users"
    email = Column("email", String(30), primary_key=True)
    password = Column("password", String(60))
    title = Column("title", String(20))
    first_name = Column("first_name", String(20))
    last_name = Column("last_name", String(20))
    suffix = Column("suffix", String(10))
    orcid = Column("orcid", String(16))
    linkedin_link = Column("linkedin_link", String(200))
    website = Column("website", String(200))

class DatasetMetadata(SQL_BASE):
    __tablename__ = "dataset_metadata"
    table_name = Column("table_name", String(100), primary_key = True)
    email = Column("email", String(30), ForeignKey(Users.email))
    title = Column("title", String(50))
    description = Column("description", String(200))
    author = Column("author", String(50))
    date_collected = Column("date_collected", Date)
    is_public = Column("is_public", Boolean)
    clicks = Column("clicks", Integer)
    preprocessing_type = Column("preprocessing_type", String(5))
    embeddings = Column("embeddings", Boolean)
    date_posted = Column("date_posted", Date)
    embed_col = Column("embed_col", String(50))
    language = Column("language", String(20))
    likes = Column("likes", Integer)
    embeddings_done = Column("embeddings_done", Boolean)
    tokens_done = Column("tokens_done", Boolean)
    distributed = Column("distributed", BigInteger)
    unprocessed_updates = Column("unprocessed_updates", Integer)
    uploaded = Column("uploaded", Boolean)
    num_records = Column("num_records", Integer)
    license = Column("license", String(200))
    reprocess_start = Column("reprocess_start", Boolean)
    num_batches = Column("num_batches", Integer)
    batches_done = Column("batches_done", Integer)
    
class Tags(SQL_BASE):
    __tablename__ = "tags"
    table_name = Column("table_name", String(100), ForeignKey(DatasetMetadata.table_name), primary_key = True)
    col = Column("tag_name", String(25), primary_key = True)

class DatasetTextCols(SQL_BASE):
    __tablename__ = "dataset_text_cols"
    table_name = Column("table_name", String(100), ForeignKey(DatasetMetadata.table_name), primary_key = True)
    col = Column("col", String(50), primary_key = True)
    
class DatasetEmbedCols(SQL_BASE):
    __tablename__ = "dataset_embed_cols"
    table_name = Column("table_name", String(100), ForeignKey(DatasetMetadata.table_name), primary_key = True)
    col = Column("col", String(50), primary_key = True)

class DatasetClusteringResults(SQL_BASE):
    __tablename__ = "dataset_clustering_results"
    table_name = Column("table_name", String(100), ForeignKey(DatasetMetadata.table_name), primary_key=True)
    method = Column("method", String(50), primary_key=True)
    param_hash = Column("param_hash", String(64), primary_key=True)
    embed_col = Column("embed_col", String(50))
    embed_value = Column("embed_value", String(100))
    s3_path = Column("s3_path", String(500))
    n_neighbors = Column("n_neighbors", Integer)
    min_dist = Column("min_dist", String(10))
    min_cluster_size = Column("min_cluster_size", Integer)
    min_samples = Column("min_samples", Integer)
    metric = Column("metric", String(20))
    status = Column("status", String(20))
    