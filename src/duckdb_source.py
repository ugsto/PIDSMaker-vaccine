"""DuckDB-over-Parquet source for PIDSMaker (drop-in for the postgres reads).

PIDSMaker reads its graph from a postgres database whose tables are created by
`postgres/init-create-databases.sh`. We already have those tables exported as
parquet (per dataset, e.g. cadets_e3) — locally or in S3-compatible storage — so
instead of standing up postgres we register the parquet as DuckDB views that replicate
the postgres schema EXACTLY (column names, order, and types), then hand back a
connection whose cursor API matches psycopg2's. Every existing `SELECT *` +
positional-unpack site in the pipeline then works unchanged.

Activate by setting PIDS_DUCKDB=1. Source of the parquet:
  - PIDS_PARQUET_DIR=/path/to/dir   -> local dir holding events_*.parquet +
                                       {subject,file,netflow}_node_table.parquet
  - else S3: s3://$PIDS_S3_BUCKET/<database>/...  (needs AWS_ENDPOINT_URL and
    standard AWS credentials from the environment or an AWS profile; optional
    PIDS_S3_PREFIX is inserted before <database>). <database> is
    cfg.dataset.database (e.g. cadets_e3).

The postgres DDL layouts we replicate (init-create-databases.sh):
  event_table       (src_node, src_index_id, operation, dst_node, dst_index_id,
                     event_uuid, timestamp_rec, _id)
  file_node_table   (node_uuid, hash_id, path, index_id)
  netflow_node_table(node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port,
                     index_id)
  subject_node_table(node_uuid, hash_id, path, cmd, index_id)
Our parquet carries the same columns in a different order (and the event parquet
omits src_node/dst_node/_id); the views below normalize to the DDL. Event index
ids are cast to VARCHAR to match the postgres DDL (src/dst_index_id VARCHAR),
while node index_id stays BIGINT — exactly the postgres type split the pipeline
was written against.
"""

import os
from urllib.parse import urlparse

import duckdb


class _DuckCursor:
    """psycopg2-cursor-shaped wrapper over a DuckDB connection.

    The pipeline uses only .execute / .fetchall / .fetchmany / .fetchone /
    .close / .cursor / iteration and treats the same object as both cursor and
    connection. DuckDB is single-connection here; reads are sequential, so one
    shared underlying connection is safe.
    """

    def __init__(self, con):
        self._con = con

    def execute(self, sql, params=None):
        # psycopg2 uses %s placeholders; the pipeline's reads interpolate the
        # SQL themselves (no bound params), so we pass through as-is.
        self._con.execute(sql, params) if params else self._con.execute(sql)
        return self

    def fetchall(self):
        return self._con.fetchall()

    def fetchmany(self, size=1):
        return self._con.fetchmany(size)

    def fetchone(self):
        return self._con.fetchone()

    def cursor(self):
        return self

    def commit(self):
        pass

    def close(self):
        pass

    def __iter__(self):
        return iter(self._con.fetchall())


def _parquet_uri(database, table, glob=False):
    local = os.environ.get("PIDS_PARQUET_DIR")
    if local:
        name = "events_*.parquet" if glob else f"{table}.parquet"
        return os.path.join(local, name)
    bucket = os.environ.get("PIDS_S3_BUCKET")
    if not bucket:
        raise ValueError("PIDS_S3_BUCKET must be set when using S3 parquet input")

    prefix = os.environ.get("PIDS_S3_PREFIX", "").strip("/")
    path_parts = [part for part in (prefix, database) if part]
    name = "events_*.parquet" if glob else f"{table}.parquet"
    return f"s3://{bucket}/{'/'.join(path_parts)}/{name}"


def _duckdb_s3_endpoint(endpoint_url):
    """Normalize an S3-compatible endpoint for DuckDB httpfs."""
    if not endpoint_url:
        raise ValueError("AWS_ENDPOINT_URL must be set when using S3 parquet input")

    parsed = urlparse(endpoint_url)
    if parsed.scheme:
        endpoint = parsed.netloc + parsed.path
        use_ssl = parsed.scheme != "http"
    else:
        parsed = urlparse(f"//{endpoint_url}")
        endpoint = parsed.netloc + parsed.path
        use_ssl = True

    endpoint = endpoint.rstrip("/")
    if not endpoint:
        raise ValueError(f"Invalid AWS_ENDPOINT_URL: {endpoint_url!r}")

    return endpoint, use_ssl


def duckdb_connection(database):
    """Return a psycopg2-cursor-shaped DuckDB connection with the 4 views."""
    con = duckdb.connect(":memory:")
    if not os.environ.get("PIDS_PARQUET_DIR"):
        # S3-compatible source
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws;")
        endpoint, use_ssl = _duckdb_s3_endpoint(os.environ.get("AWS_ENDPOINT_URL"))
        profile = os.environ.get("AWS_PROFILE") or os.environ.get("AWS_DEFAULT_PROFILE")
        use_ssl_sql = "true" if use_ssl else "false"
        secret_sql = (
            "CREATE SECRET pids_s3 (TYPE S3, PROVIDER credential_chain, "
            "CHAIN 'env;config', "
        )
        secret_params = []
        if profile:
            secret_sql += "PROFILE ?, "
            secret_params.append(profile)
        secret_sql += f"ENDPOINT ?, URL_STYLE 'path', USE_SSL {use_ssl_sql}, REGION ?);"
        secret_params.extend([endpoint, os.environ.get("AWS_DEFAULT_REGION", "auto")])
        con.execute(secret_sql, secret_params)

    ev = _parquet_uri(database, "events", glob=True)
    subj = _parquet_uri(database, "subject_node_table")
    filet = _parquet_uri(database, "file_node_table")
    net = _parquet_uri(database, "netflow_node_table")

    con.execute(f"""
        CREATE VIEW event_table AS
        SELECT CAST(NULL AS VARCHAR)        AS src_node,
               CAST(src_index_id AS VARCHAR) AS src_index_id,
               operation,
               CAST(NULL AS VARCHAR)        AS dst_node,
               CAST(dst_index_id AS VARCHAR) AS dst_index_id,
               event_uuid,
               timestamp_rec,
               row_number() OVER ()          AS _id
        FROM read_parquet('{ev}');
    """)
    con.execute(f"""
        CREATE VIEW subject_node_table AS
        SELECT node_uuid, hash_id, path, cmd, index_id
        FROM read_parquet('{subj}');
    """)
    con.execute(f"""
        CREATE VIEW file_node_table AS
        SELECT node_uuid, hash_id, path, index_id
        FROM read_parquet('{filet}');
    """)
    con.execute(f"""
        CREATE VIEW netflow_node_table AS
        SELECT node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id
        FROM read_parquet('{net}');
    """)
    return _DuckCursor(con)
