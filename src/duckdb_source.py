"""DuckDB-over-Parquet source for PIDSMaker (drop-in for the postgres reads).

PIDSMaker reads its graph from a postgres database whose tables are created by
`postgres/init-create-databases.sh`. We already have those tables exported as
parquet (per dataset, e.g. cadets_e3) — locally or in Cloudflare R2 — so instead
of standing up postgres we register the parquet as DuckDB views that replicate
the postgres schema EXACTLY (column names, order, and types), then hand back a
connection whose cursor API matches psycopg2's. Every existing `SELECT *` +
positional-unpack site in the pipeline then works unchanged.

Activate by setting PIDS_DUCKDB=1. Source of the parquet:
  - PIDS_PARQUET_DIR=/path/to/dir   -> local dir holding events_*.parquet +
                                       {subject,file,netflow}_node_table.parquet
  - else R2: s3://$PIDS_R2_BUCKET/<database>/...  (needs AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, R2_ENDPOINT; bucket defaults to $R2_BUCKET_NAME or
    'vaccine'). <database> is cfg.dataset.database (e.g. cadets_e3).

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
        name = f"events_*.parquet" if glob else f"{table}.parquet"
        return os.path.join(local, name)
    bucket = os.environ.get("PIDS_R2_BUCKET") or os.environ.get("R2_BUCKET_NAME") or "vaccine"
    name = "events_*.parquet" if glob else f"{table}.parquet"
    return f"s3://{bucket}/{database}/{name}"


def duckdb_connection(database):
    """Return a psycopg2-cursor-shaped DuckDB connection with the 4 views."""
    con = duckdb.connect(":memory:")
    if not os.environ.get("PIDS_PARQUET_DIR"):
        # R2 / S3-compatible source
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(
            "CREATE SECRET r2 (TYPE S3, KEY_ID ?, SECRET ?, ENDPOINT ?, "
            "URL_STYLE 'path', USE_SSL true, REGION 'auto');",
            [os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"],
             os.environ["R2_ENDPOINT"]],
        )

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
