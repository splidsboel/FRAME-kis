"""
frame.prep — the backend-neutral data-processing pipeline (Tier 1 -> Tier 2 in
[[Data pipeline and adapter load refactor]]).

Turns a raw, extracted V3C shard (keyframes/ info/ msb/ on disk) into the
canonical dataset (parquet + HDF5) that any adapter's load_data() ingests. The
V3C1 canonical set was produced by dumping the pgvector DB (scripts/export_v3c.py);
this package produces the SAME format for shards that were never loaded into pg
(V3C2, V3C3) by re-running the model passes and writing straight to files.

See frame.prep.common for the shared schema/constants + iteration/staging helpers.
"""
