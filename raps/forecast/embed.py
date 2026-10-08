"""
Identity embeddings for jobs: all-MiniLM-L6-v2 over "user,job name,account", the same recipe F-Data
uses for Fugaku (usr, jnam, jobenv_req merged into one string). Each unique string is encoded once.

The vectors are an identity fingerprint (a user's recurring jobs map to near-identical vectors),
so they stay in the private data directory with the raw sacct data.

    python -m raps.forecast.embed --jobs $D/jobs.parquet --out $D/embeddings.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def identity_string(df):
    return df.user.astype(str) + "," + df.name.astype(str) + "," + df.account.astype(str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    from sentence_transformers import SentenceTransformer

    jobs = pd.read_parquet(a.jobs, columns=["user", "name", "account"])
    texts = identity_string(jobs).unique()
    print(f"{len(texts):,} unique identity strings from {len(jobs):,} jobs", flush=True)
    model = SentenceTransformer(MODEL, device="cpu")
    t = time.time()
    vecs = model.encode(list(texts), batch_size=a.batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
    print(f"encoded in {time.time() - t:.0f}s, dim {vecs.shape[1]}", flush=True)
    out = pd.DataFrame(vecs, columns=[f"e{i}" for i in range(vecs.shape[1])])
    out.insert(0, "identity", texts)
    out.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
