import hashlib, os
from ..config import UPLOAD_DIR, CACHE_DIR

def file_path_in_uploads(name: str) -> str:
    return os.path.join(UPLOAD_DIR, name)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def cache_summary_path(file_hash: str) -> str:
    return os.path.join(CACHE_DIR, f"{file_hash}.summary.json")
