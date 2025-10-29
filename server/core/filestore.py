import os, pandas as pd
from ..config import UPLOAD_DIR

def save_file_bytes(file_bytes: bytes, filename: str) -> str:
    name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, name)
    # If clashes matter, you can uniquify; keeping simple to mirror your style:
    with open(path, "wb") as f:
        f.write(file_bytes)
    return name  # return file_id == filename

def csv_path(file_id_or_name: str) -> str:
    return os.path.join(UPLOAD_DIR, file_id_or_name)

def load_df(file_id_or_name: str) -> pd.DataFrame:
    return pd.read_csv(csv_path(file_id_or_name))
