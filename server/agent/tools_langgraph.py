from typing import Optional
from langchain_core.tools import tool

from ..services.datasets import (
    tool_get_dataset_summary,
    tool_run_pandas_op,
    tool_plot_histogram,
    tool_compute_dataset_facts,
)

@tool("list_datasets", return_direct=False)
def list_datasets() -> dict:
    """List files in uploads directory."""
    import os
    from ..config import UPLOAD_DIR
    items = [{"id": f, "size": os.path.getsize(os.path.join(UPLOAD_DIR, f))}
             for f in os.listdir(UPLOAD_DIR)
             if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and not f.startswith("plots")]
    return {"datasets": items}

@tool("get_dataset_summary", return_direct=False)
def get_dataset_summary(file_id_or_name: str) -> dict:
    """Return cached (or compute) dataset summary."""
    return tool_get_dataset_summary(file_id_or_name)

@tool("run_pandas_op", return_direct=False)
def run_pandas_op(file_id_or_name: str, op: str, column: Optional[str] = None, top_n: int = 10) -> dict:
    """Run a safe pandas op on the dataset."""
    return tool_run_pandas_op(file_id_or_name, op, column, top_n)

@tool("plot_histogram", return_direct=False)
def plot_histogram(file_id_or_name: str, column: str, bins: int = 30) -> dict:
    """Create a histogram PNG for a numeric column; returns /files URL."""
    return tool_plot_histogram(file_id_or_name, column, bins)

@tool("compute_dataset_facts", return_direct=False)
def compute_dataset_facts(file_id_or_name: str) -> dict:
    """Compute descriptive stats and technical indicators for dataset."""
    return tool_compute_dataset_facts(file_id_or_name)

TOOLS = [list_datasets, get_dataset_summary, run_pandas_op, plot_histogram, compute_dataset_facts]
