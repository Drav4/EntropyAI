import re
from typing import List, Dict

# Matches:
#  [Attachments]
#  #1: test.csv → /files/abcd123.csv (id:abcd123.csv)
#  or any /files/<id.ext> without the (id:...) tail.
_PATTERN = re.compile(
    r"(?:#\d+\s*:\s*)?(?P<name>[\w\-.]+\.(?:csv|xlsx|xls|parquet|txt))"
    r".*?(?:\(id:(?P<id1>[\w\-.]+)\)|/files/(?P<id2>[\w\-.]+))",
    re.IGNORECASE | re.DOTALL,
)

def extract_attachment_candidates(text: str | None) -> List[Dict[str, str]]:
    if not text:
        return []
    out: List[Dict[str, str]] = []
    for m in _PATTERN.finditer(text):
        name = m.group("name")
        fid = m.group("id1") or m.group("id2")
        if name and fid:
            out.append({"name": name, "id": fid})
    return out
