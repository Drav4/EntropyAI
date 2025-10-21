import os, re
from typing import List, Optional
from ..config import UPLOAD_DIR

ATTACH_RE = re.compile(
    r"^\s*#\d+:\s*(?P<name>.+?)(?:\s*→\s*(?P<url>\S+))?(?:.*?\(id:(?P<id>[^)]+)\))?",
    re.IGNORECASE,
)

def extract_attachment_candidates(text: str) -> List[str]:
    if "[Attachments]" not in text:
        return []
    out: List[str] = []
    for line in text.splitlines():
        m = ATTACH_RE.search(line)
        if not m: 
            continue
        fid = (m.group("id") or "").strip()
        if fid:
            out.append(fid)
        else:
            name = (m.group("name") or "").strip()
            if name:
                out.append(name)
    return out

def resolve_to_upload_path(candidate: str) -> Optional[str]:
    direct = os.path.join(UPLOAD_DIR, candidate)
    if os.path.exists(direct):
        return direct
    base = os.path.splitext(candidate)[0]
    for saved in os.listdir(UPLOAD_DIR):
        if saved.startswith(base):
            return os.path.join(UPLOAD_DIR, saved)
    return None
