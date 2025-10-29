from __future__ import annotations
from typing import Dict, Any, List

def _fmt_metrics(metrics: List[Dict[str, Any]]) -> str:
    lines = []
    for m in metrics or []:
        name = m.get("name", "")
        val  = m.get("value", "")
        unit = m.get("unit")
        hc   = m.get("how_computed")
        line = f"- **{name}**: {val}" + (f" {unit}" if unit else "")
        if hc:
            line += f" _(how: {hc})_"
        lines.append(line)
    return "\n".join(lines)

def _fmt_list(label: str, xs: List[str]) -> str:
    if not xs:
        return ""
    body = "\n".join(f"- {x}" for x in xs)
    return f"### {label}\n{body}\n"

def _fmt_equations(eqs: List[str]) -> str:
    if not eqs:
        return ""
    # Keep it UI-friendly (inline or fenced). Many UIs render Markdown math.
    body = "\n".join(f"- `{e}`" for e in eqs)
    return f"### Equations / Definitions\n{body}\n"

def _fmt_repro(repro: Dict[str, Any]) -> str:
    if not repro:
        return ""
    calls = repro.get("tool_calls") or []
    results = repro.get("results") or []

    call_lines = []
    for c in calls:
        call_lines.append(
            f"- **{c.get('tool_name','')}** args={c.get('args',{})} ts={c.get('timestamp','')}"
        )
    res_lines = []
    for r in results:
        res_lines.append(
            f"- **{r.get('tool','')}** → ok={r.get('result',{}).get('ok', True)} ts={r.get('timestamp','')}"
        )

    parts = []
    if call_lines:
        parts.append("**Tool Calls**\n" + "\n".join(call_lines))
    if res_lines:
        parts.append("**Tool Results**\n" + "\n".join(res_lines))

    files = repro.get("files") or []
    if files:
        parts.append("**Files**\n" + "\n".join(f"- {f}" if isinstance(f, str) else f"- {f}" for f in files))

    if not parts:
        return ""
    return "### Reproducibility\n" + "\n\n".join(parts) + "\n"

def _fmt_references(refs: List[Dict[str, Any]]) -> str:
    if not refs:
        return ""
    lines = []
    for r in refs:
        title = r.get("title", "Reference")
        src   = r.get("source", "")
        url   = r.get("url_or_doi")
        acc   = r.get("accessed")
        tail = f" — {src}" if src else ""
        if url:
            tail += f" ({url})"
        if acc:
            tail += f", accessed {acc}"
        lines.append(f"- {title}{tail}")
    return "### References\n" + "\n".join(lines) + "\n"

def format_report_markdown(report: Dict[str, Any]) -> str:
    """
    Convert EvidenceReport (as dict) to a concise, technical Markdown reply.
    Safe to display directly in UI.
    """
    summary = report.get("summary", "").strip() or "No summary available."
    key_findings = report.get("key_findings") or []
    metrics = report.get("metrics") or []
    equations = report.get("equations") or []
    stats = report.get("statistical_tests") or []
    assumptions = report.get("assumptions") or []
    limitations = report.get("limitations") or []
    repro = report.get("reproducibility") or {}
    refs = report.get("references") or []
    confidence = report.get("confidence", None)

    parts: List[str] = []
    parts.append(f"## Technical Summary\n{summary}\n")

    if metrics:
        parts.append("### Metrics\n" + _fmt_metrics(metrics))

    parts.append(_fmt_list("Key Findings", key_findings))
    parts.append(_fmt_equations(equations))
    parts.append(_fmt_list("Statistical Tests", stats))
    parts.append(_fmt_list("Assumptions", assumptions))
    parts.append(_fmt_list("Limitations", limitations))
    parts.append(_fmt_repro(repro))
    parts.append(_fmt_references(refs))

    if confidence is not None:
        parts.append(f"_Confidence: {confidence:.2f}_")

    # Join and strip extra blank lines
    out = "\n".join(p for p in parts if p).strip()
    return out or "No technical details available."
