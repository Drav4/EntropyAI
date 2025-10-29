from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

Number = Union[int, float]

class Metric(BaseModel):
    name: str
    value: Union[Number, str]
    unit: Optional[str] = None
    how_computed: Optional[str] = None

class Reference(BaseModel):
    title: str
    source: str
    url_or_doi: Optional[str] = None
    accessed: Optional[str] = None

class EvidenceReport(BaseModel):
    summary: str = Field(..., description="Short, precise, technical summary")
    key_findings: List[str] = Field(default_factory=list)
    metrics: List[Metric] = Field(default_factory=list)
    equations: List[str] = Field(default_factory=list)
    statistical_tests: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    reproducibility: Dict[str, Any] = Field(default_factory=dict)
    references: List[Reference] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
