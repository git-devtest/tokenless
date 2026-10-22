from pydantic import BaseModel, Field
from typing import List, Literal

class AuditRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Texto del prompt a auditar")
    target_model: Literal["gemini-2.5-flash", "gpt-4o", "gpt-4o-mini"] = "gemini-2.5-flash"

class TokenMetrics(BaseModel):
    token_count: int
    char_count: int
    char_to_token_ratio: float
    sample_tokens: List[str] = Field(default_factory=list, description="Muestra visual de tokens segmentados")

class ChecklistItem(BaseModel):
    criterion: str
    passed: bool
    feedback: str

class LLMJudgeOutput(BaseModel):
    """Esquema para Structured Outputs con Gemini 2.0 Flash"""
    acceptability_percentage: float = Field(..., ge=0.0, le=100.0)
    category: Literal["Excelente (≥90%)", "Aceptable (75-89%)", "Deficiente (<75%)"]
    checklist: List[ChecklistItem]
    token_waste_found: List[str]
    critical_flaws: List[str]
    optimized_prompt_es: str
    optimized_prompt_en: str
    optimization_notes: str

class AuditResponse(BaseModel):
    acceptability_percentage: float
    category: str
    metrics_original: TokenMetrics
    metrics_optimized_es: TokenMetrics
    metrics_optimized_en: TokenMetrics
    token_savings_en_percentage: float
    checklist: List[ChecklistItem]
    token_waste_found: List[str]
    critical_flaws: List[str]
    optimized_prompt_es: str
    optimized_prompt_en: str
    optimization_notes: str