from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.audit import AuditRequest, AuditResponse, TokenMetrics
from app.services.tokenizer_service import tokenizer_service
from app.services.evaluator_service import evaluator_service

app = FastAPI(
    title="TokenLens Engine API",
    description="Servicio de tokenización y auditoría de prompts basado en heurísticas de Lee Boonstra",
    version="1.0.0"
)

# Habilitar CORS para conectar con el frontend React / Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TokenLens Engine"}

@app.post("/api/audit", response_model=AuditResponse)
async def audit_prompt(request: AuditRequest):
    try:
        # 1. Tokenización del prompt original
        orig_count, orig_tokens = tokenizer_service.count_and_slice(request.prompt, request.target_model)
        orig_char_count = len(request.prompt)
        metrics_original = TokenMetrics(
            token_count=orig_count,
            char_count=orig_char_count,
            char_to_token_ratio=round(orig_char_count / max(1, orig_count), 2),
            sample_tokens=orig_tokens
        )

        # 2. Evaluación con Gemini 2.0 Flash (LLM-as-a-Judge)
        judge_output = await evaluator_service.evaluate_prompt(request.prompt)

        # 3. Métricas de las versiones optimizadas
        opt_es_count, opt_es_tokens = tokenizer_service.count_and_slice(
            judge_output.optimized_prompt_es, request.target_model
        )
        metrics_opt_es = TokenMetrics(
            token_count=opt_es_count,
            char_count=len(judge_output.optimized_prompt_es),
            char_to_token_ratio=round(len(judge_output.optimized_prompt_es) / max(1, opt_es_count), 2),
            sample_tokens=opt_es_tokens
        )

        opt_en_count, opt_en_tokens = tokenizer_service.count_and_slice(
            judge_output.optimized_prompt_en, request.target_model
        )
        metrics_opt_en = TokenMetrics(
            token_count=opt_en_count,
            char_count=len(judge_output.optimized_prompt_en),
            char_to_token_ratio=round(len(judge_output.optimized_prompt_en) / max(1, opt_en_count), 2),
            sample_tokens=opt_en_tokens
        )

        # 4. Cálculo del porcentaje neto de ahorro de tokens en inglés
        if metrics_original.token_count > 0:
            savings = round(
                ((metrics_original.token_count - metrics_opt_en.token_count) / metrics_original.token_count) * 100, 
                1
            )
        else:
            savings = 0.0

        return AuditResponse(
            acceptability_percentage=judge_output.acceptability_percentage,
            category=judge_output.category,
            metrics_original=metrics_original,
            metrics_optimized_es=metrics_opt_es,
            metrics_optimized_en=metrics_opt_en,
            token_savings_en_percentage=savings,
            checklist=judge_output.checklist,
            token_waste_found=judge_output.token_waste_found,
            critical_flaws=judge_output.critical_flaws,
            optimized_prompt_es=judge_output.optimized_prompt_es,
            optimized_prompt_en=judge_output.optimized_prompt_en,
            optimization_notes=judge_output.optimization_notes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))