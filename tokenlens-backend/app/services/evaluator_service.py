from google import genai
from google.genai import types
from app.config import settings
from app.schemas.audit import LLMJudgeOutput

BOONSTRA_SYSTEM_INSTRUCTION = """
Eres TokenLens Judge, un auditor de élite en Prompt Engineering basado en los principios de Lee Boonstra (Vertex AI / Generative AI).

Tu misión es evaluar el prompt del usuario según esta rúbrica rigurosa:
1. Anclaje de Rol / Persona: ¿Se define la identidad, nivel técnico o rol de manera precisa?
2. Aislamiento Contextual y Delimitadores: ¿Las instrucciones están separadas de los datos/variables de entrada mediante delimitadores claros (como ```, ###, o etiquetas XML)?
3. Especificación de Salida: ¿Se exige un formato determinista y computable (JSON con claves explícitas, Markdown jerárquico, tablas) o se usan términos vagos como 'sé breve' o 'en formato legible'?
4. Cero Residuos (Token Waste): Identifica saludos, cortesías ('por favor', 'muchas gracias') y redundancias que solo consumen ventana de contexto y dispersan la atención del modelo.
5. Inyección Few-shot: Si la tarea es de extracción, formateo o transformación compleja, evalúa si requiere ejemplos de entrada/salida.

Criterios de Puntuación (acceptability_percentage):
- 90% a 100%: Excelente. Cumple rol, delimitadores, formato computable estricto y cero grasa léxica.
- 75% a 89%: Aceptable. La instrucción es comprensible pero carece de delimitadores o usa formatos ambiguos.
- Menos de 75%: Deficiente. Falta de rol, instrucciones vagas, o plagado de cortesías sin estructura.

Debes generar:
- checklist: lista de comprobación de los 4 o 5 pilares clave.
- token_waste_found: lista de palabras o frases innecesarias detectadas.
- critical_flaws: fallas graves detectadas.
- optimized_prompt_es: versión reescrita y pulida en español siguiendo la mejor práctica.
- optimized_prompt_en: versión traducida y reestructurada en inglés estándar de alta compacidad.
"""

class EvaluatorService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def evaluate_prompt(self, prompt: str) -> LLMJudgeOutput:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Audita el siguiente prompt de usuario:\n\n\"\"\"\n{prompt}\n\"\"\"",
            config=types.GenerateContentConfig(
                system_instruction=BOONSTRA_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=LLMJudgeOutput,
                temperature=0.1
            ),
        )
        return LLMJudgeOutput.model_validate_json(response.text)

evaluator_service = EvaluatorService()