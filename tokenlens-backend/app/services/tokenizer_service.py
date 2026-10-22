import tiktoken
from typing import List, Tuple
from google import genai
from app.config import settings

class TokenizerService:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.openai_encoding = tiktoken.get_encoding("cl100k_base")

    def count_and_slice(self, text: str, model: str = "gemini-2.5-flash") -> Tuple[int, List[str]]:
        if not text.strip():
            return 0, []

        if "gemini" in model:
            # Conteo oficial vía Gemini SDK
            try:
                response = self.gemini_client.models.count_tokens(
                    model=model,
                    contents=text
                )
                count = response.total_tokens
            except Exception:
                # Estimación fallback heurística para SentencePiece (~4 caracteres por token)
                count = max(1, round(len(text) / 4))

            # Segmentación de visualización (simulación de subwords por BPE)
            tokens_visual = self._slice_tokens_tiktoken(text)
            return count, tokens_visual[:40]

        else:
            # Tokenización determinista con tiktoken para OpenAI
            tokens = self.openai_encoding.encode(text)
            count = len(tokens)
            tokens_visual = [
                self.openai_encoding.decode([t]) for t in tokens[:40]
            ]
            return count, tokens_visual

    def _slice_tokens_tiktoken(self, text: str) -> List[str]:
        tokens = self.openai_encoding.encode(text)
        return [self.openai_encoding.decode([t]) for t in tokens]

tokenizer_service = TokenizerService()