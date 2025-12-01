# imports
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from websocietysimulator.llm import LLMBase

class GeminiEmbeddingModel:
    """Wrapper for Gemini embeddings, compatible with LangChain/Chroma."""
    def __init__(self, client: genai.Client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (list of strings)."""
        vectors: List[List[float]] = []
        # Batching (max 100) is recommended for production, but loop is safe for small N
        for t in texts:
            if not t:
                vectors.append([])
                continue
            try:
                # Task type 'RETRIEVAL_DOCUMENT' is optimal for indexing
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=t,
                    config={'task_type': 'RETRIEVAL_DOCUMENT'}
                )
                vectors.append(resp.embeddings[0].values)
            except Exception as e:
                print(f"Embedding error: {e}")
                vectors.append([]) 
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        try:
            # Task type 'RETRIEVAL_QUERY' is optimal for search queries
            resp = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={'task_type': 'RETRIEVAL_QUERY'}
            )
            return resp.embeddings[0].values
        except Exception as e:
            print(f"Embedding query error: {e}")
            return []

class GeminiLLM(LLMBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash", 
        embedding_model: str = "text-embedding-004",
    ):
        super().__init__(model=model)

        load_dotenv()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set and no api_key passed to GeminiLLM")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.embedding_model_name = embedding_model
        self._embedding_model = GeminiEmbeddingModel(self.client, model=embedding_model)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> str:
        model_name = model or self.model_name
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            gemini_role = "model" if role == "assistant" else role
            contents.append({
                "role": gemini_role,
                "parts": [{"text": m.get("content", "")}],
            })

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            # Robust parsing for safety blocks or empty responses
            try:
                if response.text:
                    return response.text
            except Exception:
                pass

            if response.candidates:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and 
                    candidate.content and 
                    hasattr(candidate.content, 'parts') and 
                    candidate.content.parts):
                    return "".join(getattr(p, "text", "") for p in candidate.content.parts)
            
            print(f"Warning: Gemini returned empty response. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'Unknown') if response.candidates else 'No candidates'}")
            return ""

        except Exception as e:
            print("Error calling Gemini:", repr(e))
            return ""

    def get_embedding_model(self):
        return self._embedding_model