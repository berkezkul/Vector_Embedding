import openai
from typing import List
from fastapi import HTTPException
from openai import OpenAI



class Embedder:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key
        #self.client = OpenAI(api_key=api_key)# OpenAI api_key'i ayarla

    def generate_embeddings(self, texts: List[str]) -> List[dict]:
        """OpenAI API kullanarak metinler için gömme vektörler üretir."""
        try:
            # OpenAI ile gömme vektörleri oluştur

            embeddings_responses = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=texts  # Birden fazla metin gönderiliyor
            ).data[0]
            # Yanıttan gerekli bilgiyi çek

            embeddings = [response for response in embeddings_responses]
            if len(embeddings) == 0:
                print("embeddings bozuk")
            # Qdrant'ın beklediği formata dönüştür
            qdrant_embeddings = [{'vector': embedding} for embedding in embeddings]
            return qdrant_embeddings
        except openai.OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
