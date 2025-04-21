import os
import faiss
import numpy as np
from openai import OpenAI

# Чтение путей из окружения
INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
SENT_FILE  = os.getenv("SENTENCES_PATH")

if not INDEX_PATH or not SENT_FILE:
    raise RuntimeError("Set FAISS_INDEX_PATH and SENTENCES_PATH in .env")

# 1. Загружаем индекс
INDEX = faiss.read_index(INDEX_PATH)  # :contentReference[oaicite:0]{index=0}

# 2. Загружаем предложения корпуса
with open(SENT_FILE, encoding="utf-8") as f:
    SENTENCES = [line.strip() for line in f if line.strip()]

# 3. Инициализируем OpenAI клиент
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_relevant_examples(text: str, k: int = 3) -> list[str]:
    """
    Возвращает k наиболее релевантных предложений из корпуса.
    """
    # 3.1 Запрос эмбеддинга
    resp = client.embeddings.create(
        model="text-embedding-3-small",     # дешевле; 1536‑д
        input=[text]                         # единичный элемент
    )
    emb = resp.data[0].embedding            # 

    # 3.2 Поиск в FAISS
    D, I = INDEX.search(np.array([emb], dtype="float32"), k)  # :contentReference[oaicite:1]{index=1}

    return [SENTENCES[i] for i in I[0]]
