import os
from dotenv import load_dotenv
import faiss
import numpy as np
from openai import OpenAI

# Load variables from .env file
load_dotenv()

API_KEY     = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"

# Define data directory path
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SENT_FILE   = os.path.join(DATA_DIR, "kk_sentences.txt")
INDEX_FILE  = os.path.join(DATA_DIR, "kk_faiss.index")

# 1. Инициализация клиента
client = OpenAI(api_key=API_KEY)

# 2. Чтение корпуса
with open(SENT_FILE, encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# 3. Генерация эмбеддингов батчами
embeddings = []
batch_size = 200
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    embeddings.extend(d.embedding for d in resp.data)
    print(f"Обработано {min(i+batch_size, len(texts))}/{len(texts)} строк")

# 4. Построение FAISS‑индекса
vectors = np.array(embeddings, dtype="float32")
d = vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(vectors)
print(f"Индекс построен: всего {index.ntotal} векторов")

# 5. Сохранение индекса
faiss.write_index(index, INDEX_FILE)
print(f"Индекс сохранён: {INDEX_FILE}")
