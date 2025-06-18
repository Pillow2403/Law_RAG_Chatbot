from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Load metadata
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embeddings
data = torch.load("embeddings.pt")
keys, embs = data["keys"], data["embs"]

# Load lại model tương thích
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Khởi tạo API
app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_luat(req: SearchRequest):
    q_emb = model.encode([req.query], convert_to_tensor=True)
    sims = F.cosine_similarity(q_emb, embs)
    top = torch.topk(sims, k=req.top_k)

    results = []
    for score, idx in zip(top.values, top.indices):
        key  = keys[idx]
        m    = metadata[key]
        results.append({
            "key": key,
            "dieu": m["dieu"],
            "title": m["title"],
            "score": float(score),
            "text": m["text"]
        })
    
    if not results:
        raise HTTPException(status_code=404, detail="Không tìm thấy điều luật phù hợp.")
    return results