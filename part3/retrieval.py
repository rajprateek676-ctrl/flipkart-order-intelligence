
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[1]
KB=ROOT/"part3/knowledge_base"
INDEX=ROOT/"part3/vector.index"; META=ROOT/"part3/metadata.json"

def build():
    from sentence_transformers import SentenceTransformer
    import faiss
    texts=[]; docs=[]
    for f in sorted(KB.glob("*.txt")):
        for s in f.read_text().split(". "):
            s=s.strip()
            if s:
                texts.append(s); docs.append(f.name)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    emb=model.encode(texts,normalize_embeddings=True)
    idx=faiss.IndexFlatIP(emb.shape[1]); idx.add(np.asarray(emb,dtype="float32"))
    faiss.write_index(idx,str(INDEX))
    META.write_text(json.dumps({"texts":texts,"documents":docs},indent=2))
    return len(texts)

def search(query,k=3):
    from sentence_transformers import SentenceTransformer
    import faiss
    idx=faiss.read_index(str(INDEX)); meta=json.loads(META.read_text())
    model=SentenceTransformer("all-MiniLM-L6-v2")
    q=model.encode([query],normalize_embeddings=True).astype("float32")
    scores, ids=idx.search(q,k)
    return [{"text":meta["texts"][int(i)],"document":meta["documents"][int(i)],"score":float(s)}
            for s,i in zip(scores[0],ids[0]) if int(i)>=0]
