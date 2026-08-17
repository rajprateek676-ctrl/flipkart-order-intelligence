
import json
from retrieval import search
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
key=json.loads((ROOT/"part3/retrieval_eval_key.json").read_text())
rows=[]
for item in key:
    hits=search(item["query"],3)
    docs=[]
    for h in hits:
        if h["document"] not in docs: docs.append(h["document"])
    rel=set(item["relevant_documents"])
    tp=len(set(docs[:3]) & rel)
    precision=tp/3
    recall=tp/len(rel)
    rows.append((item["query"],item["relevant_documents"],docs[:3],precision,recall))
avgp=sum(x[3] for x in rows)/len(rows); avgr=sum(x[4] for x in rows)/len(rows)
out=["RETRIEVAL EVALUATION (DOCUMENT LEVEL)",""]
for q,r,d,p,rec in rows:
    out += [f"Query: {q}",f"Relevant: {r}",f"Top-3 documents: {d}",f"Precision@3 = {p:.4f}",f"Recall@3 = {rec:.4f}",""]
out += [f"Average Precision@3 = {avgp:.4f}",f"Average Recall@3 = {avgr:.4f}"]
(ROOT/"part3/retrieval_evaluation.txt").write_text("\n".join(out))
print("\n".join(out))
