
import re, json
from pathlib import Path
from part3.tools import check_return_risk, classify_product_image
from part3.retrieval import search

INJECTION=[r"ignore previous instructions",r"ignore all rules",r"pretend you are"]
GROUND_THRESHOLD=0.35

def intent(text):
    t=text.lower()
    if any(x in t for x in ["return risk","risk of return","likely to return","high risk","return probability"]): return "return_risk"
    if any(x in t for x in ["classify","product category","what category","image"]): return "product_category"
    return "policy"

def guarded(text):
    return any(re.search(p,text,re.I) for p in INJECTION)

def mock_answer(answer,source,confidence):
    return {"answer":answer,"source":source,"confidence":round(float(confidence),3)}

def run(text, state=None, order_features=None, image_path=None):
    state = {} if state is None else state
    if guarded(text):
        return mock_answer("I can’t follow that instruction. I can help with supported Flipkart questions.", "policy_kb", 1.0), state
    route=intent(text)
    state["last_intent"]=route
    if route=="return_risk":
        if not order_features:
            return mock_answer("Please provide the order features needed for return-risk scoring.","return_risk_tool",0.0),state
        out=check_return_risk(order_features); state["last_risk"]=out
        return mock_answer(f"Predicted return probability is {out['return_probability']:.3f}, bucket {out['risk_bucket']}.","return_risk_tool",0.9),state
    if route=="product_category":
        if not image_path:
            return mock_answer("Please provide a committed sample image path.","image_classifier_tool",0.0),state
        out=classify_product_image(image_path); state["last_category"]=out
        return mock_answer(f"The predicted category is {out['category']} with confidence {out['confidence']:.3f}.","image_classifier_tool",out["confidence"]),state
    hits=search(text,3)
    top=hits[0] if hits else None
    if not top or top["score"]<GROUND_THRESHOLD:
        return mock_answer(f"I can’t answer that from the policy knowledge base. Best similarity={top['score']:.3f} and threshold={GROUND_THRESHOLD:.2f}.","policy_kb",0.0),state
    state["last_policy_document"]=top["document"]
    return mock_answer(top["text"],"policy_kb",top["score"]),state

if __name__=="__main__":
    print(json.dumps(run("What is the return window for apparel?")[0],indent=2))
