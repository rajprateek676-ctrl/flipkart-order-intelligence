
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "orders_dataset.csv")
X = df.drop(columns=["returned", "order_id"])
y = df["returned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model_path = ROOT / "models" / "return_risk_model.pkl"
model = joblib.load(model_path)
rf_prob = model.predict_proba(X_test)[:, 1]

# Dataset verification
lines = [
    "PART 1 — RETURN-RISK EVALUATION",
    "",
    f"Rows: {len(df)}",
    f"Columns: {len(df.columns)}",
    f"Overall return rate: {df.returned.mean():.4f} ({df.returned.mean()*100:.2f}%)",
    f"Missing rating_given: {df.rating_given.isna().mean():.4f} ({df.rating_given.isna().mean()*100:.2f}%)",
    "",
    "Return rate by product_category:",
    df.groupby("product_category").returned.mean().round(4).to_string(),
    "",
    "Return rate by payment_method:",
    df.groupby("payment_method").returned.mean().round(4).to_string(),
]

cod_missing = df.loc[df.payment_method=="COD","rating_given"].isna().mean()
noncod_missing = df.loc[df.payment_method!="COD","rating_given"].isna().mean()
lines += [
    "",
    f"rating_given missing rate for COD: {cod_missing:.4f} ({cod_missing*100:.2f}%)",
    f"rating_given missing rate for non-COD: {noncod_missing:.4f} ({noncod_missing*100:.2f}%)",
    "Missingness classification: MAR.",
    "Justification: missingness is generated from the observed payment_method (22% for COD versus 6% otherwise), not from the unobserved rating value itself.",
    "",
    "Saved-model RF ROC-AUC:",
    f"{roc_auc_score(y_test, rf_prob):.6f}",
]

# RF threshold from its own probabilities
thresholds = np.arange(0.10, 0.901, 0.01)
rows = []
for t in thresholds:
    pred = (rf_prob >= t).astype(int)
    rows.append({
        "threshold": round(float(t), 2),
        "f1": f1_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
    })
ts = pd.DataFrame(rows)
best = ts.loc[ts.f1.idxmax()]
lines += [
    "",
    "Random Forest threshold sweep:",
    ts.to_string(index=False),
    "",
    f"t*_rf = {best.threshold:.2f}",
    f"t*_rf F1 = {best.f1:.4f}",
    f"t*_rf recall = {best.recall:.4f}",
    f"t*_rf precision = {best.precision:.4f}",
]

# Feature importances: aggregate one-hot columns back to original feature names.
pre = model.named_steps["pre"]
rf = model.named_steps["model"]
feature_names = pre.get_feature_names_out()
raw_importance = pd.Series(rf.feature_importances_, index=feature_names)
agg = {}
for name, value in raw_importance.items():
    original = name.split("__", 1)[-1]
    if original.startswith("product_category_"):
        original = "product_category"
    elif original.startswith("payment_method_"):
        original = "payment_method"
    agg[original] = agg.get(original, 0.0) + float(value)
imp = pd.Series(agg).sort_values(ascending=False)
lines += ["", "Top-5 impurity feature importance (aggregated to original columns):",
          imp.head(5).round(6).to_string()]

perm = permutation_importance(
    model, X_test, y_test, scoring="roc_auc",
    n_repeats=30, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": X_test.columns,
    "permutation_mean_drop": perm.importances_mean,
    "permutation_std": perm.importances_std
}).sort_values("permutation_mean_drop", ascending=False)
lines += ["", "Permutation importance on held-out test split:",
          perm_df.round(6).to_string(index=False)]

# Subgroup metrics at t*_rf
pred = (rf_prob >= float(best.threshold)).astype(int)
test_view = X_test.copy()
test_view["y"] = y_test.to_numpy()
test_view["pred"] = pred

for col in ["product_category", "payment_method"]:
    rows = []
    for key, g in test_view.groupby(col):
        rows.append({
            col: key,
            "n": len(g),
            "recall": recall_score(g.y, g.pred, zero_division=0),
            "precision": precision_score(g.y, g.pred, zero_division=0),
        })
    lines += ["", f"Subgroup metrics by {col}:", pd.DataFrame(rows).round(4).to_string(index=False)]

(ROOT/"part1"/"evaluation_results.txt").write_text("\n".join(lines))
(ROOT/"part1"/"threshold_sweep.csv").write_text(ts.to_csv(index=False))
(ROOT/"part1"/"permutation_importance.csv").write_text(perm_df.to_csv(index=False))
print(ROOT/"part1"/"evaluation_results.txt")
