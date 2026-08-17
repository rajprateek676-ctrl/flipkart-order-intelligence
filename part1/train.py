import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"orders_dataset.csv")
X = df.drop(columns=["returned","order_id"])
y = df["returned"]

num = X.select_dtypes(include=np.number).columns.tolist()
cat = ["product_category","payment_method"]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), num),
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                      ("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

dummy=Pipeline([("pre",pre),("model",DummyClassifier(strategy="most_frequent",random_state=42))])
dummy.fit(Xtr,ytr)
print("Dummy accuracy",accuracy_score(yte,dummy.predict(Xte)))
print("Dummy F1",f1_score(yte,dummy.predict(Xte),zero_division=0))

lr=Pipeline([("pre",pre),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))])
lr.fit(Xtr,ytr)
prob=lr.predict_proba(Xte)[:,1]
pred=(prob>=.5).astype(int)
print("LR",accuracy_score(yte,pred),f1_score(yte,pred),recall_score(yte,pred),precision_score(yte,pred),roc_auc_score(yte,prob))

best_t=max(np.arange(.1,.901,.01), key=lambda t:f1_score(yte,(prob>=t).astype(int),zero_division=0))
print("LR best threshold",best_t)
for t in [.1,best_t,.5,.9]:
    q=(prob>=t).astype(int)
    print(t, f1_score(yte,q,zero_division=0), recall_score(yte,q,zero_division=0), precision_score(yte,q,zero_division=0))

rf=Pipeline([("pre",pre),("model",RandomForestClassifier(class_weight="balanced",random_state=42,n_jobs=-1))])
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
grid=GridSearchCV(rf,{"model__n_estimators":[100,200],"model__max_depth":[6,10,None]},
                  scoring="roc_auc",cv=cv,n_jobs=-1)
grid.fit(Xtr,ytr)
print("RF best params",grid.best_params_)
print("RF CV ROC-AUC",grid.best_score_)
rfbest=grid.best_estimator_
rfprob=rfbest.predict_proba(Xte)[:,1]
print("RF test ROC-AUC",roc_auc_score(yte,rfprob))
rf_t=max(np.arange(.1,.901,.01), key=lambda t:f1_score(yte,(rfprob>=t).astype(int),zero_division=0))
print("t*_rf",rf_t)

(Path(ROOT)/"models").mkdir(parents=True,exist_ok=True)
joblib.dump(rfbest,Path(ROOT)/"models"/"return_risk_model.pkl")
(Path(ROOT)/"models"/"t_rf.txt").write_text(str(rf_t))
(Path(ROOT)/"part1"/"models").mkdir(parents=True,exist_ok=True)
joblib.dump(rfbest,Path(ROOT)/"part1"/"models"/"return_risk_model.pkl")
(Path(ROOT)/"part1"/"t_rf.txt").write_text(str(rf_t))
