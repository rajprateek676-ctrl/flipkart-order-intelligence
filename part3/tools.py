
import joblib, pandas as pd, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
CLASSES=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def check_return_risk(order_features:dict)->dict:
    model=joblib.load(ROOT/"models/return_risk_model.pkl")
    p=float(model.predict_proba(pd.DataFrame([order_features]))[0,1])
    t=float((ROOT/"models/t_rf.txt").read_text())
    bucket="Low" if p<t else ("High" if p>=t+0.15 else "Medium")
    return {"return_probability":round(p,6),"risk_bucket":bucket,"threshold_rf":t}

def classify_product_image(image_path:str)->dict:
    model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,10)
    state=ROOT/"models/product_classifier.pt"
    model.load_state_dict(torch.load(state,map_location="cpu")); model.eval()
    tfm=transforms.Compose([transforms.Grayscale(3),transforms.Resize((224,224)),transforms.ToTensor(),
                            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with torch.no_grad():
        p=torch.softmax(model(tfm(Image.open(image_path)).unsqueeze(0)),1)[0]
    i=int(p.argmax())
    return {"category":CLASSES[i],"confidence":round(float(p[i]),6)}
